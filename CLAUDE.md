# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A SimPy discrete-event mixnet simulator used for research on **dummy-traffic strategies**: how client dummies, link-based dummies, and multi-hop dummies affect anonymity (entropy) versus the traffic load they put on mix-to-mix links.

No tests, no linter, no build step. Validation is "run a simulation and look at the CSVs/plots".

## Commands

```bash
source venv_mixim/bin/activate          # venv is gitignored; recreate with python3.12 -m venv --clear venv_mixim
pip install -r requirements.txt

python main.py                          # one simulation using ConfigFile.ini as-is
python run.py                           # sweep: mixes-per-layer 3..10 x 4 dummy scenarios -> files/
python run-corrupt.py                   # sweep: corrupt_mixes 0/3/6 x 3 dummy scenarios -> files/corrupt/
python plot.py                          # entropy-vs-arrival-time grid from files/*_Entropy.csv

sbatch mixim.sbatch                     # SLURM (epyc-256, 50G, 8h); recreates the venv, then runs run-corrupt.py
```

`mixim.sbatch` hardcodes which driver it runs — edit the last `python3.12 <script>` line to switch between `run.py`, `run-corrupt.py`, and `main.py`.

`plot.py` only reads the **flat** `files/` directory and silently skips anything outside `3 <= M <= 9`, so it pairs with `run.py` output, not `run-corrupt.py`'s `files/corrupt/`. Plotting a corrupt sweep means writing a new script (see `experiments/*/` for prior ones).

Simulations print one or more lines **per message per hop**. A single sweep produces tens of GB of stdout — current `mixim.sbatch` sends it to `Logs/mixim-<jobid>.out` and a `tee` copy in `Logs/mixim-detailed-<jobid>.log`; the `slurm-*.out` files in the repo root are from an older version of the script. Never `cat` any of them; use `head`/`grep`/`tail`. `printing`/`logging` are hardcoded to `True` in [main.py:90](main.py#L90).

## Architecture

Flow: `main.py` (config → derived rates) → `Simulation` (owns the SimPy env, builds `Network` + `Client`s, runs, computes entropy) → CSVs in `Logs/`.

**Every hop goes through `Attacker.relay()`** in [Relay.py](Relay.py). Nodes never call each other directly — a sender always does `env.process(simulation.attacker.relay(msg, sender, receiver))`. That single choke point applies the 0.05 link delay, marks target messages, logs link load, and checks the end-of-simulation condition. Anything that must observe all traffic belongs there.

**Entropy is computed by propagating probability vectors, not by post-hoc analysis.** Each `Message` carries `pr_target` (probability it is each target) and each `Mix` carries `Pmix` (accumulated probability in its pool). `PoissonMix.update_probabilities` splits `Pmix` evenly across the pool on flush. `Simulation.run()` then reads the final `pr_target` of every received message and computes Shannon entropy **row-wise per message**. Corrupt mixes skip `update_probabilities` entirely — that is the whole mechanism by which corruption reduces anonymity.

**A message whose probability row does not sum to ~1 gets entropy 0, not a missing value.** [Simulation.py:307](Simulation.py#L307) gates on `np.isclose(row_sum, 1.0, atol=0.3)` and writes `0.0` otherwise. Messages that arrive before the vectors stabilise, or whose mass leaked, land in the same bucket as genuinely deanonymised ones. Every downstream `mean_entropy` in the manifests averages those zeros in, so a "low entropy" result is ambiguous between real deanonymisation and unconverged rows — check the row-sum prints or the entropy distribution before drawing a conclusion.

**Targeting is unconditional, and it costs 2 time units per message.** The original design gated target selection on `startAttack` and reset `var = False` after each pick; both are commented out in [Relay.py:25](Relay.py#L25) and [Relay.py:60](Relay.py#L60). So every `Real`/`ClientDummy` message is marked a target on its first hop **and** takes the `yield self.env.timeout(2)` at [Relay.py:61](Relay.py#L61) before that hop — a 2.0 delay on a 20.0 simulation. The stable-mix machinery (`set_stable_mix`, `stableMixL1`, `PoissonMix.receive_message`'s pool-average check) still runs but no longer gates anything.

**Runtime and memory are quadratic in message count.** `pr_target` and `Pmix` are grown to `len(Log.sent_messages)` on every hop, and `update_probabilities` loops over the whole vector per message per hop. With 250 clients over 20 time units that is thousands of messages each carrying a thousands-long float list — this, not SimPy, is why a sweep needs hours and 50G. Raising `n_clients`, `simDuration`, or `rho` costs O(N²).

**Mix-generated dummies claim the *previous* message's probability index.** `PoissonMix.send_dummies` computes `total_messages` and sets `pr_target[total_messages - 1] = 1.0` *before* calling `Log.sent_messages_f`, whereas a real message is logged by `Client.send_message` before `relay()` marks it. Link/multi-hop dummies therefore alias the column of the last real message instead of owning a fresh one. Treat per-column probability mass in those scenarios as suspect; the row-wise entropy that the repo actually reports is less affected.

**`n_targets` is vestigial in target-all mode.** `Simulation.__init__` still derives and prints it, and `Mix.__init__` sizes `Pmix` from it, but `PoissonMix` immediately grows `Pmix` past that size. It only really drives the dead `Pool`/`TimedMix` paths. Likewise `link_delay = [0.01, 0.1]` at the top of [Relay.py](Relay.py#L1) is unused — the real delay is the hardcoded `0.05`.

**`# to target-all-msgs` / `# to target-fixed-msgs` comment pairs mark two mutually exclusive modes.** The repo is currently in *target-all* mode: every real/ClientDummy message becomes a target, `pr_target` vectors grow dynamically as more messages are sent, and entropy is per-message. The commented-out blocks are the older *target-fixed* mode (a fixed `n_targets` set, entropy column-wise). If you switch modes, you must flip every marked block consistently — they are spread across `Client.py`, `Mix.py`, `PoissonMix.py`, `Relay.py`, and `Simulation.py`.

**Only `PoissonMix` is current.** `Pool.py` and `TimedMix.py` still use fixed `range(self.n_targets)` loops and were never updated for target-all mode or dummy traffic; setting `mix_type = pool`/`time` will `IndexError`. Treat `mix_type = poisson` as the only working path.

**`Network` uses class-level mutable state** (`all_mixes`, `network_dict`). Safe only because the sweep drivers launch each run as a fresh subprocess. Do not run two `Simulation`s in one process.

## Configuration traps

`ConfigFile.ini` is the input, but several values in it are dead:

- **`[MIXING] mu` is ignored.** `mu` is derived as `(E2E - (n_layers+1)*0.05) / n_layers` in [main.py:29](main.py#L29) — i.e. the target end-to-end latency minus link delays, split across layers.
- **`rate_client_dummies` and `rate_mix_dummies` are ignored.** Both are derived from `rho` (dummy-to-real link-load ratio) in [main.py:62-85](main.py#L62-L85), with a different closed form per strategy: `λ_cd = ρM²/C`, `λ_ld = ρM`, `λ_md = 2Mρ/L`. Rates passed to the simulation are the *inverse* (mean inter-arrival time), because `numpy.random.exponential` takes a scale.
- **The three dummy flags are an if/elif chain**, so only the first true one takes effect. Set exactly one; all false is the baseline.
- **`simDuration=20` and `burnout=0` are hardcoded** in [main.py:90](main.py#L90) and [Simulation.py:55](Simulation.py#L55).

The sweep drivers **rewrite `ConfigFile.ini` in place** and restore it in a `finally`. A crashed/killed sweep can leave it modified — `git diff ConfigFile.ini` before trusting it. `configparser.write()` also drops every comment, so a config that went through a driver loses the inline notes listing allowed values (`type` = stratified/cascade/free route, `routing` = source/hopbyhop, ...); recover them from git history rather than reconstructing them.

Only `client_dummies` reaches clients (`Client.__init__` spawns a second `send_message` process); `link_based_dummies`/`multiple_hop_dummies` are handed to `Network` → `PoissonMix`, which starts `send_dummies` only on **non-corrupt mixes outside the last layer**. Corrupt mixes emit no dummies at all, so raising `corrupt_mixes` also lowers dummy load.

## Outputs

`Logs/` is overwritten by every run, so drivers snapshot it into `files/` between runs under `<scenario>_L<layers>_M<mixes>[_C<corrupt>_R<rep>]_<kind>.csv`, plus a `summary_*.csv` manifest with one row per run. Both `Logs/` and `files/` are gitignored.

`run-corrupt.py` writes and flushes its manifest row-by-row, so a killed sweep still leaves usable rows; `run.py` only writes its manifest after the whole sweep finishes and leaves `run_timestamp` blank. Prefer the `run-corrupt.py` shape when adding a driver.

`LinkLoad.csv` is one row per hop; `LinkSummary.csv` is `Simulation.aggregate_link_load()`'s per-layer-transition aggregate plus a global `FromLayer=ALL` row. Link counts assume a fully connected stratified topology (`M²` directed links per transition) rather than being counted from the data.

`experiments/` holds frozen copies of earlier runs — plots, their driver scripts, and a readme pinning the config used. When finishing an experiment, snapshot the scripts and config there rather than relying on the working tree.
