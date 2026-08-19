#!/usr/bin/env python3
from __future__ import annotations

import configparser
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ConfigFile.ini"
LOGS_DIR = ROOT / "Logs"
OUT_DIR = ROOT / "files" / "corrupt"

# Fixed topology for this sweep.
N_LAYERS = 3
MIXES_PER_LAYER = 10

# Threat model: total number of corrupt mixes in the network.
CORRUPT_MIXES = [0, 3, 6]

# Repetitions per (corrupt_mixes, scenario) pair. Client.py calls np.random.seed()
# with no argument, so each repetition draws a fresh network/traffic sample.
REPEATS = 1

SCENARIOS = {
    "client_dummies": {
        "client_dummies": True,
        "link_based_dummies": False,
        "multiple_hop_dummies": False,
    },
    "link_based_dummies": {
        "client_dummies": False,
        "link_based_dummies": True,
        "multiple_hop_dummies": False,
    },
    "multiple_hop_dummies": {
        "client_dummies": False,
        "link_based_dummies": False,
        "multiple_hop_dummies": True,
    },
}

FIELDNAMES = [
    "run_timestamp",
    "scenario",
    "repetition",
    "n_layers",
    "l_mixes_per_layer",
    "corrupt_mixes",
    "balanced_corruption",
    "client_dummies",
    "link_based_dummies",
    "multiple_hop_dummies",
    "sent_msgs",
    "received_msgs",
    "dummy_msgs_total",
    "dummy_types_breakdown",
    "mean_entropy",
    "avg_msgs_per_link",
    "avg_real_per_link",
    "avg_dummy_per_link",
    "rate_per_link_per_time",
    "entropy",
    "targets",
    "sent",
    "received",
    "dummy",
    "link_load",
    "link_summary",
]


def bool_str(v: bool) -> str:
    return "True" if v else "False"


def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    return cfg


def write_config(cfg: configparser.ConfigParser) -> None:
    with CONFIG_PATH.open("w") as f:
        cfg.write(f)


def run_main() -> None:
    # Uses current python interpreter (works inside your venv too).
    subprocess.run([sys.executable, "main.py"], cwd=ROOT, check=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def snapshot_outputs(scenario: str, corrupt: int, rep: int) -> dict[str, str]:
    """
    Copy generated Logs/* files into files/corrupt/ with unique names.
    Every run overwrites the same names in Logs/, so this must happen
    before the next run starts.
    """
    prefix = f"{scenario}_L{N_LAYERS}_M{MIXES_PER_LAYER}_C{corrupt}_R{rep:02d}"

    sources = {
        "entropy": LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_player_Entropy.csv",
        "targets": LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_player_Targets.csv",
        "sent": LOGS_DIR / "SentMessages.csv",
        "received": LOGS_DIR / "ReceivedMessages.csv",
        "dummy": LOGS_DIR / "DummyMessages.csv",
        "link_load": LOGS_DIR / "LinkLoad.csv",
        "link_summary": LOGS_DIR / "LinkSummary.csv",
    }

    suffixes = {
        "entropy": "Entropy",
        "targets": "Targets",
        "sent": "SentMessages",
        "received": "ReceivedMessages",
        "dummy": "DummyMessages",
        "link_load": "LinkLoad",
        "link_summary": "LinkSummary",
    }

    outputs: dict[str, str] = {}
    for key, src in sources.items():
        dst = OUT_DIR / f"{prefix}_{suffixes[key]}.csv"
        copy_if_exists(src, dst)
        outputs[key] = str(dst) if dst.exists() else ""

    return outputs


def compute_run_metrics(outputs: dict[str, str]) -> dict[str, str]:
    sent_count = 0
    received_count = 0
    dummy_count = 0
    dummy_breakdown: dict[str, int] = {}
    mean_entropy = 0.0
    avg_per_link = 0.0
    avg_real_per_link = 0.0
    avg_dummy_per_link = 0.0
    rate_per_link = 0.0

    if outputs.get("sent"):
        df_sent = pd.read_csv(outputs["sent"])
        sent_count = len(df_sent)

    if outputs.get("received"):
        df_recv = pd.read_csv(outputs["received"])
        received_count = len(df_recv)

    if outputs.get("dummy"):
        df_dummy = pd.read_csv(outputs["dummy"])
        dummy_count = len(df_dummy)
        if "DummyType" in df_dummy.columns and not df_dummy.empty:
            counts = df_dummy["DummyType"].value_counts(dropna=False).to_dict()
            dummy_breakdown = {str(k): int(v) for k, v in counts.items()}

    if outputs.get("entropy"):
        df_entropy = pd.read_csv(outputs["entropy"])
        if "Entropy" in df_entropy.columns and not df_entropy.empty:
            mean_entropy = float(df_entropy["Entropy"].mean())

    if outputs.get("link_summary"):
        df_links = pd.read_csv(outputs["link_summary"])
        # aggregate_link_load() appends a global mix-mix row tagged ALL->ALL.
        overall = df_links[df_links["FromLayer"].astype(str) == "ALL"]
        if not overall.empty:
            row = overall.iloc[0]
            avg_per_link = float(row["AvgMsgsPerLink"])
            avg_real_per_link = float(row["AvgRealPerLink"])
            avg_dummy_per_link = float(row["AvgDummyPerLink"])
            rate_per_link = float(row["RatePerLinkPerTime"])

    return {
        "sent_msgs": str(sent_count),
        "received_msgs": str(received_count),
        "dummy_msgs_total": str(dummy_count),
        "dummy_types_breakdown": json.dumps(dummy_breakdown, sort_keys=True),
        "mean_entropy": f"{mean_entropy:.8f}",
        "avg_msgs_per_link": f"{avg_per_link:.8f}",
        "avg_real_per_link": f"{avg_real_per_link:.8f}",
        "avg_dummy_per_link": f"{avg_dummy_per_link:.8f}",
        "rate_per_link_per_time": f"{rate_per_link:.8f}",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup config so it gets restored even if a run fails.
    original_config_text = CONFIG_PATH.read_text()

    manifest_path = OUT_DIR / f"summary_corrupt_{run_ts}.csv"
    total_runs = len(CORRUPT_MIXES) * len(SCENARIOS) * REPEATS
    run_index = 0
    started = time.time()

    print(
        f"Sweep: n_layers={N_LAYERS}, l_mixes_per_layer={MIXES_PER_LAYER}, "
        f"corrupt_mixes={CORRUPT_MIXES}, scenarios={list(SCENARIOS)}, "
        f"repeats={REPEATS} -> {total_runs} runs"
    )

    try:
        # Written incrementally so a crash partway through still leaves a usable manifest.
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            f.flush()

            for corrupt in CORRUPT_MIXES:
                for scenario_name, flags in SCENARIOS.items():
                    for rep in range(REPEATS):
                        run_index += 1

                        cfg = load_config()

                        cfg["TOPOLOGY"]["n_layers"] = str(N_LAYERS)
                        cfg["TOPOLOGY"]["l_mixes_per_layer"] = str(MIXES_PER_LAYER)

                        cfg["THREATMODEL"]["corrupt_mixes"] = str(corrupt)

                        cfg["DUMMIES"]["client_dummies"] = bool_str(flags["client_dummies"])
                        cfg["DUMMIES"]["link_based_dummies"] = bool_str(flags["link_based_dummies"])
                        cfg["DUMMIES"]["multiple_hop_dummies"] = bool_str(flags["multiple_hop_dummies"])

                        write_config(cfg)

                        print(
                            f"[{run_index}/{total_runs}] scenario={scenario_name}, "
                            f"n_layers={N_LAYERS}, l_mixes_per_layer={MIXES_PER_LAYER}, "
                            f"corrupt_mixes={corrupt}, rep={rep}",
                            flush=True,
                        )
                        run_main()

                        outputs = snapshot_outputs(scenario_name, corrupt, rep)
                        metrics = compute_run_metrics(outputs)

                        row = {
                            "run_timestamp": run_ts,
                            "scenario": scenario_name,
                            "repetition": str(rep),
                            "n_layers": str(N_LAYERS),
                            "l_mixes_per_layer": str(MIXES_PER_LAYER),
                            "corrupt_mixes": str(corrupt),
                            "balanced_corruption": cfg["THREATMODEL"]["balanced_corruption"],
                            "client_dummies": bool_str(flags["client_dummies"]),
                            "link_based_dummies": bool_str(flags["link_based_dummies"]),
                            "multiple_hop_dummies": bool_str(flags["multiple_hop_dummies"]),
                            **metrics,
                            **outputs,
                        }
                        writer.writerow(row)
                        f.flush()

    finally:
        # Always restore original config file
        CONFIG_PATH.write_text(original_config_text)

    elapsed = time.time() - started
    print(f"Done in {elapsed / 60:.1f} min. Saved outputs to: {OUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
