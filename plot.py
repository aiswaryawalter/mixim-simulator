#!/usr/bin/env python3
"""Thesis-grade analysis of campaign04 mixnet entropy and dummy traffic strategies.

Outputs (under <campaign>/analysis_plots/):
    Plots
        baseline_dip_motivation.png         A1   end-of-session dip with no dummies
        entropy_progression_delivery{X}.png B1   one PNG per delivery%, 3x3 grid
        mean_entropy_grid.png               B2   mean entropy summary
        auc_grid.png                        B3   AUC of entropy curve
        delta_auc_vs_baseline_grid.png      B4   Delta AUC vs baseline w/ CI on diff
        dummy_per_link_grid.png             C1   bandwidth cost
        extra_time_grid.png                 C2   latency cost
        pareto_entropy_vs_cost.png          C3   privacy/cost trade-off

    Tables
        summary_metrics_per_condition.csv   thesis-ready per-condition summary
        delta_vs_baseline_95ci.csv          deltas with 95% CI on the difference
        best_strategy_per_condition.csv     argmax-AUC scenario per condition
        transition_window_dip.csv           baseline dip magnitude/duration
        auc_summary_vs_baseline_95ci.csv    AUC summary
        mean_entropy_summary_95ci.csv       per-condition mean entropy
        avg_dummy_per_link_summary_95ci.csv bandwidth summary
        extra_time_summary_95ci.csv         latency summary
        run_level_auc.csv                   per-run AUC values
        entropy_run_level_long.csv          long-format entropy curves

Statistical conventions:
    - Bootstrap 95% CI on run-level data (3000 resamples by default).
    - Entropy curves smoothed for display with a centred rolling window of 25
      0.5-unit bins; the bootstrap CI is computed on raw bins.
    - AUC integrated over a fixed horizon (= configured sim duration = 20).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_DIR = ROOT / "files" / "campaign04"
DEFAULT_PLOTS_DIR = DEFAULT_CAMPAIGN_DIR / "analysis_plots"

DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_FIXED_HORIZON = 20.0

SCENARIO_ORDER = [
    "baseline",
    "client_dummies",
    "link_based_dummies",
    "multiple_hop_dummies",
]

SCENARIO_LABELS = {
    "baseline": "baseline",
    "client_dummies": "client dummies",
    "link_based_dummies": "link-based dummies",
    "multiple_hop_dummies": "multiple-hop dummies",
}

SCENARIO_COLORS = {
    "baseline": "#1f77b4",
    "client_dummies": "#2ca02c",
    "link_based_dummies": "#ff7f0e",
    "multiple_hop_dummies": "#d62728",
}

DELIVERY_COLORS = {
    80: "#440154",
    90: "#3b528b",
    95: "#21918c",
    99: "#f1c40f",
}


# ------------------------------- bootstrap helpers ------------------------------- #


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 3000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[samples].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def bootstrap_mean_diff_ci(
    values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 3000, seed: int = 0
):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if values_a.size == 0 or values_b.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_a = rng.integers(0, values_a.size, size=(n_boot, values_a.size))
    boot_b = rng.integers(0, values_b.size, size=(n_boot, values_b.size))
    boot_diff = values_a[boot_a].mean(axis=1) - values_b[boot_b].mean(axis=1)
    mean_diff = float(values_a.mean() - values_b.mean())
    lo, hi = np.percentile(boot_diff, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def scenario_sort_key(s: str):
    try:
        return (SCENARIO_ORDER.index(s), s)
    except ValueError:
        return (999, s)


def compute_auc(time_values: np.ndarray, entropy_values: np.ndarray, horizon: float):
    x = np.asarray(time_values, dtype=float)
    y = np.asarray(entropy_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.nan

    order = np.argsort(x)
    x, y = x[order], y[order]

    ux, inv = np.unique(x, return_inverse=True)
    uy = np.zeros_like(ux, dtype=float)
    cnt = np.zeros_like(ux, dtype=float)
    np.add.at(uy, inv, y)
    np.add.at(cnt, inv, 1.0)
    y, x = uy / cnt, ux
    if x.size < 2:
        return np.nan

    h = float(horizon)
    if h <= x[0]:
        return np.nan
    if h < x[-1]:
        left = x < h
        x_left, y_left = x[left], y[left]
        if x_left.size < 1:
            return np.nan
        y_h = float(np.interp(h, x, y))
        x = np.append(x_left, h)
        y = np.append(y_left, y_h)
    elif h > x[-1]:
        y_h = float(np.interp(h, x, y))
        x = np.append(x, h)
        y = np.append(y, y_h)

    if x.size < 2:
        return np.nan
    return float(np.trapz(y, x))


# ----------------------------------- loaders ----------------------------------- #


def load_combo_meta(combo_dir: Path) -> dict:
    meta_path = combo_dir / "combo_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def load_campaign_entropy_runs(
    campaign_dir: Path, run_bins_name: str = "EntropyBins_0p5.csv"
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    combo_dirs = sorted(p for p in campaign_dir.glob("combo_*") if p.is_dir())

    for combo_dir in combo_dirs:
        meta = load_combo_meta(combo_dir)
        if not meta:
            continue

        combo_id = int(meta.get("combo_id", -1))
        scenario = str(meta.get("scenario", ""))
        corrupt_mixes = int(meta.get("corrupt_mixes", -1))
        e2e = int(meta.get("E2E", -1))
        msg_delivery_percent = int(meta.get("msg_delivery_percent", -1))

        run_dirs = sorted(p for p in combo_dir.glob("run_*") if p.is_dir())
        for run_dir in run_dirs:
            bins_path = run_dir / run_bins_name
            if not bins_path.exists():
                continue
            d = pd.read_csv(bins_path)
            if d.empty or not {"time_bin_end", "avg_entropy"}.issubset(d.columns):
                continue

            d = d.copy()
            d["combo_id"] = combo_id
            d["scenario"] = scenario
            d["corrupt_mixes"] = corrupt_mixes
            d["E2E"] = e2e
            d["msg_delivery_percent"] = msg_delivery_percent

            if "repeat_idx" not in d.columns:
                rep = run_dir.name.replace("run_", "")
                try:
                    d["repeat_idx"] = int(rep)
                except ValueError:
                    d["repeat_idx"] = -1
            rows.append(d)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    numeric = [
        "time_bin_end",
        "avg_entropy",
        "combo_id",
        "corrupt_mixes",
        "E2E",
        "msg_delivery_percent",
        "repeat_idx",
    ]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=numeric)
    return out


def load_per_run_metrics(campaign_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    combo_dirs = sorted(p for p in campaign_dir.glob("combo_*") if p.is_dir())
    for combo_dir in combo_dirs:
        meta = load_combo_meta(combo_dir)
        if not meta:
            continue
        p = combo_dir / "summary_per_run.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue

        d = d.copy()
        for col, default in [
            ("combo_id", meta.get("combo_id")),
            ("scenario", meta.get("scenario", "")),
            ("corrupt_mixes", meta.get("corrupt_mixes")),
            ("E2E", meta.get("E2E")),
            ("msg_delivery_percent", meta.get("msg_delivery_percent")),
        ]:
            if col not in d.columns:
                d[col] = default

        for c in (
            "combo_id",
            "corrupt_mixes",
            "E2E",
            "msg_delivery_percent",
            "repeat_idx",
            "sim_extra_time",
            "sim_end_time",
            "mean_entropy",
            "avg_dummy_per_link",
            "avg_real_per_link",
            "avg_msgs_per_link",
        ):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        rows.append(d)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["combo_id", "corrupt_mixes", "E2E", "msg_delivery_percent"])
    return out


# --------------------------- curve summarization & smoothing -------------------- #


def summarize_curve(df: pd.DataFrame, n_boot: int = 3000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for idx, (time_bin_end, grp) in enumerate(df.groupby("time_bin_end", sort=True)):
        vals = grp["avg_entropy"].astype(float).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed + idx)
        rows.append(
            {
                "time_bin_end": float(time_bin_end),
                "mean_entropy": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": int(np.isfinite(vals).sum()),
            }
        )
    return pd.DataFrame(rows)


def smooth_for_display(curve_df: pd.DataFrame, window: int) -> pd.DataFrame:
    if curve_df.empty or window <= 1:
        return curve_df
    out = curve_df.copy().sort_values("time_bin_end").reset_index(drop=True)
    for col in ("mean_entropy", "ci95_low", "ci95_high"):
        if col in out.columns:
            out[col] = (
                out[col].rolling(window=window, center=True, min_periods=1).mean()
            )
    return out


def plot_band_on_axis(ax, curve_disp: pd.DataFrame, color, label, linewidth=2.0, alpha=0.18):
    if curve_disp.empty:
        return
    x = curve_disp["time_bin_end"].to_numpy()
    y = curve_disp["mean_entropy"].to_numpy()
    lo = curve_disp["ci95_low"].to_numpy()
    hi = curve_disp["ci95_high"].to_numpy()
    ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


# --------------------------------- A1 plot ------------------------------------- #


def plot_baseline_dip(
    curve_df: pd.DataFrame,
    plots_dir: Path,
    n_boot: int,
    smooth_window: int,
    ref_corrupt: int,
    ref_e2e: int,
):
    sub = curve_df[
        (curve_df["scenario"] == "baseline")
        & (curve_df["corrupt_mixes"] == ref_corrupt)
        & (curve_df["E2E"] == ref_e2e)
    ]
    if sub.empty:
        return None

    delivery_values = sorted(sub["msg_delivery_percent"].dropna().astype(int).unique())
    if not delivery_values:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for d_idx, delivery in enumerate(delivery_values):
        scen = sub[sub["msg_delivery_percent"] == delivery]
        if scen.empty:
            continue
        curve = summarize_curve(scen, n_boot=n_boot, seed=20000 + d_idx * 100)
        curve_disp = smooth_for_display(curve, window=smooth_window)
        color = DELIVERY_COLORS.get(int(delivery))
        plot_band_on_axis(ax, curve_disp, color=color, label=f"delivery {delivery}%")

    ax.set_xlabel("Arrival time")
    ax.set_ylabel("Entropy")
    ax.set_title(
        f"Baseline entropy with end-of-session dip "
        f"(corrupt={ref_corrupt}, E2E={ref_e2e}, no dummies)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=10, frameon=False)
    fig.tight_layout()
    out_path = plots_dir / "baseline_dip_motivation.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------- B1 plot ------------------------------------- #


def plot_entropy_progression_per_delivery(
    curve_df: pd.DataFrame,
    plots_dir: Path,
    n_boot: int,
    smooth_window: int,
) -> list[Path]:
    saved: list[Path] = []
    delivery_values = sorted(
        curve_df["msg_delivery_percent"].dropna().astype(int).unique()
    )

    for delivery in delivery_values:
        sub = curve_df[curve_df["msg_delivery_percent"] == delivery]
        if sub.empty:
            continue

        corrupt_values = sorted(sub["corrupt_mixes"].dropna().astype(int).unique())
        e2e_values = sorted(sub["E2E"].dropna().astype(int).unique())
        nrows, ncols = len(corrupt_values), len(e2e_values)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.6 * ncols, 4.2 * nrows),
            sharex=True,
            sharey=True,
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        for r, corrupt in enumerate(corrupt_values):
            for c, e2e in enumerate(e2e_values):
                ax = axes[r, c]
                cell = sub[
                    (sub["corrupt_mixes"] == corrupt) & (sub["E2E"] == e2e)
                ]
                scenarios = sorted(
                    cell["scenario"].dropna().astype(str).unique(),
                    key=scenario_sort_key,
                )
                for s_idx, scenario in enumerate(scenarios):
                    scen = cell[cell["scenario"] == scenario]
                    curve = summarize_curve(
                        scen,
                        n_boot=n_boot,
                        seed=int(delivery) * 1000 + r * 100 + c * 10 + s_idx,
                    )
                    curve_disp = smooth_for_display(curve, window=smooth_window)
                    plot_band_on_axis(
                        ax,
                        curve_disp,
                        color=SCENARIO_COLORS.get(scenario),
                        label=SCENARIO_LABELS.get(scenario, scenario),
                    )

                ax.set_title(f"corrupt={corrupt}, E2E={e2e}", fontsize=10)
                ax.grid(True, alpha=0.25)
                if r == nrows - 1:
                    ax.set_xlabel("Arrival time")
                if c == 0:
                    ax.set_ylabel("Entropy")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(4, len(labels)),
                frameon=False,
            )
        fig.suptitle(
            f"Entropy progression | delivery%={delivery}", fontsize=13, y=0.98
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = plots_dir / f"entropy_progression_delivery{delivery}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved.append(out_path)
    return saved


# ----------------------- per-condition aggregation & generic grid --------------- #


def aggregate_per_run_metric(
    per_run_df: pd.DataFrame, metric_col: str, n_boot: int = 3000, seed_base: int = 30000
) -> pd.DataFrame:
    if metric_col not in per_run_df.columns or per_run_df.empty:
        return pd.DataFrame()
    cond_cols = ["corrupt_mixes", "E2E", "msg_delivery_percent", "scenario"]
    rows = []
    grouped = per_run_df.dropna(subset=[metric_col]).groupby(cond_cols, sort=True)
    for i, (key, grp) in enumerate(grouped):
        corrupt, e2e, delivery, scenario = key
        vals = grp[metric_col].astype(float).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed_base + i)
        rows.append(
            {
                "corrupt_mixes": int(corrupt),
                "E2E": int(e2e),
                "msg_delivery_percent": int(delivery),
                "scenario": str(scenario),
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_runs": int(np.isfinite(vals).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_grid(
    metric_summary: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str,
    ylabel: str,
    filename_stem: str,
    scenarios: list[str] | None = None,
    marker: str = "o",
):
    if metric_summary.empty:
        return None

    if scenarios is None:
        scenarios = sorted(
            metric_summary["scenario"].dropna().astype(str).unique(),
            key=scenario_sort_key,
        )
    corrupt_values = sorted(
        metric_summary["corrupt_mixes"].dropna().astype(int).unique()
    )
    e2e_values = sorted(metric_summary["E2E"].dropna().astype(int).unique())
    nrows, ncols = len(corrupt_values), len(e2e_values)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), sharex=True, sharey=True
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, corrupt in enumerate(corrupt_values):
        for c, e2e in enumerate(e2e_values):
            ax = axes[r, c]
            cell = metric_summary[
                (metric_summary["corrupt_mixes"] == corrupt)
                & (metric_summary["E2E"] == e2e)
            ]
            for scenario in scenarios:
                d = cell[cell["scenario"] == scenario].sort_values(
                    "msg_delivery_percent"
                )
                if d.empty:
                    continue
                x = d["msg_delivery_percent"].to_numpy()
                y = d["mean"].to_numpy()
                lo = d["ci95_low"].to_numpy()
                hi = d["ci95_high"].to_numpy()
                color = SCENARIO_COLORS.get(scenario)
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linewidth=2.0,
                    color=color,
                    label=SCENARIO_LABELS.get(scenario, scenario),
                )
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.set_title(f"corrupt={corrupt}, E2E={e2e}", fontsize=10)
            ax.grid(True, alpha=0.25)
            if r == nrows - 1:
                ax.set_xlabel("Message delivery percent")
            if c == 0:
                ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            frameon=False,
        )
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = plots_dir / f"{filename_stem}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------- AUC core ------------------------------------ #


def compute_run_level_auc(curve_df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    cond_cols = ["corrupt_mixes", "E2E", "msg_delivery_percent"]
    run_cols = ["combo_id", "scenario", "repeat_idx"]
    rows = []
    for cond_key, cond_df in curve_df.groupby(cond_cols, sort=True):
        corrupt, e2e, delivery = cond_key
        for run_key, run_df in cond_df.groupby(run_cols, sort=False):
            combo_id, scenario, repeat_idx = run_key
            auc = compute_auc(
                run_df["time_bin_end"].to_numpy(),
                run_df["avg_entropy"].to_numpy(),
                horizon=float(horizon),
            )
            rows.append(
                {
                    "combo_id": int(combo_id),
                    "scenario": str(scenario),
                    "repeat_idx": int(repeat_idx),
                    "corrupt_mixes": int(corrupt),
                    "E2E": int(e2e),
                    "msg_delivery_percent": int(delivery),
                    "horizon": float(horizon),
                    "auc": auc,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[np.isfinite(out["auc"])]


def build_auc_summary(run_auc_df: pd.DataFrame, n_boot: int = 3000) -> pd.DataFrame:
    cond_cols = ["corrupt_mixes", "E2E", "msg_delivery_percent"]
    rows = []
    for cond_key, cond_df in run_auc_df.groupby(cond_cols, sort=True):
        corrupt, e2e, delivery = cond_key
        baseline = cond_df[cond_df["scenario"] == "baseline"]["auc"].to_numpy()
        scenarios_present = sorted(
            cond_df["scenario"].dropna().astype(str).unique(), key=scenario_sort_key
        )
        condition_rows = []
        for s_idx, scenario in enumerate(scenarios_present):
            scen_vals = cond_df[cond_df["scenario"] == scenario]["auc"].to_numpy()
            if scen_vals.size == 0:
                continue
            mean_auc, lo_auc, hi_auc = bootstrap_mean_ci(
                scen_vals,
                n_boot=n_boot,
                seed=5000
                + int(corrupt) * 100
                + int(e2e) * 10
                + int(delivery)
                + s_idx,
            )
            row = {
                "corrupt_mixes": int(corrupt),
                "E2E": int(e2e),
                "msg_delivery_percent": int(delivery),
                "scenario": scenario,
                "mean_auc": mean_auc,
                "ci95_low_auc": lo_auc,
                "ci95_high_auc": hi_auc,
                "n_runs": int(scen_vals.size),
            }
            if scenario != "baseline" and baseline.size > 0:
                delta, d_lo, d_hi = bootstrap_mean_diff_ci(
                    scen_vals,
                    baseline,
                    n_boot=n_boot,
                    seed=9000
                    + int(corrupt) * 100
                    + int(e2e) * 10
                    + int(delivery)
                    + s_idx,
                )
                row["delta_vs_baseline_auc"] = delta
                row["delta_vs_baseline_ci95_low"] = d_lo
                row["delta_vs_baseline_ci95_high"] = d_hi
                row["significant_vs_baseline_95"] = bool(
                    (d_lo > 0.0) or (d_hi < 0.0)
                )
                row["pct_delta_vs_baseline"] = (
                    float(100.0 * delta / baseline.mean())
                    if baseline.mean() != 0
                    else np.nan
                )
            else:
                row["delta_vs_baseline_auc"] = 0.0
                row["delta_vs_baseline_ci95_low"] = 0.0
                row["delta_vs_baseline_ci95_high"] = 0.0
                row["significant_vs_baseline_95"] = False
                row["pct_delta_vs_baseline"] = 0.0
            condition_rows.append(row)
        if condition_rows:
            best_idx = int(np.argmax([r["mean_auc"] for r in condition_rows]))
            for i, r in enumerate(condition_rows):
                r["best_in_condition"] = i == best_idx
            rows.extend(condition_rows)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(cond_cols + ["scenario"]).reset_index(drop=True)


# ----------------------------- AUC summary plots ------------------------------- #


def plot_auc_grid(auc_summary: pd.DataFrame, plots_dir: Path):
    if auc_summary.empty:
        return None
    df = auc_summary.rename(
        columns={
            "mean_auc": "mean",
            "ci95_low_auc": "ci95_low",
            "ci95_high_auc": "ci95_high",
        }
    )
    return plot_metric_grid(
        df,
        plots_dir,
        title="AUC of entropy over arrival time (horizon = sim duration)",
        ylabel="AUC entropy",
        filename_stem="auc_grid",
    )


def plot_delta_auc_grid(auc_summary: pd.DataFrame, plots_dir: Path):
    if auc_summary.empty:
        return None
    sub = auc_summary[auc_summary["scenario"] != "baseline"].copy()
    if sub.empty:
        return None
    df = sub.rename(
        columns={
            "delta_vs_baseline_auc": "mean",
            "delta_vs_baseline_ci95_low": "ci95_low",
            "delta_vs_baseline_ci95_high": "ci95_high",
        }
    )
    scenarios = sorted(df["scenario"].dropna().astype(str).unique(), key=scenario_sort_key)
    return plot_metric_grid(
        df,
        plots_dir,
        title="Δ AUC vs baseline (95% CI on the difference)",
        ylabel="Δ AUC vs baseline",
        filename_stem="delta_auc_vs_baseline_grid",
        scenarios=scenarios,
    )


# -------------------------------- C3: Pareto ----------------------------------- #


def plot_pareto_entropy_vs_cost(
    per_run_df: pd.DataFrame,
    auc_summary: pd.DataFrame,
    plots_dir: Path,
    n_boot: int,
):
    if per_run_df.empty or auc_summary.empty:
        return None
    cost = aggregate_per_run_metric(
        per_run_df, "avg_dummy_per_link", n_boot=n_boot, seed_base=40000
    )
    if cost.empty:
        return None

    merged = cost.merge(
        auc_summary[
            [
                "corrupt_mixes",
                "E2E",
                "msg_delivery_percent",
                "scenario",
                "mean_auc",
                "ci95_low_auc",
                "ci95_high_auc",
            ]
        ],
        on=["corrupt_mixes", "E2E", "msg_delivery_percent", "scenario"],
        how="inner",
    ).rename(
        columns={"mean": "cost_mean", "ci95_low": "cost_lo", "ci95_high": "cost_hi"}
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for scenario in sorted(
        merged["scenario"].dropna().astype(str).unique(), key=scenario_sort_key
    ):
        d = merged[merged["scenario"] == scenario]
        x = d["cost_mean"].to_numpy()
        y = d["mean_auc"].to_numpy()
        xerr = np.vstack([x - d["cost_lo"].to_numpy(), d["cost_hi"].to_numpy() - x])
        yerr = np.vstack(
            [y - d["ci95_low_auc"].to_numpy(), d["ci95_high_auc"].to_numpy() - y]
        )
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            linestyle="none",
            color=SCENARIO_COLORS.get(scenario),
            alpha=0.75,
            markersize=6,
            label=SCENARIO_LABELS.get(scenario, scenario),
            capsize=2,
            elinewidth=0.7,
        )

    ax.set_xlabel("Avg dummy messages per link (cost)")
    ax.set_ylabel("AUC entropy (anonymity benefit)")
    ax.set_title(
        "Privacy vs cost — each point = one (corrupt_mixes, E2E, delivery%) condition"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=10, frameon=False)
    fig.tight_layout()

    out_path = plots_dir / "pareto_entropy_vs_cost.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ----------------------------------- tables ------------------------------------ #


def build_summary_metrics_per_condition(
    per_run_df: pd.DataFrame,
    auc_summary: pd.DataFrame,
    plots_dir: Path,
    n_boot: int = 3000,
) -> pd.DataFrame:
    metric_cols = [
        "mean_entropy",
        "sim_extra_time",
        "avg_dummy_per_link",
        "avg_real_per_link",
        "avg_msgs_per_link",
    ]
    parts: list[pd.DataFrame] = []
    for i, m in enumerate(metric_cols):
        if m not in per_run_df.columns:
            continue
        agg = aggregate_per_run_metric(
            per_run_df, m, n_boot=n_boot, seed_base=50000 + i * 1000
        )
        if agg.empty:
            continue
        agg = agg.rename(
            columns={
                "mean": f"mean_{m}",
                "ci95_low": f"ci95_low_{m}",
                "ci95_high": f"ci95_high_{m}",
            }
        )
        parts.append(agg)

    if parts:
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(
                p.drop(columns=[c for c in p.columns if c == "n_runs"]),
                on=["corrupt_mixes", "E2E", "msg_delivery_percent", "scenario"],
                how="outer",
            )
    else:
        out = pd.DataFrame()

    if not auc_summary.empty:
        keep = [
            "corrupt_mixes",
            "E2E",
            "msg_delivery_percent",
            "scenario",
            "mean_auc",
            "ci95_low_auc",
            "ci95_high_auc",
            "delta_vs_baseline_auc",
            "delta_vs_baseline_ci95_low",
            "delta_vs_baseline_ci95_high",
            "significant_vs_baseline_95",
            "pct_delta_vs_baseline",
            "best_in_condition",
        ]
        if out.empty:
            out = auc_summary[keep].copy()
        else:
            out = out.merge(
                auc_summary[keep],
                on=["corrupt_mixes", "E2E", "msg_delivery_percent", "scenario"],
                how="outer",
            )

    if not out.empty:
        out = out.sort_values(
            ["corrupt_mixes", "E2E", "msg_delivery_percent", "scenario"]
        ).reset_index(drop=True)
        out.to_csv(
            plots_dir / "summary_metrics_per_condition.csv",
            index=False,
            float_format="%.6f",
        )
    return out


def build_delta_table(
    per_run_df: pd.DataFrame,
    run_auc_df: pd.DataFrame,
    plots_dir: Path,
    n_boot: int = 3000,
) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame()

    cond_cols = ["corrupt_mixes", "E2E", "msg_delivery_percent"]
    metric_cols = [
        "mean_entropy",
        "sim_extra_time",
        "avg_dummy_per_link",
        "avg_msgs_per_link",
    ]

    auc_view = (
        run_auc_df.rename(columns={"auc": "auc_value"})
        if not run_auc_df.empty
        else pd.DataFrame()
    )

    rows = []
    for ci, (cond_key, cond_df) in enumerate(per_run_df.groupby(cond_cols, sort=True)):
        corrupt, e2e, delivery = cond_key
        baseline_df = cond_df[cond_df["scenario"] == "baseline"]
        baseline_auc = (
            auc_view[
                (auc_view["corrupt_mixes"] == corrupt)
                & (auc_view["E2E"] == e2e)
                & (auc_view["msg_delivery_percent"] == delivery)
                & (auc_view["scenario"] == "baseline")
            ]["auc_value"].to_numpy()
            if not auc_view.empty
            else np.array([])
        )

        scenarios = sorted(
            cond_df["scenario"].dropna().astype(str).unique(), key=scenario_sort_key
        )
        for s_idx, scenario in enumerate(scenarios):
            if scenario == "baseline":
                continue
            scen_df = cond_df[cond_df["scenario"] == scenario]
            row = {
                "corrupt_mixes": int(corrupt),
                "E2E": int(e2e),
                "msg_delivery_percent": int(delivery),
                "scenario": scenario,
            }

            for m in metric_cols:
                if m not in cond_df.columns:
                    continue
                a = scen_df[m].dropna().to_numpy()
                b = baseline_df[m].dropna().to_numpy()
                if a.size == 0 or b.size == 0:
                    continue
                delta, d_lo, d_hi = bootstrap_mean_diff_ci(
                    a, b, n_boot=n_boot, seed=70000 + ci * 100 + s_idx * 10
                )
                row[f"delta_{m}"] = delta
                row[f"delta_{m}_ci95_low"] = d_lo
                row[f"delta_{m}_ci95_high"] = d_hi
                row[f"delta_{m}_significant_95"] = bool((d_lo > 0.0) or (d_hi < 0.0))

            if baseline_auc.size > 0 and not auc_view.empty:
                scen_auc = auc_view[
                    (auc_view["corrupt_mixes"] == corrupt)
                    & (auc_view["E2E"] == e2e)
                    & (auc_view["msg_delivery_percent"] == delivery)
                    & (auc_view["scenario"] == scenario)
                ]["auc_value"].to_numpy()
                if scen_auc.size > 0:
                    delta, d_lo, d_hi = bootstrap_mean_diff_ci(
                        scen_auc,
                        baseline_auc,
                        n_boot=n_boot,
                        seed=80000 + ci * 100 + s_idx * 10,
                    )
                    row["delta_auc"] = delta
                    row["delta_auc_ci95_low"] = d_lo
                    row["delta_auc_ci95_high"] = d_hi
                    row["delta_auc_significant_95"] = bool(
                        (d_lo > 0.0) or (d_hi < 0.0)
                    )
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["corrupt_mixes", "E2E", "msg_delivery_percent", "scenario"]
        ).reset_index(drop=True)
        out.to_csv(
            plots_dir / "delta_vs_baseline_95ci.csv",
            index=False,
            float_format="%.6f",
        )
    return out


def build_best_strategy(auc_summary: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    if auc_summary.empty:
        return pd.DataFrame()
    cond_cols = ["corrupt_mixes", "E2E", "msg_delivery_percent"]
    runner_rows = []
    for cond_key, cond_df in auc_summary.groupby(cond_cols, sort=True):
        sorted_aucs = sorted(cond_df["mean_auc"].dropna().tolist(), reverse=True)
        gap = (
            float(sorted_aucs[0] - sorted_aucs[1])
            if len(sorted_aucs) >= 2
            else float("nan")
        )
        runner_rows.append(
            {
                "corrupt_mixes": int(cond_key[0]),
                "E2E": int(cond_key[1]),
                "msg_delivery_percent": int(cond_key[2]),
                "best_minus_runner_up_auc": gap,
            }
        )
    runner_df = pd.DataFrame(runner_rows)

    best = auc_summary[auc_summary["best_in_condition"]].copy()
    best = best.merge(runner_df, on=cond_cols, how="left")
    best = best.sort_values(cond_cols).reset_index(drop=True)
    best.to_csv(
        plots_dir / "best_strategy_per_condition.csv",
        index=False,
        float_format="%.6f",
    )
    return best


def build_transition_window_dip(
    curve_df: pd.DataFrame, plots_dir: Path, smooth_window: int
) -> pd.DataFrame:
    sub = curve_df[curve_df["scenario"] == "baseline"]
    if sub.empty:
        return pd.DataFrame()

    cond_cols = ["corrupt_mixes", "E2E", "msg_delivery_percent"]
    rows = []
    for cond_key, cond_df in sub.groupby(cond_cols, sort=True):
        corrupt, e2e, delivery = cond_key
        curve = summarize_curve(
            cond_df,
            n_boot=200,
            seed=11000 + int(corrupt) * 100 + int(e2e) * 10 + int(delivery),
        )
        if curve.empty:
            continue
        smoothed = smooth_for_display(curve, window=smooth_window)
        if smoothed.empty:
            continue
        peak_idx = int(smoothed["mean_entropy"].idxmax())
        end_idx = int(smoothed.index[-1])
        peak_t = float(smoothed.loc[peak_idx, "time_bin_end"])
        peak_e = float(smoothed.loc[peak_idx, "mean_entropy"])
        end_t = float(smoothed.loc[end_idx, "time_bin_end"])
        end_e = float(smoothed.loc[end_idx, "mean_entropy"])
        rows.append(
            {
                "corrupt_mixes": int(corrupt),
                "E2E": int(e2e),
                "msg_delivery_percent": int(delivery),
                "peak_time": peak_t,
                "peak_entropy": peak_e,
                "end_time": end_t,
                "end_entropy": end_e,
                "dip_magnitude": float(peak_e - end_e),
                "dip_duration": float(end_t - peak_t),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(cond_cols).reset_index(drop=True)
        out.to_csv(
            plots_dir / "transition_window_dip.csv",
            index=False,
            float_format="%.6f",
        )
    return out


# ----------------------------------- main -------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thesis-grade analysis of campaign04 mixnet entropy and dummy strategies."
    )
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Rolling window for displayed entropy curves (in 0.5-unit bins).",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=DEFAULT_FIXED_HORIZON,
        help="Fixed AUC integration horizon (defaults to configured sim duration = 20).",
    )
    parser.add_argument("--run-bins-name", type=str, default="EntropyBins_0p5.csv")
    parser.add_argument(
        "--ref-corrupt",
        type=int,
        default=0,
        help="Reference corrupt_mixes value for the baseline-dip motivation plot.",
    )
    parser.add_argument(
        "--ref-e2e",
        type=int,
        default=1,
        help="Reference E2E value for the baseline-dip motivation plot.",
    )
    args = parser.parse_args()

    campaign_dir: Path = args.campaign_dir
    plots_dir: Path = args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading run-level entropy bins from {campaign_dir} ...")
    curve_df = load_campaign_entropy_runs(
        campaign_dir, run_bins_name=args.run_bins_name
    )
    if curve_df.empty:
        raise FileNotFoundError(
            f"No run-level entropy bins found under {campaign_dir}"
        )
    curve_df = curve_df.sort_values(
        [
            "msg_delivery_percent",
            "corrupt_mixes",
            "E2E",
            "scenario",
            "combo_id",
            "repeat_idx",
            "time_bin_end",
        ]
    ).reset_index(drop=True)
    curve_df.to_csv(
        plots_dir / "entropy_run_level_long.csv", index=False, float_format="%.6f"
    )
    print(
        f"  loaded {len(curve_df)} bin rows across "
        f"{curve_df['combo_id'].nunique()} combos"
    )

    print("Loading per-run summary metrics ...")
    per_run_df = load_per_run_metrics(campaign_dir)
    print(f"  loaded {len(per_run_df)} per-run summary rows")

    saved_plots: list[Path] = []

    print("Plotting A1: baseline transition-window dip ...")
    p = plot_baseline_dip(
        curve_df,
        plots_dir,
        n_boot=args.bootstrap_samples,
        smooth_window=args.smooth_window,
        ref_corrupt=args.ref_corrupt,
        ref_e2e=args.ref_e2e,
    )
    if p is not None:
        saved_plots.append(p)

    print("Plotting B1: entropy progression grids (one per delivery%) ...")
    saved_plots.extend(
        plot_entropy_progression_per_delivery(
            curve_df,
            plots_dir,
            n_boot=args.bootstrap_samples,
            smooth_window=args.smooth_window,
        )
    )

    print("Plotting B2: mean entropy summary ...")
    mean_entropy_summary = aggregate_per_run_metric(
        per_run_df, "mean_entropy", n_boot=args.bootstrap_samples, seed_base=60000
    )
    if not mean_entropy_summary.empty:
        mean_entropy_summary.to_csv(
            plots_dir / "mean_entropy_summary_95ci.csv",
            index=False,
            float_format="%.6f",
        )
        p = plot_metric_grid(
            mean_entropy_summary,
            plots_dir,
            title="Mean entropy per condition (95% CI)",
            ylabel="Mean entropy",
            filename_stem="mean_entropy_grid",
        )
        if p is not None:
            saved_plots.append(p)

    print(f"Computing run-level AUC at horizon={args.horizon} ...")
    run_auc_df = compute_run_level_auc(curve_df, horizon=args.horizon)
    if run_auc_df.empty:
        raise RuntimeError("Could not compute run-level AUC rows.")
    run_auc_df.to_csv(
        plots_dir / "run_level_auc.csv", index=False, float_format="%.6f"
    )

    print("Building AUC summary vs baseline ...")
    auc_summary = build_auc_summary(run_auc_df, n_boot=args.bootstrap_samples)
    if auc_summary.empty:
        raise RuntimeError("AUC summary is empty.")
    auc_summary.to_csv(
        plots_dir / "auc_summary_vs_baseline_95ci.csv",
        index=False,
        float_format="%.6f",
    )

    print("Plotting B3: AUC grid ...")
    p = plot_auc_grid(auc_summary, plots_dir)
    if p is not None:
        saved_plots.append(p)

    print("Plotting B4: Δ AUC vs baseline grid ...")
    p = plot_delta_auc_grid(auc_summary, plots_dir)
    if p is not None:
        saved_plots.append(p)

    print("Plotting C1: avg dummy messages per link ...")
    cost_summary = aggregate_per_run_metric(
        per_run_df, "avg_dummy_per_link", n_boot=args.bootstrap_samples, seed_base=80000
    )
    if not cost_summary.empty:
        cost_summary.to_csv(
            plots_dir / "avg_dummy_per_link_summary_95ci.csv",
            index=False,
            float_format="%.6f",
        )
        p = plot_metric_grid(
            cost_summary,
            plots_dir,
            title="Average dummy messages per link (bandwidth cost, 95% CI)",
            ylabel="Avg dummy msgs/link",
            filename_stem="dummy_per_link_grid",
        )
        if p is not None:
            saved_plots.append(p)

    print("Plotting C2: simulation extra time ...")
    extra_summary = aggregate_per_run_metric(
        per_run_df, "sim_extra_time", n_boot=args.bootstrap_samples, seed_base=90000
    )
    if not extra_summary.empty:
        extra_summary.to_csv(
            plots_dir / "extra_time_summary_95ci.csv",
            index=False,
            float_format="%.6f",
        )
        p = plot_metric_grid(
            extra_summary,
            plots_dir,
            title="Simulation extra time (latency cost, 95% CI)",
            ylabel="Sim extra time",
            filename_stem="extra_time_grid",
        )
        if p is not None:
            saved_plots.append(p)

    print("Plotting C3: Pareto entropy vs cost ...")
    p = plot_pareto_entropy_vs_cost(
        per_run_df, auc_summary, plots_dir, n_boot=args.bootstrap_samples
    )
    if p is not None:
        saved_plots.append(p)

    print("Building summary table: per-condition metrics ...")
    build_summary_metrics_per_condition(
        per_run_df, auc_summary, plots_dir, n_boot=args.bootstrap_samples
    )
    print("Building summary table: Δ vs baseline ...")
    build_delta_table(
        per_run_df, run_auc_df, plots_dir, n_boot=args.bootstrap_samples
    )
    print("Building summary table: best strategy per condition ...")
    build_best_strategy(auc_summary, plots_dir)
    print("Building summary table: baseline transition-window dip ...")
    build_transition_window_dip(curve_df, plots_dir, smooth_window=args.smooth_window)

    print("\nSaved plots:")
    for p in saved_plots:
        print(f"  {p}")
    print(f"\nAll CSV outputs saved under: {plots_dir}")


if __name__ == "__main__":
    main()
