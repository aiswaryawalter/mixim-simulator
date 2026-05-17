#!/usr/bin/env python3
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
DEFAULT_CAMPAIGN_DIR = ROOT / "files" / "campaign01"
DEFAULT_PLOTS_DIR = ROOT / "files" / "analysis-plots"
DEFAULT_CLIENTDUMMY_DIR = ROOT / "files" / "clientdummy"
DEFAULT_RHO_SWEEP_DIR = ROOT / "files" / "rho_sweep"

TRAFFIC_BIN_SIZE = 0.5
TRAFFIC_REAL_COLOR = "#1f77b4"
TRAFFIC_DUMMY_COLOR = "#2ca02c"

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

CORRUPT_MIXES_ORDER = [0, 3, 6]
CORRUPT_PCT_LABELS = {0: "0% corrupt", 3: "10% corrupt", 6: "20% corrupt"}
E2E_ORDER = [1, 3, 5]

RHO_VALUES = [1.0, 2.0, 4.0]
RHO_COLORS = {1.0: "#1f77b4", 2.0: "#2ca02c", 4.0: "#d62728"}
RHO_LABELS = {1.0: r"$\rho = 1$", 2.0: r"$\rho = 2$", 4.0: r"$\rho = 4$"}
RHO_LINESTYLES = {1.0: "-", 2.0: "-.", 4.0: "--"}
RHO_SCENARIO_ORDER = ["client_dummies", "link_based_dummies", "multiple_hop_dummies"]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 3000, seed: int = 0) -> tuple[float, float, float]:
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
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = 3000,
    seed: int = 0,
) -> tuple[float, float, float]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]

    if values_a.size == 0 or values_b.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_a = rng.integers(0, values_a.size, size=(n_boot, values_a.size))
    boot_b = rng.integers(0, values_b.size, size=(n_boot, values_b.size))
    diff = values_a[boot_a].mean(axis=1) - values_b[boot_b].mean(axis=1)
    mean_diff = float(values_a.mean() - values_b.mean())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def compute_auc(time_values: np.ndarray, entropy_values: np.ndarray, horizon: float | None = None) -> float:
    x = np.asarray(time_values, dtype=float)
    y = np.asarray(entropy_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if horizon is not None:
        if x[-1] < horizon:
            if x.size < 2:
                return np.nan
            y_h = float(np.interp(horizon, x, y))
            x = np.append(x, horizon)
            y = np.append(y, y_h)
        else:
            keep = x < horizon
            if not np.any(x == horizon):
                if np.sum(keep) < 1:
                    return np.nan
                x_keep = x[keep]
                y_keep = y[keep]
                if x_keep.size < 2:
                    return np.nan
                y_h = float(np.interp(horizon, x, y))
                x = np.append(x_keep, horizon)
                y = np.append(y_keep, y_h)
            else:
                mask = x <= horizon
                x = x[mask]
                y = y[mask]

    if x.size < 2:
        return np.nan

    return float(np.trapezoid(y, x))


def load_combo_meta(combo_dir: Path) -> dict:
    meta_path = combo_dir / "combo_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    summary_path = combo_dir / "summary_combo.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty:
            row = df.iloc[0].to_dict()
            return {
                "combo_id": int(row.get("combo_id", -1)),
                "scenario": str(row.get("scenario", "")),
                "corrupt_mixes": int(row.get("corrupt_mixes", -1)),
                "E2E": int(row.get("E2E", -1)),
                "rho": float(row.get("rho", float("nan"))) if "rho" in row else float("nan"),
            }

    return {}


def load_campaign_runs(campaign_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    combo_dirs = sorted([p for p in campaign_dir.glob("combo_*") if p.is_dir()])

    for combo_dir in combo_dirs:
        meta = load_combo_meta(combo_dir)
        if not meta:
            continue

        combo_id = int(meta.get("combo_id", -1))
        scenario = str(meta.get("scenario", ""))
        corrupt_mixes = int(meta.get("corrupt_mixes", -1))
        e2e = int(meta.get("E2E", -1))
        rho = float(meta.get("rho", float("nan")))

        run_dirs = sorted([p for p in combo_dir.glob("run_*") if p.is_dir()])
        for run_dir in run_dirs:
            bins_path = run_dir / "EntropyBins_0p5.csv"
            if not bins_path.exists():
                continue

            df = pd.read_csv(bins_path)
            if df.empty:
                continue
            if not {"time_bin_end", "avg_entropy"}.issubset(df.columns):
                continue

            df = df.copy()
            df["combo_id"] = combo_id
            df["scenario"] = scenario
            df["corrupt_mixes"] = corrupt_mixes
            df["E2E"] = e2e
            df["rho"] = rho
            df["repeat_idx"] = df.get("repeat_idx", np.nan)
            if pd.isna(df["repeat_idx"]).all():
                repeat_name = run_dir.name.replace("run_", "")
                try:
                    repeat_idx = int(repeat_name)
                except ValueError:
                    repeat_idx = -1
                df["repeat_idx"] = repeat_idx

            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["time_bin_end"] = pd.to_numeric(out["time_bin_end"], errors="coerce")
    out["avg_entropy"] = pd.to_numeric(out["avg_entropy"], errors="coerce")
    out["combo_id"] = pd.to_numeric(out["combo_id"], errors="coerce")
    out["corrupt_mixes"] = pd.to_numeric(out["corrupt_mixes"], errors="coerce")
    out["E2E"] = pd.to_numeric(out["E2E"], errors="coerce")
    out["repeat_idx"] = pd.to_numeric(out["repeat_idx"], errors="coerce")
    if "rho" in out.columns:
        out["rho"] = pd.to_numeric(out["rho"], errors="coerce")
    out = out.dropna(subset=["time_bin_end", "avg_entropy", "combo_id", "corrupt_mixes", "E2E"])
    return out


def summarize_curve(df: pd.DataFrame, n_boot: int = 3000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for idx, (time_bin_end, group) in enumerate(df.groupby("time_bin_end", sort=True)):
        values = group["avg_entropy"].astype(float).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed + idx)
        rows.append(
            {
                "time_bin_end": float(time_bin_end),
                "mean_entropy": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": int(np.isfinite(values).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_run_level_aucs(df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    rows = []
    group_cols = ["combo_id", "scenario", "corrupt_mixes", "E2E", "repeat_idx"]

    for key, group in df.groupby(group_cols):
        combo_id, scenario, corrupt_mixes, e2e, repeat_idx = key
        auc = compute_auc(group["time_bin_end"].to_numpy(), group["avg_entropy"].to_numpy(), horizon=horizon)
        rows.append(
            {
                "combo_id": int(combo_id),
                "scenario": str(scenario),
                "corrupt_mixes": int(corrupt_mixes),
                "E2E": int(e2e),
                "repeat_idx": int(repeat_idx),
                "auc": auc,
            }
        )

    out = pd.DataFrame(rows)
    out = out[np.isfinite(out["auc"])]
    return out


def load_sent_messages_for_combo(combo_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    run_dirs = sorted([p for p in combo_dir.glob("run_*") if p.is_dir()])
    for run_dir in run_dirs:
        sent_path = run_dir / "SentMessages.csv"
        if not sent_path.exists():
            continue
        df = pd.read_csv(sent_path)
        if df.empty:
            continue
        if not {"MessageType", "MessageTimeLeft"}.issubset(df.columns):
            continue
        try:
            repeat_idx = int(run_dir.name.replace("run_", ""))
        except ValueError:
            repeat_idx = -1
        sub = df[["MessageType", "MessageTimeLeft"]].copy()
        sub["repeat_idx"] = repeat_idx
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["MessageTimeLeft"] = pd.to_numeric(out["MessageTimeLeft"], errors="coerce")
    out = out.dropna(subset=["MessageTimeLeft"])
    return out


def summarize_volume_with_ci(
    messages_df: pd.DataFrame,
    bin_size: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    if messages_df.empty:
        return pd.DataFrame()

    df = messages_df.copy()
    df["is_real"] = df["MessageType"].astype(str) == "Real"

    t_max = float(df["MessageTimeLeft"].max())
    if not np.isfinite(t_max) or t_max <= 0:
        return pd.DataFrame()

    n_bins = int(np.ceil(t_max / bin_size))
    if n_bins <= 0:
        return pd.DataFrame()

    edges = np.arange(0, (n_bins + 1) * bin_size, bin_size)
    bin_centers = edges[:-1] + bin_size / 2.0

    df["bin_idx"] = np.clip(
        np.floor(df["MessageTimeLeft"].to_numpy() / bin_size).astype(int),
        0,
        n_bins - 1,
    )

    runs = sorted(df["repeat_idx"].dropna().unique().tolist())
    n_runs = len(runs)
    if n_runs == 0:
        return pd.DataFrame()

    counts = df.groupby(["repeat_idx", "bin_idx", "is_real"]).size().reset_index(name="count")
    real_matrix = np.zeros((n_runs, n_bins), dtype=float)
    dummy_matrix = np.zeros((n_runs, n_bins), dtype=float)
    run_to_idx = {r: i for i, r in enumerate(runs)}

    for _, row in counts.iterrows():
        i = run_to_idx[row["repeat_idx"]]
        b = int(row["bin_idx"])
        if row["is_real"]:
            real_matrix[i, b] = row["count"]
        else:
            dummy_matrix[i, b] = row["count"]

    rows = []
    for b in range(n_bins):
        r_mean, r_lo, r_hi = bootstrap_mean_ci(real_matrix[:, b], n_boot=n_boot, seed=seed + b)
        d_mean, d_lo, d_hi = bootstrap_mean_ci(dummy_matrix[:, b], n_boot=n_boot, seed=seed + 10_000 + b)
        rows.append(
            {
                "bin_idx": b,
                "time": float(bin_centers[b]),
                "real_mean": r_mean,
                "real_ci95_low": r_lo,
                "real_ci95_high": r_hi,
                "dummy_mean": d_mean,
                "dummy_ci95_low": d_lo,
                "dummy_ci95_high": d_hi,
                "n_runs": n_runs,
            }
        )
    return pd.DataFrame(rows)


def plot_traffic_volume_grid(
    clientdummy_dir: Path,
    out_dir: Path,
    bin_size: float = TRAFFIC_BIN_SIZE,
    n_boot: int = 1000,
) -> Path | None:
    matrix_path = clientdummy_dir / "experiment_matrix.csv"
    if not matrix_path.exists():
        print(f"experiment_matrix.csv not found in {clientdummy_dir}; skipping traffic volume plot.")
        return None

    matrix = pd.read_csv(matrix_path)
    needed = {"combo_id", "corrupt_mixes", "E2E"}
    if not needed.issubset(matrix.columns):
        print(f"experiment_matrix.csv missing required columns {needed}; skipping traffic volume plot.")
        return None

    fig, axes = plt.subplots(3, 3, figsize=(19, 14), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.90, wspace=0.12, hspace=0.18)

    any_data = False
    csv_rows: list[pd.DataFrame] = []
    for r, corrupt_mixes in enumerate(CORRUPT_MIXES_ORDER):
        for c, e2e in enumerate(E2E_ORDER):
            ax = axes[r, c]
            corrupt_label = CORRUPT_PCT_LABELS.get(corrupt_mixes, f"corrupt={corrupt_mixes}")
            ax.set_title(f"{corrupt_label}, E2E={e2e}", fontsize=10)
            ax.grid(True, alpha=0.25)
            if r == 2:
                ax.set_xlabel("Simulation time")
            if c == 0:
                ax.set_ylabel(f"Messages per {bin_size}s bin (mean across runs)")

            row = matrix[(matrix["corrupt_mixes"] == corrupt_mixes) & (matrix["E2E"] == e2e)]
            if row.empty:
                continue

            combo_id = int(row.iloc[0]["combo_id"])
            combo_dir = clientdummy_dir / f"combo_{combo_id:03d}"
            if not combo_dir.exists():
                continue

            messages_df = load_sent_messages_for_combo(combo_dir)
            volume = summarize_volume_with_ci(
                messages_df,
                bin_size=bin_size,
                n_boot=n_boot,
                seed=3000 + int(corrupt_mixes) * 100 + int(e2e) * 10,
            )
            if volume.empty:
                continue

            x = volume["time"].to_numpy()
            ax.plot(x, volume["real_mean"].to_numpy(), color=TRAFFIC_REAL_COLOR, linewidth=1.8, label="real")
            ax.fill_between(
                x,
                volume["real_ci95_low"].to_numpy(),
                volume["real_ci95_high"].to_numpy(),
                color=TRAFFIC_REAL_COLOR,
                alpha=0.14,
                linewidth=0,
            )
            ax.plot(x, volume["dummy_mean"].to_numpy(), color=TRAFFIC_DUMMY_COLOR, linewidth=1.8, label="client dummy")
            ax.fill_between(
                x,
                volume["dummy_ci95_low"].to_numpy(),
                volume["dummy_ci95_high"].to_numpy(),
                color=TRAFFIC_DUMMY_COLOR,
                alpha=0.14,
                linewidth=0,
            )
            any_data = True

            csv_part = volume.copy()
            csv_part.insert(0, "corrupt_mixes", corrupt_mixes)
            csv_part.insert(1, "E2E", e2e)
            csv_part.insert(2, "combo_id", combo_id)
            csv_rows.append(csv_part)

    if not any_data:
        plt.close(fig)
        print("No SentMessages.csv data found under client dummies; skipping traffic volume plot.")
        return None

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(
        f"Traffic volume progression — client dummies scenario (bin={bin_size}s)",
        fontsize=14,
        y=0.965,
    )

    out_path = out_dir / "traffic_volume_progression_clientdummy.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(
            out_dir / "traffic_volume_progression_clientdummy.csv",
            index=False,
            float_format="%.6f",
        )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rho_entropy_grid(
    curve_df: pd.DataFrame,
    scenario: str,
    out_dir: Path,
    n_boot: int = 3000,
) -> Path | None:
    if "rho" not in curve_df.columns:
        return None

    subset = curve_df[(curve_df["scenario"] == scenario) & curve_df["rho"].isin(RHO_VALUES)].copy()
    if subset.empty:
        return None

    fig, axes = plt.subplots(3, 3, figsize=(19, 14), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.90, wspace=0.12, hspace=0.18)

    any_data = False
    csv_rows: list[pd.DataFrame] = []
    for r, corrupt_mixes in enumerate(CORRUPT_MIXES_ORDER):
        for c, e2e in enumerate(E2E_ORDER):
            ax = axes[r, c]
            cell = subset[(subset["corrupt_mixes"] == corrupt_mixes) & (subset["E2E"] == e2e)]

            for rho in RHO_VALUES:
                rho_slice = cell[cell["rho"] == rho]
                if rho_slice.empty:
                    continue

                curve = summarize_curve(
                    rho_slice,
                    n_boot=n_boot,
                    seed=2000 + int(corrupt_mixes) * 100 + int(e2e) * 10 + int(rho),
                )
                if curve.empty:
                    continue

                x = curve["time_bin_end"].to_numpy()
                y = curve["mean_entropy"].to_numpy()
                lo = curve["ci95_low"].to_numpy()
                hi = curve["ci95_high"].to_numpy()

                ax.plot(x, y, color=RHO_COLORS[rho], linewidth=2.0, label=RHO_LABELS[rho])
                ax.fill_between(x, lo, hi, color=RHO_COLORS[rho], alpha=0.18, linewidth=0)
                any_data = True

                csv_part = curve.copy()
                csv_part.insert(0, "scenario", scenario)
                csv_part.insert(1, "corrupt_mixes", corrupt_mixes)
                csv_part.insert(2, "E2E", e2e)
                csv_part.insert(3, "rho", rho)
                csv_rows.append(csv_part)

            corrupt_label = CORRUPT_PCT_LABELS.get(corrupt_mixes, f"corrupt={corrupt_mixes}")
            ax.set_title(f"{corrupt_label}, E2E={e2e}", fontsize=10)
            ax.grid(True, alpha=0.25)
            if r == 2:
                ax.set_xlabel("Arrival time")
            if c == 0:
                ax.set_ylabel("Entropy")

    if not any_data:
        plt.close(fig)
        return None

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)

    rho_present = sorted(subset["rho"].dropna().unique())
    rho_str = ", ".join(f"ρ = {int(r) if r == int(r) else r}" for r in rho_present)
    fig.suptitle(
        f"Entropy progression — {SCENARIO_LABELS.get(scenario, scenario)} ({rho_str})",
        fontsize=14,
        y=0.965,
    )

    out_path = out_dir / f"entropy_rho_compare_{scenario}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(
            out_dir / f"entropy_rho_compare_{scenario}.csv",
            index=False,
            float_format="%.6f",
        )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rho_traffic_grid(
    rho_to_campaign: dict,
    scenario: str,
    out_dir: Path,
    bin_size: float = TRAFFIC_BIN_SIZE,
    n_boot: int = 1000,
) -> Path | None:
    rho_to_matrix: dict[float, tuple[pd.DataFrame, Path]] = {}
    for rho, campaign_dir in rho_to_campaign.items():
        matrix_path = campaign_dir / "experiment_matrix.csv"
        if not matrix_path.exists():
            continue
        matrix = pd.read_csv(matrix_path)
        needed = {"combo_id", "corrupt_mixes", "E2E", "scenario"}
        if not needed.issubset(matrix.columns):
            continue
        sub = matrix[matrix["scenario"] == scenario].copy()
        if sub.empty:
            continue
        rho_to_matrix[rho] = (sub, campaign_dir)

    if not rho_to_matrix:
        return None

    fig, axes = plt.subplots(3, 3, figsize=(19, 14), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.90, wspace=0.12, hspace=0.18)

    any_data = False
    csv_rows: list[pd.DataFrame] = []
    for r, corrupt_mixes in enumerate(CORRUPT_MIXES_ORDER):
        for c, e2e in enumerate(E2E_ORDER):
            ax = axes[r, c]
            corrupt_label = CORRUPT_PCT_LABELS.get(corrupt_mixes, f"corrupt={corrupt_mixes}")
            ax.set_title(f"{corrupt_label}, E2E={e2e}", fontsize=10)
            ax.grid(True, alpha=0.25)
            if r == 2:
                ax.set_xlabel("Simulation time")
            if c == 0:
                ax.set_ylabel(f"Messages per {bin_size}s bin (mean across runs)")

            for rho in RHO_VALUES:
                if rho not in rho_to_matrix:
                    continue
                matrix, campaign_dir = rho_to_matrix[rho]
                row = matrix[(matrix["corrupt_mixes"] == corrupt_mixes) & (matrix["E2E"] == e2e)]
                if row.empty:
                    continue

                combo_id = int(row.iloc[0]["combo_id"])
                combo_dir = campaign_dir / f"combo_{combo_id:03d}"
                if not combo_dir.exists():
                    continue

                messages_df = load_sent_messages_for_combo(combo_dir)
                volume = summarize_volume_with_ci(
                    messages_df,
                    bin_size=bin_size,
                    n_boot=n_boot,
                    seed=4000 + int(corrupt_mixes) * 100 + int(e2e) * 10 + int(rho),
                )
                if volume.empty:
                    continue

                x = volume["time"].to_numpy()
                ls = RHO_LINESTYLES[rho]
                ax.plot(
                    x,
                    volume["real_mean"].to_numpy(),
                    color=TRAFFIC_REAL_COLOR,
                    linestyle=ls,
                    linewidth=1.8,
                    label=f"real, {RHO_LABELS[rho]}",
                )
                ax.fill_between(
                    x,
                    volume["real_ci95_low"].to_numpy(),
                    volume["real_ci95_high"].to_numpy(),
                    color=TRAFFIC_REAL_COLOR,
                    alpha=0.10,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    volume["dummy_mean"].to_numpy(),
                    color=TRAFFIC_DUMMY_COLOR,
                    linestyle=ls,
                    linewidth=1.8,
                    label=f"dummy, {RHO_LABELS[rho]}",
                )
                ax.fill_between(
                    x,
                    volume["dummy_ci95_low"].to_numpy(),
                    volume["dummy_ci95_high"].to_numpy(),
                    color=TRAFFIC_DUMMY_COLOR,
                    alpha=0.10,
                    linewidth=0,
                )
                any_data = True

                csv_part = volume.copy()
                csv_part.insert(0, "scenario", scenario)
                csv_part.insert(1, "corrupt_mixes", corrupt_mixes)
                csv_part.insert(2, "E2E", e2e)
                csv_part.insert(3, "rho", rho)
                csv_part.insert(4, "combo_id", combo_id)
                csv_rows.append(csv_part)

    if not any_data:
        plt.close(fig)
        return None

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)

    rho_str = ", ".join(f"ρ = {int(r) if r == int(r) else r}" for r in sorted(rho_to_matrix.keys()))
    fig.suptitle(
        f"Traffic volume — {SCENARIO_LABELS.get(scenario, scenario)} ({rho_str}, bin={bin_size}s)",
        fontsize=14,
        y=0.965,
    )

    out_path = out_dir / f"traffic_rho_compare_{scenario}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(
            out_dir / f"traffic_rho_compare_{scenario}.csv",
            index=False,
            float_format="%.6f",
        )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_condition_grid(
    curve_df: pd.DataFrame,
    out_dir: Path,
    n_boot: int = 3000,
) -> Path | None:
    if curve_df.empty:
        return None

    fig, axes = plt.subplots(3, 3, figsize=(19, 14), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.90, wspace=0.12, hspace=0.18)

    csv_rows: list[pd.DataFrame] = []
    for r, corrupt_mixes in enumerate(CORRUPT_MIXES_ORDER):
        for c, e2e in enumerate(E2E_ORDER):
            ax = axes[r, c]
            cell = curve_df[(curve_df["corrupt_mixes"] == corrupt_mixes) & (curve_df["E2E"] == e2e)]

            for scenario in SCENARIO_ORDER:
                scen = cell[cell["scenario"] == scenario]
                if scen.empty:
                    continue

                curve = summarize_curve(scen, n_boot=n_boot, seed=1000 + corrupt_mixes * 100 + e2e * 10)
                if curve.empty:
                    continue

                x = curve["time_bin_end"].to_numpy()
                y = curve["mean_entropy"].to_numpy()
                lo = curve["ci95_low"].to_numpy()
                hi = curve["ci95_high"].to_numpy()

                ax.plot(
                    x,
                    y,
                    color=SCENARIO_COLORS[scenario],
                    linewidth=2.0,
                    label=SCENARIO_LABELS[scenario],
                )
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    color=SCENARIO_COLORS[scenario],
                    alpha=0.14,
                    linewidth=0,
                )

                csv_part = curve.copy()
                csv_part.insert(0, "corrupt_mixes", corrupt_mixes)
                csv_part.insert(1, "E2E", e2e)
                csv_part.insert(2, "scenario", scenario)
                csv_rows.append(csv_part)

            corrupt_label = CORRUPT_PCT_LABELS.get(corrupt_mixes, f"corrupt={corrupt_mixes}")
            ax.set_title(f"{corrupt_label}, E2E={e2e}", fontsize=10)
            ax.grid(True, alpha=0.25)

            if r == 2:
                ax.set_xlabel("Arrival time")
            if c == 0:
                ax.set_ylabel("Entropy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    fig.suptitle(
        "Entropy progression over arrival time",
        fontsize=14,
        y=0.965,
    )

    out_path = out_dir / "entropy_progression.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(
            out_dir / "entropy_progression.csv",
            index=False,
            float_format="%.6f",
        )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def build_auc_summary(run_auc_df: pd.DataFrame, out_dir: Path, n_boot: int = 3000) -> pd.DataFrame:
    rows = []

    for key, group in run_auc_df.groupby(["corrupt_mixes", "E2E"], sort=True):
        corrupt_mixes, e2e = key
        baseline = group[group["scenario"] == "baseline"]["auc"].to_numpy()

        condition_rows = []
        for scenario in SCENARIO_ORDER:
            scen = group[group["scenario"] == scenario]["auc"].to_numpy()
            if scen.size == 0:
                continue

            mean, lo, hi = bootstrap_mean_ci(scen, n_boot=n_boot, seed=5000 + int(corrupt_mixes) * 100 + int(e2e) * 10)
            row = {
                "corrupt_mixes": int(corrupt_mixes),
                "E2E": int(e2e),
                "scenario": scenario,
                "mean_auc": mean,
                "ci95_low_auc": lo,
                "ci95_high_auc": hi,
                "n_runs": int(scen.size),
            }

            if scenario != "baseline" and baseline.size > 0:
                delta, d_lo, d_hi = bootstrap_mean_diff_ci(
                    scen,
                    baseline,
                    n_boot=n_boot,
                    seed=9000 + int(corrupt_mixes) * 100 + int(e2e) * 10,
                )
                row["delta_vs_baseline_auc"] = delta
                row["delta_vs_baseline_ci95_low"] = d_lo
                row["delta_vs_baseline_ci95_high"] = d_hi
                row["significant_vs_baseline_95"] = bool(d_lo > 0 or d_hi < 0)
                row["pct_delta_vs_baseline"] = float(100.0 * delta / baseline.mean()) if baseline.size and baseline.mean() != 0 else np.nan
            else:
                row["delta_vs_baseline_auc"] = 0.0
                row["delta_vs_baseline_ci95_low"] = 0.0
                row["delta_vs_baseline_ci95_high"] = 0.0
                row["significant_vs_baseline_95"] = False
                row["pct_delta_vs_baseline"] = 0.0

            condition_rows.append(row)

        if condition_rows:
            best_idx = int(np.argmax([r["mean_auc"] for r in condition_rows]))
            for idx, row in enumerate(condition_rows):
                row["best_in_condition"] = idx == best_idx
            rows.extend(condition_rows)

    out = pd.DataFrame(rows)
    out_path = out_dir / "auc_summary_vs_baseline_95ci.csv"
    out.to_csv(out_path, index=False, float_format="%.6f")
    return out


def print_summary_table(auc_summary: pd.DataFrame) -> None:
    if auc_summary.empty:
        print("No AUC summary available.")
        return

    display_cols = [
        "corrupt_mixes",
        "E2E",
        "scenario",
        "mean_auc",
        "delta_vs_baseline_auc",
        "delta_vs_baseline_ci95_low",
        "delta_vs_baseline_ci95_high",
        "significant_vs_baseline_95",
        "best_in_condition",
    ]
    cols = [c for c in display_cols if c in auc_summary.columns]
    view = auc_summary[cols].copy()
    view = view.sort_values(["corrupt_mixes", "E2E", "scenario"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nAUC summary vs baseline, 95% bootstrap CI:")
    print(view.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze campaign01 entropy progression and generate thesis-ready plots."
    )
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--clientdummy-dir", type=Path, default=DEFAULT_CLIENTDUMMY_DIR)
    parser.add_argument("--rho-sweep-dir", type=Path, default=DEFAULT_RHO_SWEEP_DIR)
    parser.add_argument("--traffic-bin-size", type=float, default=TRAFFIC_BIN_SIZE)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--traffic-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--horizon", type=float, default=None, help="Optional fixed AUC horizon. Default uses shared max time per condition.")
    args = parser.parse_args()

    campaign_dir: Path = args.campaign_dir
    plots_dir: Path = args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    curve_df = load_campaign_runs(campaign_dir)
    if curve_df.empty:
        raise FileNotFoundError(f"No run-level entropy bins found under {campaign_dir}")

    curve_df = curve_df.sort_values(
        ["corrupt_mixes", "E2E", "scenario", "combo_id", "repeat_idx", "time_bin_end"]
    ).reset_index(drop=True)

    saved_plots: list[Path] = []

    traffic_path = plot_traffic_volume_grid(
        clientdummy_dir=args.clientdummy_dir,
        out_dir=plots_dir,
        bin_size=args.traffic_bin_size,
        n_boot=args.traffic_bootstrap_samples,
    )
    if traffic_path is not None:
        saved_plots.append(traffic_path)

    out_path = plot_condition_grid(
        curve_df=curve_df,
        out_dir=plots_dir,
        n_boot=args.bootstrap_samples,
    )
    if out_path is not None:
        saved_plots.append(out_path)

    rho_sweep_dir: Path = args.rho_sweep_dir
    rho_curve_parts: list[pd.DataFrame] = []
    rho_to_campaign: dict[float, Path] = {}

    if rho_sweep_dir.exists():
        rho_sweep_df = load_campaign_runs(rho_sweep_dir)
        if not rho_sweep_df.empty:
            rho_curve_parts.append(rho_sweep_df)
            for rho in rho_sweep_df["rho"].dropna().unique():
                rho_to_campaign[float(rho)] = rho_sweep_dir
    else:
        print(f"Rho sweep directory not found at {rho_sweep_dir}; ρ=1 and ρ=4 will be missing.")

    if "rho" in curve_df.columns:
        campaign_rho_df = curve_df[curve_df["scenario"].isin(RHO_SCENARIO_ORDER)].copy()
        if not campaign_rho_df.empty:
            rho_curve_parts.append(campaign_rho_df)
            for rho in campaign_rho_df["rho"].dropna().unique():
                rho_to_campaign.setdefault(float(rho), campaign_dir)

    if rho_curve_parts:
        rho_curve_df = pd.concat(rho_curve_parts, ignore_index=True).sort_values(
            ["scenario", "rho", "corrupt_mixes", "E2E", "combo_id", "repeat_idx", "time_bin_end"]
        ).reset_index(drop=True)
        for scenario in RHO_SCENARIO_ORDER:
            ent_path = plot_rho_entropy_grid(
                curve_df=rho_curve_df,
                scenario=scenario,
                out_dir=plots_dir,
                n_boot=args.bootstrap_samples,
            )
            if ent_path is not None:
                saved_plots.append(ent_path)

            traf_path = plot_rho_traffic_grid(
                rho_to_campaign=rho_to_campaign,
                scenario=scenario,
                out_dir=plots_dir,
                bin_size=args.traffic_bin_size,
                n_boot=args.traffic_bootstrap_samples,
            )
            if traf_path is not None:
                saved_plots.append(traf_path)
    else:
        print("No data found for rho-compare plots; skipping.")

    auc_rows = []
    for _, group in curve_df.groupby(["corrupt_mixes", "E2E"], sort=True):
        if args.horizon is None:
            scenario_maxes = group.groupby("scenario")["time_bin_end"].max()
            if scenario_maxes.empty:
                continue
            horizon = float(scenario_maxes.min())
        else:
            horizon = float(args.horizon)

        run_auc_df = compute_run_level_aucs(group, horizon=horizon)
        if run_auc_df.empty:
            continue

        run_auc_df["horizon"] = horizon
        auc_rows.append(run_auc_df)

    if not auc_rows:
        raise RuntimeError("Could not compute any AUC rows.")

    run_auc_df = pd.concat(auc_rows, ignore_index=True)
    run_auc_df.to_csv(plots_dir / "run_level_auc.csv", index=False, float_format="%.6f")

    auc_summary = build_auc_summary(run_auc_df, plots_dir, n_boot=args.bootstrap_samples)
    print_summary_table(auc_summary)

    best_rows = auc_summary[auc_summary["best_in_condition"]].copy()
    best_rows = best_rows.sort_values(["corrupt_mixes", "E2E"])
    best_rows.to_csv(plots_dir / "best_scenario_per_condition.csv", index=False, float_format="%.6f")

    print("\nSaved plots:")
    for p in saved_plots:
        print(p)

    print(f"\nSaved statistical summaries in: {plots_dir}")


if __name__ == "__main__":
    main()