#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
PLOTS_DIR = FILES_DIR / "plots"

# Expected from runner script
SCENARIO_ORDER = ["baseline", "client_dummies", "link_based_dummies", "multiple_hop_dummies"]
STOP_PERCENTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]

SCENARIO_COLORS = {
    "baseline": "#1f77b4",
    "client_dummies": "#2ca02c",
    "link_based_dummies": "#ff7f0e",
    "multiple_hop_dummies": "#d62728",
}

# Linestyles for different stop percentages
STOP_LINESTYLES = {
    1: "-",
    2: "-",
    3: "-",
    4: "-",
    5: "-",
    6: "-",
    7: "-",
    8: "-",
    9: "-",
    10: "-",
    20: "--",
}


def parse_name(path: Path) -> tuple[str, int, int] | None:
    # e.g. baseline_L3_M5_Stop10_Entropy.csv
    m = re.match(r"(.+)_L(\d+)_M(\d+)_Stop(\d+)_Entropy\.csv$", path.name)
    if not m:
        return None
    scenario = m.group(1)
    mixes = int(m.group(3))
    stop_pct = int(m.group(4))
    return scenario, mixes, stop_pct


def load_entropy_with_time(entropy_file: Path) -> pd.DataFrame:
    """
    Returns dataframe with columns:
    - MessageID
    - send_time
    - Entropy
    """
    df_entropy = pd.read_csv(entropy_file)

    # Prefer TimeLeft already stored in entropy output (from received record)
    if "TimeLeft" in df_entropy.columns:
        df = df_entropy[["MessageID", "TimeLeft", "Entropy"]].copy()
        df.rename(columns={"TimeLeft": "send_time"}, inplace=True)
        return df

    # Fallback: join with corresponding SentMessages file
    sent_file = Path(str(entropy_file).replace("_Entropy.csv", "_SentMessages.csv"))
    if not sent_file.exists():
        raise FileNotFoundError(f"Missing sent messages file: {sent_file}")

    df_sent = pd.read_csv(sent_file)
    needed = {"MessageID", "MessageTimeLeft"}
    if not needed.issubset(df_sent.columns):
        raise ValueError(f"{sent_file} missing columns {needed}")

    df = df_entropy.merge(df_sent[["MessageID", "MessageTimeLeft"]], on="MessageID", how="left")
    df.rename(columns={"MessageTimeLeft": "send_time"}, inplace=True)
    return df[["MessageID", "send_time", "Entropy"]]


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    entropy_files = sorted(FILES_DIR.glob("*_Entropy.csv"))
    if not entropy_files:
        raise FileNotFoundError(f"No *_Entropy.csv files found in {FILES_DIR}")

    # Group by (mixes, scenario, stop_pct)
    grouped: dict[tuple[int, str], dict[int, Path]] = {}
    for f in entropy_files:
        parsed = parse_name(f)
        if not parsed:
            continue
        scenario, mixes, stop_pct = parsed
        key = (mixes, scenario)
        grouped.setdefault(key, {})[stop_pct] = f

    if not grouped:
        raise RuntimeError("No files matched expected naming pattern *_L*_M*_Stop*_Entropy.csv")

    # Plot 1: Entropy progression for each (mixes, scenario) combo, colored by stop%
    print("Generating entropy progression plots grouped by mixes and scenario...")
    mixes_values = sorted(set(k[0] for k in grouped.keys()))
    scenarios = SCENARIO_ORDER

    fig, axes = plt.subplots(len(scenarios), len(mixes_values), figsize=(24, 16))
    if len(scenarios) == 1:
        axes = axes.reshape(1, -1)
    if len(mixes_values) == 1:
        axes = axes.reshape(-1, 1)

    for s_idx, scenario in enumerate(scenarios):
        for m_idx, mixes in enumerate(mixes_values):
            ax = axes[s_idx, m_idx]
            key = (mixes, scenario)

            if key not in grouped:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{scenario} / M={mixes}")
                continue

            stop_files = grouped[key]
            for stop_pct in sorted(stop_files.keys()):
                path = stop_files[stop_pct]
                try:
                    df = load_entropy_with_time(path)
                    df = df.dropna(subset=["send_time", "Entropy"]).sort_values("send_time")
                    df["EntropySmoothed"] = df["Entropy"].rolling(window=25, min_periods=1).mean()

<<<<<<< HEAD:plot_entropy_progression.py
                    ax.plot(
                        df["send_time"],
                        df["EntropySmoothed"],
                        label=f"Stop {stop_pct}%",
                        alpha=0.8,
                        linewidth=1.5,
                        linestyle=STOP_LINESTYLES.get(stop_pct, "-"),
                    )
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")

            ax.set_title(f"{scenario} / M={mixes}")
            ax.set_xlabel("Message sent time")
            ax.set_ylabel("Entropy")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
=======
            # Optional smoothing to make progression easier to see
            df["EntropySmoothed"] = df["Entropy"].rolling(window=5, min_periods=1).mean()

            ax.plot(
                df["send_time"],
                df["Entropy"],
                label=scenario,
                color=COLORS.get(scenario, None),
                linewidth=1.8,
                alpha=0.95,
            )
>>>>>>> origin/avg-link-load:experiments/ target-row-sent-time/plot.py

    fig.suptitle("Entropy progression vs. stop_real_msgs_percent", y=0.995, fontsize=14)
    fig.tight_layout()

    out_path = PLOTS_DIR / "entropy_vs_stop_percent.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    # Plot 2: Heatmap comparing entropy across mixes for each (scenario, stop%)
    print("Generating entropy heatmap across scenarios...")
    fig, axes = plt.subplots(1, len(STOP_PERCENTS), figsize=(28, 4))
    if len(STOP_PERCENTS) == 1:
        axes = [axes]

    for pct_idx, stop_pct in enumerate(STOP_PERCENTS):
        ax = axes[pct_idx]

        # Build a matrix: rows=scenarios, cols=mixes
        data = []
        for scenario in scenarios:
            row = []
            for mixes in mixes_values:
                key = (mixes, scenario)
                if key in grouped and stop_pct in grouped[key]:
                    try:
                        path = grouped[key][stop_pct]
                        df = load_entropy_with_time(path)
                        mean_entropy = df["Entropy"].mean()
                        row.append(mean_entropy)
                    except Exception:
                        row.append(0.0)
                else:
                    row.append(0.0)
            data.append(row)

        # Create heatmap
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=10)
        ax.set_xticks(range(len(mixes_values)))
        ax.set_xticklabels(mixes_values)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios)
        ax.set_title(f"Stop {stop_pct}% - Mean Entropy")
        ax.set_xlabel("Mixes per layer")

        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(mixes_values)):
                text = ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax, label="Mean Entropy")

    fig.tight_layout()
    out_path = PLOTS_DIR / "entropy_heatmap_vs_stop_percent.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()