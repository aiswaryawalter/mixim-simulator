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
COLORS = {
    "baseline": "#1f77b4",
    "client_dummies": "#2ca02c",
    "link_based_dummies": "#ff7f0e",
    "multiple_hop_dummies": "#d62728",
}


def parse_name(path: Path) -> tuple[str, int] | None:
    # e.g. baseline_L3_M5_Entropy.csv
    m = re.match(r"(.+)_L(\d+)_M(\d+)_Entropy\.csv$", path.name)
    if not m:
        return None
    scenario = m.group(1)
    mixes = int(m.group(3))
    return scenario, mixes


def load_entropy_with_arrival(entropy_file: Path) -> pd.DataFrame:
    """
    Returns dataframe with columns:
    - MessageID
    - arrival_time
    - Entropy
    """
    df_entropy = pd.read_csv(entropy_file)

    # Preferred: already exported by Simulation.py into entropy output
    if "TimeReceived" in df_entropy.columns:
        df = df_entropy[["MessageID", "TimeReceived", "Entropy"]].copy()
        df.rename(columns={"TimeReceived": "arrival_time"}, inplace=True)
        return df

    # Fallback: join with corresponding ReceivedMessages file
    recv_file = Path(str(entropy_file).replace("_Entropy.csv", "_ReceivedMessages.csv"))
    if not recv_file.exists():
        raise FileNotFoundError(f"Missing received messages file: {recv_file}")

    df_recv = pd.read_csv(recv_file)
    needed = {"MessageID", "MessageTimeReceived"}
    if not needed.issubset(df_recv.columns):
        raise ValueError(f"{recv_file} missing columns {needed}")

    df = df_entropy.merge(df_recv[["MessageID", "MessageTimeReceived"]], on="MessageID", how="left")
    df.rename(columns={"MessageTimeReceived": "arrival_time"}, inplace=True)
    return df[["MessageID", "arrival_time", "Entropy"]]


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    entropy_files = sorted(FILES_DIR.glob("*_Entropy.csv"))
    if not entropy_files:
        raise FileNotFoundError(f"No *_Entropy.csv files found in {FILES_DIR}")

    grouped: dict[int, dict[str, Path]] = {}
    for f in entropy_files:
        parsed = parse_name(f)
        if not parsed:
            continue
        scenario, mixes = parsed
        grouped.setdefault(mixes, {})[scenario] = f

    if not grouped:
        raise RuntimeError("No files matched expected naming pattern *_L*_M*_Entropy.csv")

    mixes_values = sorted(grouped.keys())

    # 2x4 layout for mixes 3..10
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)
    axes = axes.flatten()

    for idx, mixes in enumerate(mixes_values):
        ax = axes[idx]
        present_scenarios = grouped[mixes]

        for scenario in SCENARIO_ORDER:
            if scenario not in present_scenarios:
                continue

            path = present_scenarios[scenario]
            df = load_entropy_with_arrival(path)
            df = df.dropna(subset=["arrival_time", "Entropy"]).sort_values("arrival_time")

            df["EntropySmoothed"] = df["Entropy"].rolling(window=5, min_periods=1).mean()

            ax.plot(
                df["arrival_time"],
                df["Entropy"],
                label=scenario,
                color=COLORS.get(scenario, None),
                linewidth=1.8,
                alpha=0.95,
            )

        ax.set_title(f"l_mixes_per_layer = {mixes}")
        ax.set_xlabel("Message arrival time")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.25)

    # Hide unused axes if any
    for j in range(len(mixes_values), len(axes)):
        axes[j].axis("off")

    # Single legend for whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    fig.suptitle("Entropy progression over message arrival time", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"entropy_progression_{ts}.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()