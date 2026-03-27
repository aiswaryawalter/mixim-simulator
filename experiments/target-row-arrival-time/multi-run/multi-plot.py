#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
PLOTS_DIR = FILES_DIR / "plots"

SCENARIO_ORDER = ["baseline", "client_dummies", "link_based_dummies", "multiple_hop_dummies"]
COLORS = {
    "baseline": "#1f77b4",
    "client_dummies": "#2ca02c",
    "link_based_dummies": "#ff7f0e",
    "multiple_hop_dummies": "#d62728",
}


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    avg_files = sorted(FILES_DIR.glob("summary_avg_mean_entropy_*.csv"))
    if not avg_files:
        raise FileNotFoundError(f"No summary_avg_mean_entropy_*.csv found in {FILES_DIR}")

    # Use latest averaged summary
    summary_file = avg_files[-1]
    df = pd.read_csv(summary_file)

    required = {"scenario", "l_mixes_per_layer", "avg_mean_entropy"}
    if not required.issubset(df.columns):
        raise ValueError(f"{summary_file} missing columns: {required}")

    df["l_mixes_per_layer"] = pd.to_numeric(df["l_mixes_per_layer"], errors="coerce")
    df["avg_mean_entropy"] = pd.to_numeric(df["avg_mean_entropy"], errors="coerce")
    df = df.dropna(subset=["l_mixes_per_layer", "avg_mean_entropy"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario in SCENARIO_ORDER:
        sdf = df[df["scenario"] == scenario].sort_values("l_mixes_per_layer")
        if sdf.empty:
            continue

        ax.plot(
            sdf["l_mixes_per_layer"],
            sdf["avg_mean_entropy"],
            marker="o",
            linewidth=2.2,
            label=scenario,
            color=COLORS.get(scenario, None),
        )

    ax.set_title("Average Mean Entropy by Mixes per Layer")
    ax.set_xlabel("l_mixes_per_layer")
    ax.set_ylabel("avg_mean_entropy")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PLOTS_DIR / f"entropy_comparison_{ts}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Read summary: {summary_file}")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()