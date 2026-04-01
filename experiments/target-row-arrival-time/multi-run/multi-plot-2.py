#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_FILES_DIR = ROOT / "files"
DEFAULT_PLOTS_DIR = DEFAULT_FILES_DIR / "plots"

ENTROPY_PATTERN = re.compile(
    r"^(?P<scenario>.+)_L(?P<n_layers>\d+)_M(?P<mixes>\d+)_R(?P<repeat>\d+)_Entropy\.csv$"
)

SCENARIO_NORMALIZE = {
    "baseline": "baseline",
    "client_dummies": "client_dummies",
    "clientdummy": "client_dummies",
    "client_dummy": "client_dummies",
    "link_based_dummies": "link_based_dummies",
    "linkdummy": "link_based_dummies",
    "link_dummy": "link_based_dummies",
    "multiple_hop_dummies": "multiple_hop_dummies",
    "multiplehopdummy": "multiple_hop_dummies",
    "multihopdummy": "multiple_hop_dummies",
    "multi_hop_dummy": "multiple_hop_dummies",
}

SCENARIO_ORDER = [
    "baseline",
    "client_dummies",
    "link_based_dummies",
    "multiple_hop_dummies",
]

SCENARIO_LABELS = {
    "baseline": "baseline",
    "client_dummies": "client dummy",
    "link_based_dummies": "link dummy",
    "multiple_hop_dummies": "multi hop dummy",
}

SCENARIO_COLORS = {
    "baseline": "#1f77b4",
    "client_dummies": "#2ca02c",
    "link_based_dummies": "#ff7f0e",
    "multiple_hop_dummies": "#d62728",
}


def normalize_scenario(name: str) -> str:
    key = name.strip().lower()
    return SCENARIO_NORMALIZE.get(key, name.strip())


def parse_entropy_filename(path: Path) -> tuple[str, int, int, int] | None:
    match = ENTROPY_PATTERN.match(path.name)
    if not match:
        return None
    scenario = normalize_scenario(match.group("scenario"))
    n_layers = int(match.group("n_layers"))
    mixes = int(match.group("mixes"))
    repeat = int(match.group("repeat"))
    return scenario, n_layers, mixes, repeat


def load_entropy_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"Entropy", "TimeReceived"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns {missing} in {path}")

    out = df[["Entropy", "TimeReceived"]].copy()
    out["Entropy"] = pd.to_numeric(out["Entropy"], errors="coerce")
    out["TimeReceived"] = pd.to_numeric(out["TimeReceived"], errors="coerce")
    out = out.dropna(subset=["Entropy", "TimeReceived"])
    out = out[out["TimeReceived"] >= 0.0]
    return out


def aggregate_entropy_by_bins(files: list[Path], bin_size: float) -> pd.DataFrame:
    all_parts: list[pd.DataFrame] = []
    for path in files:
        df = load_entropy_file(path)
        if df.empty:
            continue
        df = df.copy()
        df["bin_idx"] = (df["TimeReceived"] / bin_size).apply(math.floor).astype(int)
        all_parts.append(df[["bin_idx", "Entropy"]])

    if not all_parts:
        return pd.DataFrame(
            columns=[
                "bin_idx",
                "time_bin_start",
                "time_bin_end",
                "avg_entropy",
                "num_messages",
            ]
        )

    pooled = pd.concat(all_parts, ignore_index=True)
    agg = (
        pooled.groupby("bin_idx", as_index=False)
        .agg(
            avg_entropy=("Entropy", "mean"),
            num_messages=("Entropy", "count"),
        )
        .sort_values("bin_idx")
    )

    agg["time_bin_start"] = agg["bin_idx"] * bin_size
    agg["time_bin_end"] = agg["time_bin_start"] + bin_size
    return agg[
        ["bin_idx", "time_bin_start", "time_bin_end", "avg_entropy", "num_messages"]
    ]


def save_avg_csv(
    df: pd.DataFrame,
    out_dir: Path,
    scenario: str,
    n_layers: int,
    mixes: int,
    runs_found: int,
    timestamp: str,
) -> Path:
    out_name = f"avg_{scenario}_L{n_layers}_M{mixes}_entropy_{timestamp}.csv"
    out_path = out_dir / out_name

    out_df = df.copy()
    out_df.insert(0, "scenario", scenario)
    out_df.insert(1, "n_layers", n_layers)
    out_df.insert(2, "mixes_per_layer", mixes)
    out_df.insert(3, "runs_found", runs_found)
    out_df["time_bin_label"] = out_df["time_bin_end"].map(lambda x: f"{x:.1f}")

    out_df.to_csv(out_path, index=False)
    return out_path


def plot_comparisons(
    avg_csv_map: dict[tuple[int, int], dict[str, Path]],
    plots_dir: Path,
    timestamp: str,
) -> list[Path]:
    saved_plots: list[Path] = []
    plots_dir.mkdir(parents=True, exist_ok=True)

    for (n_layers, mixes), scen_files in sorted(avg_csv_map.items()):
        fig, ax = plt.subplots(figsize=(10, 6))

        for scenario in SCENARIO_ORDER:
            csv_path = scen_files.get(scenario)
            if not csv_path or not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            x = pd.to_numeric(df["time_bin_end"], errors="coerce")
            y = pd.to_numeric(df["avg_entropy"], errors="coerce")
            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]

            if x.empty:
                continue

            ax.plot(
                x,
                y,
                label=SCENARIO_LABELS.get(scenario, scenario),
                color=SCENARIO_COLORS.get(scenario),
                linewidth=2.0,
            )

        ax.set_title(f"Entropy vs Arrival Time | L={n_layers}, M={mixes}")
        ax.set_xlabel("Message arrival time")
        ax.set_ylabel("Average entropy in 0.2 bins")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        out_path = (
            plots_dir / f"comparison_dummy_strategies_L{n_layers}_M{mixes}_{timestamp}.png"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved_plots.append(out_path)

    by_layer: dict[int, list[int]] = {}
    for n_layers, mixes in avg_csv_map:
        by_layer.setdefault(n_layers, []).append(mixes)

    for n_layers, mixes_list in sorted(by_layer.items()):
        mixes_sorted = sorted(set(mixes_list))
        n = len(mixes_sorted)
        if n == 0:
            continue

        ncols = 4 if n >= 4 else n
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), sharey=True
        )

        if nrows == 1 and ncols == 1:
            axes_flat = [axes]
        elif nrows == 1:
            axes_flat = list(axes)
        elif ncols == 1:
            axes_flat = list(axes)
        else:
            axes_flat = [ax for row in axes for ax in row]

        for idx, mixes in enumerate(mixes_sorted):
            ax = axes_flat[idx]
            scen_files = avg_csv_map.get((n_layers, mixes), {})

            for scenario in SCENARIO_ORDER:
                csv_path = scen_files.get(scenario)
                if not csv_path or not csv_path.exists():
                    continue

                df = pd.read_csv(csv_path)
                if df.empty:
                    continue

                x = pd.to_numeric(df["time_bin_end"], errors="coerce")
                y = pd.to_numeric(df["avg_entropy"], errors="coerce")
                valid = x.notna() & y.notna()
                x = x[valid]
                y = y[valid]

                if x.empty:
                    continue

                ax.plot(
                    x,
                    y,
                    label=SCENARIO_LABELS.get(scenario, scenario),
                    color=SCENARIO_COLORS.get(scenario),
                    linewidth=1.8,
                )

            ax.set_title(f"M={mixes}")
            ax.set_xlabel("Message arrival time")
            ax.set_ylabel("Avg entropy")
            ax.grid(True, alpha=0.3)

        for idx in range(len(mixes_sorted), len(axes_flat)):
            axes_flat[idx].axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

        fig.suptitle(
            f"Dummy strategy comparison by arrival time bins | L={n_layers}", y=0.99
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = plots_dir / f"comparison_dummy_strategies_grid_L{n_layers}_{timestamp}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved_plots.append(out_path)

    return saved_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Average entropy across runs in 0.2 arrival-time bins and plot "
            "dummy-strategy comparisons."
        )
    )
    parser.add_argument("--files-dir", type=Path, default=DEFAULT_FILES_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--bin-size", type=float, default=0.2)
    parser.add_argument("--expected-repeats", type=int, default=10)
    args = parser.parse_args()

    files_dir: Path = args.files_dir
    plots_dir: Path = args.plots_dir
    bin_size: float = args.bin_size
    expected_repeats: int = args.expected_repeats

    if bin_size <= 0:
        raise ValueError("bin-size must be > 0")

    files_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    entropy_files = sorted(files_dir.glob("*_Entropy.csv"))
    if not entropy_files:
        raise FileNotFoundError(f"No entropy files found in {files_dir}")

    grouped: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    for path in entropy_files:
        parsed = parse_entropy_filename(path)
        if not parsed:
            continue
        scenario, n_layers, mixes, repeat = parsed
        key = (scenario, n_layers, mixes)
        grouped.setdefault(key, []).append((repeat, path))

    if not grouped:
        raise RuntimeError("No files matched expected entropy naming pattern.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    avg_csv_map: dict[tuple[int, int], dict[str, Path]] = {}
    manifest_rows: list[dict[str, object]] = []

    for (scenario, n_layers, mixes), entries in sorted(
        grouped.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])
    ):
        entries_sorted = sorted(entries, key=lambda x: x[0])
        repeats = [repeat for repeat, _ in entries_sorted]
        files = [path for _, path in entries_sorted]

        if len(files) != expected_repeats:
            print(
                f"Warning: {scenario} L={n_layers} M={mixes} has {len(files)} runs "
                f"(expected {expected_repeats}). Repeats found: {repeats}"
            )

        agg = aggregate_entropy_by_bins(files, bin_size=bin_size)
        out_csv = save_avg_csv(
            agg,
            out_dir=files_dir,
            scenario=scenario,
            n_layers=n_layers,
            mixes=mixes,
            runs_found=len(files),
            timestamp=timestamp,
        )

        avg_csv_map.setdefault((n_layers, mixes), {})[scenario] = out_csv
        manifest_rows.append(
            {
                "scenario": scenario,
                "n_layers": n_layers,
                "mixes_per_layer": mixes,
                "runs_found": len(files),
                "avg_csv": str(out_csv),
            }
        )
        print(f"Saved avg CSV: {out_csv}")

    manifest_path = files_dir / f"avg_entropy_manifest_{timestamp}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")

    plot_paths = plot_comparisons(avg_csv_map, plots_dir=plots_dir, timestamp=timestamp)
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()