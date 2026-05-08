#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError, ParserError



ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ConfigFile.ini"
LOGS_DIR = ROOT / "Logs"
OUT_DIR = ROOT / "files"

N_LAYERS = 3
MIXES_PER_LAYER = 10
DEFAULT_REPEATS = 10
DEFAULT_BIN_SIZE = 0.5

CORRUPT_MIXES = [0, 3, 6]
E2E_VALUES = [1, 3, 5]
START_DUMMY_COUNTS = [0]
MSG_DELIVERY_PERCENTS = [80, 90, 95, 99]
RHO = 2.0

SCENARIOS = {
    "baseline": {
        "client_dummies": False,
        "link_based_dummies": False,
        "multiple_hop_dummies": False,
    },
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

def safe_read_csv(path: Path, required: set[str] | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, PermissionError, OSError, EmptyDataError, ParserError) as exc:
        print(f"WARNING: could not read CSV {path}: {exc}")
        return pd.DataFrame()

    if required and not required.issubset(df.columns):
        print(f"WARNING: CSV {path} missing columns {sorted(required - set(df.columns))}")
        return pd.DataFrame()

    return df

def safe_read_json(path: Path) -> dict:
    try:
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read JSON {path}: {exc}")
    return {}

def bool_str(v: bool) -> str:
    return "True" if v else "False"


def build_combos() -> list[dict]:
    combos: list[dict] = []
    combo_id = 1

    for corrupt_mixes, e2e, start_dummy_count, msg_delivery_percent, scenario_name in itertools.product(
        CORRUPT_MIXES, E2E_VALUES, START_DUMMY_COUNTS, MSG_DELIVERY_PERCENTS, SCENARIOS.keys()
    ):
        flags = SCENARIOS[scenario_name]
        combos.append(
            {
                "combo_id": combo_id,
                "corrupt_mixes": corrupt_mixes,
                "E2E": e2e,
                "start_dummy_count": start_dummy_count,
                "msg_delivery_percent": msg_delivery_percent,
                "scenario": scenario_name,
                "client_dummies": flags["client_dummies"],
                "link_based_dummies": flags["link_based_dummies"],
                "multiple_hop_dummies": flags["multiple_hop_dummies"],
                "rho": RHO,
            }
        )
        combo_id += 1

    return combos


def load_experiment_matrix(campaign_dir: Path) -> pd.DataFrame:
    matrix_path = campaign_dir / "experiment_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing experiment matrix: {matrix_path}")

    matrix = pd.read_csv(
        matrix_path,
        dtype={
            "client_dummies": "string",
            "link_based_dummies": "string",
            "multiple_hop_dummies": "string",
        },
    )

    required = {
        "combo_id",
        "corrupt_mixes",
        "E2E",
        "start_dummy_count",
        "msg_delivery_percent",
        "scenario",
        "client_dummies",
        "link_based_dummies",
        "multiple_hop_dummies",
        "rho",
    }
    missing = required - set(matrix.columns)
    if missing:
        raise ValueError(f"{matrix_path} missing columns: {sorted(missing)}")

    return matrix.sort_values("combo_id").reset_index(drop=True)

def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def select_combo(combo_id: int, matrix: pd.DataFrame) -> dict:
    match = matrix.loc[matrix["combo_id"] == combo_id]
    if match.empty:
        raise ValueError(f"combo-id {combo_id} not found in experiment matrix")

    row = match.iloc[0]
    return {
        "combo_id": int(row["combo_id"]),
        "corrupt_mixes": int(row["corrupt_mixes"]),
        "E2E": int(row["E2E"]),
        "start_dummy_count": int(row["start_dummy_count"]),
        "msg_delivery_percent": int(row["msg_delivery_percent"]),
        "scenario": str(row["scenario"]),
        "client_dummies": parse_bool(row["client_dummies"]),
        "link_based_dummies": parse_bool(row["link_based_dummies"]),
        "multiple_hop_dummies": parse_bool(row["multiple_hop_dummies"]),
        "rho": float(row["rho"]),
    }

def write_experiment_matrix(campaign_dir: Path) -> Path:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = campaign_dir / "experiment_matrix.csv"
    pd.DataFrame(build_combos()).to_csv(matrix_path, index=False)
    return matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one experiment combo, export the experiment matrix, or plot results."
    )
    parser.add_argument("--combo-id", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--campaign-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bin-size", type=float, default=DEFAULT_BIN_SIZE)
    parser.add_argument(
        "--write-matrix",
        action="store_true",
        help="Write experiment_matrix.csv and exit.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate transition-window entropy plots from finished combo outputs.",
    )
    parser.add_argument("--filter-corrupt", type=int, default=0,
                        help="Filter combos by corrupt_mixes (default 0).")
    parser.add_argument("--filter-e2e", type=int, default=1,
                        help="Filter combos by E2E (default 1).")
    parser.add_argument("--filter-start-dummy", type=int, default=0,
                        help="Filter combos by start_dummy_count (default 0).")
    args = parser.parse_args()

    if not args.write_matrix and not args.plot_only and args.combo_id is None:
        parser.error("--combo-id is required unless --write-matrix or --plot-only is set")

    if args.repeats <= 0:
        parser.error("--repeats must be > 0")

    if args.bin_size <= 0:
        parser.error("--bin-size must be > 0")

    return args


def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    return cfg


def write_config(cfg: configparser.ConfigParser) -> None:
    with CONFIG_PATH.open("w") as f:
        cfg.write(f)


def apply_combo_to_config(cfg: configparser.ConfigParser, combo: dict) -> None:
    cfg["TOPOLOGY"]["n_layers"] = str(N_LAYERS)
    cfg["TOPOLOGY"]["l_mixes_per_layer"] = str(MIXES_PER_LAYER)
    cfg["TOPOLOGY"]["E2E"] = str(combo["E2E"])

    cfg["THREATMODEL"]["corrupt_mixes"] = str(combo["corrupt_mixes"])

    cfg["MIXING"]["msg_delivery_percent"] = str(combo["msg_delivery_percent"])

    cfg["DUMMIES"]["client_dummies"] = bool_str(combo["client_dummies"])
    cfg["DUMMIES"]["link_based_dummies"] = bool_str(combo["link_based_dummies"])
    cfg["DUMMIES"]["multiple_hop_dummies"] = bool_str(combo["multiple_hop_dummies"])
    cfg["DUMMIES"]["start_dummy_count"] = str(combo["start_dummy_count"])
    cfg["DUMMIES"]["rho"] = str(combo["rho"])

    # if "rate_client_dummies" in cfg["DUMMIES"]:
    #     cfg["DUMMIES"]["rate_client_dummies"] = "1"
    # if "rate_mix_dummies" in cfg["DUMMIES"]:
    #     cfg["DUMMIES"]["rate_mix_dummies"] = "1"


def run_main() -> None:
    subprocess.run([sys.executable, "main.py"], cwd=ROOT, check=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def snapshot_outputs(combo_dir: Path, repeat_idx: int) -> dict[str, str]:
    run_dir = combo_dir / f"run_{repeat_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    entropy_src = LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_player_Entropy.csv"
    targets_src = LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_player_Targets.csv"
    sent_src = LOGS_DIR / "SentMessages.csv"
    recv_src = LOGS_DIR / "ReceivedMessages.csv"
    dummy_src = LOGS_DIR / "DummyMessages.csv"
    link_load_src = LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_LinkLoad.csv"
    link_summary_src = LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_LinkSummary.csv"
    run_meta_src = LOGS_DIR / f"{N_LAYERS}layers_{MIXES_PER_LAYER}mixes_RunMeta.json"


    entropy_dst = run_dir / "Entropy.csv"
    targets_dst = run_dir / "Targets.csv"
    sent_dst = run_dir / "SentMessages.csv"
    recv_dst = run_dir / "ReceivedMessages.csv"
    dummy_dst = run_dir / "DummyMessages.csv"
    link_load_dst = run_dir / "LinkLoad.csv"
    link_summary_dst = run_dir / "LinkSummary.csv"
    run_meta_dst = run_dir / "RunMeta.json"

    copy_if_exists(entropy_src, entropy_dst)
    copy_if_exists(targets_src, targets_dst)
    copy_if_exists(sent_src, sent_dst)
    copy_if_exists(recv_src, recv_dst)
    copy_if_exists(dummy_src, dummy_dst)
    copy_if_exists(link_load_src, link_load_dst)
    copy_if_exists(link_summary_src, link_summary_dst)
    copy_if_exists(run_meta_src, run_meta_dst)

    return {
        "entropy": str(entropy_dst) if entropy_dst.exists() else "",
        "targets": str(targets_dst) if targets_dst.exists() else "",
        "sent": str(sent_dst) if sent_dst.exists() else "",
        "received": str(recv_dst) if recv_dst.exists() else "",
        "dummy": str(dummy_dst) if dummy_dst.exists() else "",
        "link_load": str(link_load_dst) if link_load_dst.exists() else "",
        "link_summary": str(link_summary_dst) if link_summary_dst.exists() else "",
        "run_meta": str(run_meta_dst) if run_meta_dst.exists() else "",
    }

def build_run_entropy_bins(entropy_csv: Path, bin_size: float) -> pd.DataFrame:
    df = safe_read_csv(entropy_csv, required={"Entropy", "TimeReceived"})
    if df.empty:
        return pd.DataFrame(
            columns=["bin_idx", "time_bin_start", "time_bin_end", "avg_entropy", "num_messages"]
        )

    part = df[["Entropy", "TimeReceived"]].copy()
    part["Entropy"] = pd.to_numeric(part["Entropy"], errors="coerce")
    part["TimeReceived"] = pd.to_numeric(part["TimeReceived"], errors="coerce")
    part = part.dropna(subset=["Entropy", "TimeReceived"])
    part = part[part["TimeReceived"] >= 0]
    if part.empty:
        return pd.DataFrame(
            columns=["bin_idx", "time_bin_start", "time_bin_end", "avg_entropy", "num_messages"]
        )

    part["bin_idx"] = (part["TimeReceived"] / bin_size).apply(math.floor).astype(int)

    run_bins = (
        part.groupby("bin_idx", as_index=False)
        .agg(
            avg_entropy=("Entropy", "mean"),
            num_messages=("Entropy", "count"),
        )
        .sort_values("bin_idx")
    )

    run_bins["time_bin_start"] = run_bins["bin_idx"] * bin_size
    run_bins["time_bin_end"] = run_bins["time_bin_start"] + bin_size

    return run_bins[
        ["bin_idx", "time_bin_start", "time_bin_end", "avg_entropy", "num_messages"]
    ]

def compute_run_metrics(outputs: dict[str, str]) -> dict[str, str]:
    sent_count = 0
    received_count = 0
    real_sent_count = 0
    real_received_count = 0
    dummy_count = 0
    dummy_breakdown: dict[str, int] = {}
    mean_entropy = 0.0

    sim_duration_config = 0.0
    sim_end_time = 0.0
    sim_extra_time = 0.0

    link_total_msgs = 0
    link_real_msgs = 0
    link_dummy_msgs = 0
    avg_msgs_per_link = 0.0
    avg_real_per_link = 0.0
    avg_dummy_per_link = 0.0
    rate_per_link_per_time = 0.0
    rate_real_per_link_per_time = 0.0
    rate_dummy_per_link_per_time = 0.0

    if outputs.get("sent"):
        df_sent = safe_read_csv(outputs["sent"])
        if not df_sent.empty:
            sent_count = len(df_sent)
            if "MessageType" in df_sent.columns:
                real_sent_count = int((df_sent["MessageType"] == "Real").sum())

    if outputs.get("received"):
        df_recv = safe_read_csv(outputs["received"])
        if not df_recv.empty:
            received_count = len(df_recv)
            if "MessageType" in df_recv.columns:
                real_received_count = int((df_recv["MessageType"] == "Real").sum())

    if outputs.get("dummy"):
        df_dummy = safe_read_csv(outputs["dummy"])
        if not df_dummy.empty:
            dummy_count = len(df_dummy)
            if "DummyType" in df_dummy.columns:
                counts = df_dummy["DummyType"].value_counts(dropna=False).to_dict()
                dummy_breakdown = {str(k): int(v) for k, v in counts.items()}

    if outputs.get("entropy"):
        df_entropy = safe_read_csv(outputs["entropy"], required={"Entropy"})
        if not df_entropy.empty and "Entropy" in df_entropy.columns:
            mean_entropy = float(df_entropy["Entropy"].mean())

    if outputs.get("run_meta"):
        run_meta = safe_read_json(Path(outputs["run_meta"]))
        sim_duration_config = float(run_meta.get("sim_duration_config", 0.0))
        sim_end_time = float(run_meta.get("sim_end_time", 0.0))
        sim_extra_time = float(run_meta.get("sim_extra_time", 0.0))

    if outputs.get("link_summary"):
        df_link_summary = safe_read_csv(outputs["link_summary"])
        if not df_link_summary.empty:
            overall = df_link_summary[
                (df_link_summary["FromLayer"].astype(str) == "ALL")
                & (df_link_summary["ToLayer"].astype(str) == "ALL")
            ]
            if not overall.empty:
                row = overall.iloc[0]
                link_total_msgs = int(row["TotalMsgs"])
                link_real_msgs = int(row["RealMsgs"])
                link_dummy_msgs = int(row["DummyMsgs"])
                avg_msgs_per_link = float(row["AvgMsgsPerLink"])
                avg_real_per_link = float(row["AvgRealPerLink"])
                avg_dummy_per_link = float(row["AvgDummyPerLink"])
                rate_per_link_per_time = float(row["RatePerLinkPerTime"])
                rate_real_per_link_per_time = float(row["RateRealPerLinkPerTime"])
                rate_dummy_per_link_per_time = float(row["RateDummyPerLinkPerTime"])
    return {
        "sent_msgs": str(sent_count),
        "received_msgs": str(received_count),
        "real_sent_msgs": str(real_sent_count),
        "real_received_msgs": str(real_received_count),
        "dummy_msgs_total": str(dummy_count),
        "dummy_types_breakdown": json.dumps(dummy_breakdown, sort_keys=True),
        "mean_entropy": f"{mean_entropy:.8f}",
        "sim_duration_config": f"{sim_duration_config:.8f}",
        "sim_end_time": f"{sim_end_time:.8f}",
        "sim_extra_time": f"{sim_extra_time:.8f}",
        "link_total_msgs": str(link_total_msgs),
        "link_real_msgs": str(link_real_msgs),
        "link_dummy_msgs": str(link_dummy_msgs),
        "avg_msgs_per_link": f"{avg_msgs_per_link:.8f}",
        "avg_real_per_link": f"{avg_real_per_link:.8f}",
        "avg_dummy_per_link": f"{avg_dummy_per_link:.8f}",
        "rate_per_link_per_time": f"{rate_per_link_per_time:.8f}",
        "rate_real_per_link_per_time": f"{rate_real_per_link_per_time:.8f}",
        "rate_dummy_per_link_per_time": f"{rate_dummy_per_link_per_time:.8f}",
    }


def aggregate_entropy_bins_from_run_bins(run_bin_paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for p in run_bin_paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        needed = {"repeat_idx", "bin_idx", "time_bin_start", "time_bin_end", "avg_entropy", "num_messages"}
        if not needed.issubset(d.columns):
            continue
        parts.append(d)

    if not parts:
        return pd.DataFrame(
            columns=[
                "bin_idx",
                "time_bin_start",
                "time_bin_end",
                "avg_entropy",
                "std_entropy",
                "runs_with_data",
                "avg_messages_per_run",
                "total_messages",
            ]
        )

    pooled = pd.concat(parts, ignore_index=True)

    agg = (
        pooled.groupby(["bin_idx", "time_bin_start", "time_bin_end"], as_index=False)
        .agg(
            avg_entropy=("avg_entropy", "mean"),          # mean over runs
            std_entropy=("avg_entropy", "std"),
            runs_with_data=("avg_entropy", "count"),
            avg_messages_per_run=("num_messages", "mean"),
            total_messages=("num_messages", "sum"),
        )
        .sort_values("bin_idx")
    )

    agg["std_entropy"] = agg["std_entropy"].fillna(0.0)
    return agg

SIM_DURATION = 20  # time at which clients stop sending real messages

DELIVERY_COLORS = {80: "#1f77b4", 90: "#ff7f0e", 95: "#2ca02c", 99: "#d62728"}
SCENARIO_LABELS = {
    "baseline": "No dummies",
    "client_dummies": "Client dummies",
    "link_based_dummies": "Link-based dummies",
    "multiple_hop_dummies": "Multi-hop dummies",
}


def plot_transition_window(
    campaign_dir: Path,
    out_path: Path,
    filter_corrupt: int = 0,
    filter_e2e: int = 1,
    filter_start_dummy: int = 0,
) -> None:
    records: list[pd.DataFrame] = []
    for combo_dir in sorted(campaign_dir.glob("combo_*")):
        meta_path = combo_dir / "combo_meta.json"
        ensemble_path = combo_dir / "ensemble_entropy_bin_0p5.csv"
        if not meta_path.exists() or not ensemble_path.exists():
            continue
        meta = safe_read_json(meta_path)
        if (
            int(meta.get("corrupt_mixes", -1)) != filter_corrupt
            or int(meta.get("E2E", -1)) != filter_e2e
            or int(meta.get("start_dummy_count", -1)) != filter_start_dummy
        ):
            continue
        df = safe_read_csv(ensemble_path)
        if df.empty:
            continue
        df["scenario"] = meta["scenario"]
        df["msg_delivery_percent"] = int(meta["msg_delivery_percent"])
        records.append(df)

    if not records:
        print(
            f"No ensemble data found for corrupt={filter_corrupt}, "
            f"E2E={filter_e2e}, start_dummy={filter_start_dummy}."
        )
        return

    data = pd.concat(records, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes_flat = axes.flatten()

    for ax, (scenario_key, scenario_label) in zip(axes_flat, SCENARIO_LABELS.items()):
        sub = data[data["scenario"] == scenario_key]
        for pct in sorted(DELIVERY_COLORS):
            rows = sub[sub["msg_delivery_percent"] == pct].sort_values("time_bin_start")
            if rows.empty:
                continue
            color = DELIVERY_COLORS[pct]
            ax.plot(
                rows["time_bin_start"],
                rows["avg_entropy"],
                color=color,
                linewidth=1.5,
                label=f"{pct}% delivered",
            )
            ax.fill_between(
                rows["time_bin_start"],
                rows["avg_entropy"] - rows["std_entropy"],
                rows["avg_entropy"] + rows["std_entropy"],
                alpha=0.15,
                color=color,
            )
        ax.axvline(
            x=SIM_DURATION,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"Stop real msgs (t={SIM_DURATION})",
        )
        ax.set_title(scenario_label)
        ax.set_xlabel("Arrival Time")
        ax.set_ylabel("Entropy (bits)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"Entropy vs Arrival Time — Transition Window\n"
        f"(corrupt={filter_corrupt}, E2E={filter_e2e}, start_dummy={filter_start_dummy})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main() -> None:
    args = parse_args()

    if args.write_matrix:
        matrix_path = write_experiment_matrix(args.campaign_dir)
        print(f"Saved experiment matrix: {matrix_path}")
        return

    if args.plot_only:
        out_path = args.campaign_dir / (
            f"transition_window_c{args.filter_corrupt}"
            f"_e{args.filter_e2e}_sd{args.filter_start_dummy}.png"
        )
        plot_transition_window(
            args.campaign_dir,
            out_path,
            filter_corrupt=args.filter_corrupt,
            filter_e2e=args.filter_e2e,
            filter_start_dummy=args.filter_start_dummy,
        )
        return

    campaign_dir = args.campaign_dir
    matrix = load_experiment_matrix(campaign_dir)
    combo = select_combo(args.combo_id, matrix)
    combo_dir = campaign_dir / f"combo_{combo['combo_id']:03d}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    original_config_text = CONFIG_PATH.read_text()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_run_rows: list[dict[str, str]] = []
    entropy_paths: list[Path] = []
    run_bin_paths: list[Path] = []

    try:
        for rep in range(1, args.repeats + 1):
            cfg = load_config()
            apply_combo_to_config(cfg, combo)
            write_config(cfg)

            print(
                f"Run {rep}/{args.repeats}: "
                f"combo_id={combo['combo_id']}, scenario={combo['scenario']}, "
                f"corrupt_mixes={combo['corrupt_mixes']}, E2E={combo['E2E']}, "
                f"start_dummy_count={combo['start_dummy_count']}, "
                f"msg_delivery_percent={combo['msg_delivery_percent']}, "
            )

            run_main()

            outputs = snapshot_outputs(combo_dir, rep)
            metrics = compute_run_metrics(outputs)

            row = {
                "run_timestamp": run_timestamp,
                "combo_id": str(combo["combo_id"]),
                "repeat_idx": str(rep),
                "scenario": combo["scenario"],
                "corrupt_mixes": str(combo["corrupt_mixes"]),
                "E2E": str(combo["E2E"]),
                "start_dummy_count": str(combo["start_dummy_count"]),
                "msg_delivery_percent": str(combo["msg_delivery_percent"]),
                "client_dummies": bool_str(combo["client_dummies"]),
                "link_based_dummies": bool_str(combo["link_based_dummies"]),
                "multiple_hop_dummies": bool_str(combo["multiple_hop_dummies"]),
                "rho": str(combo["rho"]),
                **metrics,
                **outputs,
            }
            per_run_rows.append(row)

            if outputs["entropy"]:
                entropy_path = Path(outputs["entropy"])
                entropy_paths.append(entropy_path)

                run_bins = build_run_entropy_bins(entropy_path, args.bin_size)
                run_bins.insert(0, "repeat_idx", rep)

                run_bin_path = combo_dir / f"run_{rep:02d}" / "EntropyBins_0p5.csv"
                run_bins.to_csv(run_bin_path, index=False, float_format="%.6f")
                run_bin_paths.append(run_bin_path)

    finally:
        CONFIG_PATH.write_text(original_config_text)

    per_run_path = combo_dir / "summary_per_run.csv"
    pd.DataFrame(per_run_rows).to_csv(per_run_path, index=False, float_format="%.6f")

    ensemble_df = aggregate_entropy_bins_from_run_bins(run_bin_paths)    
    ensemble_df.insert(1, "combo_id", combo["combo_id"])
    ensemble_df.insert(2, "scenario", combo["scenario"])
    ensemble_df.insert(3, "corrupt_mixes", combo["corrupt_mixes"])
    ensemble_df.insert(4, "E2E", combo["E2E"])
    ensemble_df.insert(5, "start_dummy_count", combo["start_dummy_count"])
    ensemble_df.insert(6, "msg_delivery_percent", combo["msg_delivery_percent"])
    ensemble_df.insert(7, "repeats_requested", args.repeats)
    ensemble_df.insert(8, "runs_found", len(entropy_paths))

    ensemble_path = combo_dir / "ensemble_entropy_bin_0p5.csv"
    ensemble_df.to_csv(ensemble_path, index=False, float_format="%.6f")

    meta = {
        "run_timestamp": run_timestamp,
        "combo_id": combo["combo_id"],
        "scenario": combo["scenario"],
        "corrupt_mixes": combo["corrupt_mixes"],
        "E2E": combo["E2E"],
        "start_dummy_count": combo["start_dummy_count"],
        "msg_delivery_percent": combo["msg_delivery_percent"],
        "client_dummies": combo["client_dummies"],
        "link_based_dummies": combo["link_based_dummies"],
        "multiple_hop_dummies": combo["multiple_hop_dummies"],
        "rho": combo["rho"],
        "n_layers": N_LAYERS,
        "mixes_per_layer": MIXES_PER_LAYER,
        "repeats_requested": args.repeats,
        "repeats_completed": len(per_run_rows),
        "runs_with_entropy": len(entropy_paths),
        "bin_size": args.bin_size,
        "config_path": str(CONFIG_PATH),
    }

    summary_df = pd.DataFrame(per_run_rows)

    summary_row = {
        "run_timestamp": run_timestamp,
        "combo_id": combo["combo_id"],
        "scenario": combo["scenario"],
        "corrupt_mixes": combo["corrupt_mixes"],
        "E2E": combo["E2E"],
        "start_dummy_count": combo["start_dummy_count"],
        "msg_delivery_percent": combo["msg_delivery_percent"],
        "repeats": len(summary_df),
        "avg_sent_msgs": float(summary_df["sent_msgs"].astype(float).mean()),
        "avg_received_msgs": float(summary_df["received_msgs"].astype(float).mean()),
        "avg_real_sent_msgs": float(summary_df["real_sent_msgs"].astype(float).mean()),
        "avg_real_received_msgs": float(summary_df["real_received_msgs"].astype(float).mean()),
        "avg_dummy_msgs": float(summary_df["dummy_msgs_total"].astype(float).mean()),
        "avg_sim_end_time": float(summary_df["sim_end_time"].astype(float).mean()),
        "avg_sim_extra_time": float(summary_df["sim_extra_time"].astype(float).mean()),
        "max_sim_end_time": float(summary_df["sim_end_time"].astype(float).max()),
        "max_sim_extra_time": float(summary_df["sim_extra_time"].astype(float).max()),
        "avg_link_total_msgs": float(summary_df["link_total_msgs"].astype(float).mean()),
        "avg_link_real_msgs": float(summary_df["link_real_msgs"].astype(float).mean()),
        "avg_link_dummy_msgs": float(summary_df["link_dummy_msgs"].astype(float).mean()),
        "avg_rate_per_link_per_time": float(summary_df["rate_per_link_per_time"].astype(float).mean()),
        "avg_rate_real_per_link_per_time": float(summary_df["rate_real_per_link_per_time"].astype(float).mean()),
        "avg_rate_dummy_per_link_per_time": float(summary_df["rate_dummy_per_link_per_time"].astype(float).mean()),
        "mean_entropy": float(summary_df["mean_entropy"].astype(float).mean()),
        "std_entropy": float(summary_df["mean_entropy"].astype(float).std(ddof=1)),
    }

    total_sent = summary_df["sent_msgs"].astype(float).sum()
    total_received = summary_df["received_msgs"].astype(float).sum()
    total_real_sent = summary_df["real_sent_msgs"].astype(float).sum()
    total_real_received = summary_df["real_received_msgs"].astype(float).sum()
    total_dummy = summary_df["dummy_msgs_total"].astype(float).sum()

    summary_row["pct_real_sent_msgs"] = 100.0 * total_real_sent / total_sent if total_sent else 0.0
    summary_row["pct_real_received_msgs"] = 100.0 * total_real_received / total_received if total_received else 0.0
    summary_row["pct_dummy_msgs"] = 100.0 * total_dummy / total_sent if total_sent else 0.0

    combo_summary_path = combo_dir / "summary_combo.csv"
    pd.DataFrame([summary_row]).to_csv(combo_summary_path, index=False, float_format="%.6f")
    print(f"Saved combo summary: {combo_summary_path}")

    meta_path = combo_dir / "combo_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    print(f"Saved per-run summary: {per_run_path}")
    print(f"Saved ensemble summary: {ensemble_path}")
    print(f"Saved combo metadata: {meta_path}")


if __name__ == "__main__":
    main()