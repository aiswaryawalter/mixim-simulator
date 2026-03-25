#!/usr/bin/env python3
from __future__ import annotations

import configparser
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ConfigFile.ini"
LOGS_DIR = ROOT / "Logs"
OUT_DIR = ROOT / "files-multi-run"

N_LAYERS = 3
MIXES_PER_LAYER = [3, 4, 5, 6, 7, 8, 9, 10]
REPEATS = 10

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
    "baseline": {
        "client_dummies": False,
        "link_based_dummies": False,
        "multiple_hop_dummies": False,
    },
}


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
    subprocess.run([sys.executable, "main.py"], cwd=ROOT, check=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def snapshot_outputs(
    scenario: str, n_layers: int, mixes: int, repeat_idx: int
) -> dict[str, str]:
    prefix = f"{scenario}_L{n_layers}_M{mixes}_R{repeat_idx:02d}"

    entropy_src = LOGS_DIR / f"{n_layers}layers_{mixes}mixes_player_Entropy.csv"
    targets_src = LOGS_DIR / f"{n_layers}layers_{mixes}mixes_player_Targets.csv"
    sent_src = LOGS_DIR / "SentMessages.csv"
    recv_src = LOGS_DIR / "ReceivedMessages.csv"
    dummy_src = LOGS_DIR / "DummyMessages.csv"

    entropy_dst = OUT_DIR / f"{prefix}_Entropy.csv"
    targets_dst = OUT_DIR / f"{prefix}_Targets.csv"
    sent_dst = OUT_DIR / f"{prefix}_SentMessages.csv"
    recv_dst = OUT_DIR / f"{prefix}_ReceivedMessages.csv"
    dummy_dst = OUT_DIR / f"{prefix}_DummyMessages.csv"

    copy_if_exists(entropy_src, entropy_dst)
    copy_if_exists(targets_src, targets_dst)
    copy_if_exists(sent_src, sent_dst)
    copy_if_exists(recv_src, recv_dst)
    copy_if_exists(dummy_src, dummy_dst)

    return {
        "entropy": str(entropy_dst) if entropy_dst.exists() else "",
        "targets": str(targets_dst) if targets_dst.exists() else "",
        "sent": str(sent_dst) if sent_dst.exists() else "",
        "received": str(recv_dst) if recv_dst.exists() else "",
        "dummy": str(dummy_dst) if dummy_dst.exists() else "",
    }


def compute_run_metrics(outputs: dict[str, str]) -> dict[str, str]:
    sent_count = 0
    received_count = 0
    dummy_count = 0
    dummy_breakdown: dict[str, int] = {}
    mean_entropy = 0.0

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

    return {
        "sent_msgs": str(sent_count),
        "received_msgs": str(received_count),
        "dummy_msgs_total": str(dummy_count),
        "dummy_types_breakdown": json.dumps(dummy_breakdown, sort_keys=True),
        "mean_entropy": f"{mean_entropy:.8f}",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_config_text = CONFIG_PATH.read_text()

    per_run_rows: list[dict[str, str]] = []
    aggregate_bucket: dict[tuple[str, int], list[float]] = {}

    try:
        for mixes in MIXES_PER_LAYER:
            for scenario_name, flags in SCENARIOS.items():
                for rep in range(1, REPEATS + 1):
                    cfg = load_config()

                    cfg["TOPOLOGY"]["n_layers"] = str(N_LAYERS)
                    cfg["TOPOLOGY"]["l_mixes_per_layer"] = str(mixes)

                    cfg["DUMMIES"]["client_dummies"] = bool_str(flags["client_dummies"])
                    cfg["DUMMIES"]["link_based_dummies"] = bool_str(flags["link_based_dummies"])
                    cfg["DUMMIES"]["multiple_hop_dummies"] = bool_str(flags["multiple_hop_dummies"])

                    write_config(cfg)

                    print(
                        f"Run {rep}/{REPEATS}: "
                        f"scenario={scenario_name}, n_layers={N_LAYERS}, mixes={mixes}, flags={flags}"
                    )
                    run_main()

                    outputs = snapshot_outputs(scenario_name, N_LAYERS, mixes, rep)
                    metrics = compute_run_metrics(outputs)

                    row = {
                        "run_timestamp": run_ts,
                        "repeat_idx": str(rep),
                        "scenario": scenario_name,
                        "n_layers": str(N_LAYERS),
                        "l_mixes_per_layer": str(mixes),
                        "client_dummies": bool_str(flags["client_dummies"]),
                        "link_based_dummies": bool_str(flags["link_based_dummies"]),
                        "multiple_hop_dummies": bool_str(flags["multiple_hop_dummies"]),
                        **metrics,
                        **outputs,
                    }
                    per_run_rows.append(row)

                    key = (scenario_name, mixes)
                    aggregate_bucket.setdefault(key, []).append(float(metrics["mean_entropy"]))

    finally:
        CONFIG_PATH.write_text(original_config_text)

    per_run_path = OUT_DIR / f"summary_per_run_{run_ts}.csv"
    with per_run_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_timestamp",
                "repeat_idx",
                "scenario",
                "n_layers",
                "l_mixes_per_layer",
                "client_dummies",
                "link_based_dummies",
                "multiple_hop_dummies",
                "sent_msgs",
                "received_msgs",
                "dummy_msgs_total",
                "dummy_types_breakdown",
                "mean_entropy",
                "entropy",
                "targets",
                "sent",
                "received",
                "dummy",
            ],
        )
        writer.writeheader()
        writer.writerows(per_run_rows)

    averaged_rows: list[dict[str, str]] = []
    for (scenario, mixes), means in sorted(aggregate_bucket.items(), key=lambda x: (x[0][1], x[0][0])):
        avg_mean_entropy = sum(means) / len(means)
        averaged_rows.append(
            {
                "run_timestamp": run_ts,
                "scenario": scenario,
                "n_layers": str(N_LAYERS),
                "l_mixes_per_layer": str(mixes),
                "repeats": str(len(means)),
                "avg_mean_entropy": f"{avg_mean_entropy:.8f}",
            }
        )

    avg_path = OUT_DIR / f"summary_avg_mean_entropy_{run_ts}.csv"
    with avg_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_timestamp",
                "scenario",
                "n_layers",
                "l_mixes_per_layer",
                "repeats",
                "avg_mean_entropy",
            ],
        )
        writer.writeheader()
        writer.writerows(averaged_rows)

    print(f"Saved per-run summary: {per_run_path}")
    print(f"Saved averaged summary: {avg_path}")


if __name__ == "__main__":
    main()