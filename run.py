#!/usr/bin/env python3
from __future__ import annotations

import configparser
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ConfigFile.ini"
LOGS_DIR = ROOT / "Logs"
OUT_DIR = ROOT / "files"

N_LAYERS = 3
MIXES_PER_LAYER = [3, 4, 5, 6, 7, 8, 9, 10]
STOP_REAL_MSGS_PERCENTS = [5, 10, 15, 20]
MSG_DELIVERY_PERCENTS = [85, 90, 95, 99]

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

def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)

def compute_run_metrics(sent_path: Path, recv_path: Path, dummy_path: Path) -> dict[str, str]:
    n_sent = count_csv_rows(sent_path)
    n_received = count_csv_rows(recv_path)
    n_dummy = count_csv_rows(dummy_path)
    recv_pct = (100.0 * n_received / n_sent) if n_sent > 0 else 0.0
    return {
        "n_sent_msgs": str(n_sent),
        "n_received_msgs": str(n_received),
        "n_dummy_msgs": str(n_dummy),
        "received_vs_sent_percent": f"{recv_pct:.2f}",
    }

def run_main() -> None:
    # Uses current python interpreter (works inside your venv too).
    subprocess.run([sys.executable, "main.py"], cwd=ROOT, check=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)

def snapshot_outputs(
    scenario: str, n_layers: int, mixes: int, stop_pct: int, msg_delivery_pct: int
) -> dict[str, str]:
    prefix = f"{scenario}_L{n_layers}_M{mixes}_Stop{stop_pct}_Deliv{msg_delivery_pct}"

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
    
    metrics = compute_run_metrics(sent_dst, recv_dst, dummy_dst)

    return {
        **metrics,
        "entropy": str(entropy_dst) if entropy_dst.exists() else "",
        "targets": str(targets_dst) if targets_dst.exists() else "",
        "sent": str(sent_dst) if sent_dst.exists() else "",
        "received": str(recv_dst) if recv_dst.exists() else "",
        "dummy": str(dummy_dst) if dummy_dst.exists() else "",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Backup config so it gets restored even if a run fails.
    original_config_text = CONFIG_PATH.read_text()

    manifest_rows: list[dict[str, str]] = []

    try:
        for mixes in MIXES_PER_LAYER:
            for scenario_name, flags in SCENARIOS.items():
                for stop_pct in STOP_REAL_MSGS_PERCENTS:
                    for delivery_pct in MSG_DELIVERY_PERCENTS:
                        cfg = load_config()

                        cfg["TOPOLOGY"]["n_layers"] = str(N_LAYERS)
                        cfg["TOPOLOGY"]["l_mixes_per_layer"] = str(mixes)

                        cfg["DUMMIES"]["client_dummies"] = bool_str(flags["client_dummies"])
                        cfg["DUMMIES"]["link_based_dummies"] = bool_str(flags["link_based_dummies"])
                        cfg["DUMMIES"]["multiple_hop_dummies"] = bool_str(flags["multiple_hop_dummies"])

                        cfg["MIXING"]["stop_real_msgs_percent"] = str(stop_pct)
                        cfg["MIXING"]["msg_delivery_percent"] = str(delivery_pct)

                        write_config(cfg)

                        print(
                            f"Running scenario={scenario_name}, "
                            f"n_layers={N_LAYERS}, l_mixes_per_layer={mixes}, "
                            f"stop_real_msgs_percent={stop_pct}, "
                            f"msg_delivery_percent={delivery_pct}, "
                            f"flags={flags}"
                        )
                        run_main()

                        outputs = snapshot_outputs(
                            scenario_name, N_LAYERS, mixes, stop_pct, delivery_pct
                        )

                        row = {
                            "scenario": scenario_name,
                            "n_layers": str(N_LAYERS),
                            "l_mixes_per_layer": str(mixes),
                            "stop_real_msgs_percent": str(stop_pct),
                            "msg_delivery_percent": str(delivery_pct),
                            "client_dummies": bool_str(flags["client_dummies"]),
                            "link_based_dummies": bool_str(flags["link_based_dummies"]),
                            "multiple_hop_dummies": bool_str(flags["multiple_hop_dummies"]),
                            **outputs,
                        }
                        manifest_rows.append(row)

    finally:
        # Always restore original config file
        CONFIG_PATH.write_text(original_config_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = OUT_DIR / f"summary_{timestamp}.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "n_layers",
                "l_mixes_per_layer",
                "stop_real_msgs_percent",
                "msg_delivery_percent",
                "received_vs_sent_percent",
                "client_dummies",
                "link_based_dummies",
                "multiple_hop_dummies",
                "n_sent_msgs",
                "n_received_msgs",
                "n_dummy_msgs",
                "entropy",
                "targets",
                "sent",
                "received",
                "dummy",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Done. Saved outputs and manifest to: {OUT_DIR}")


if __name__ == "__main__":
    main()