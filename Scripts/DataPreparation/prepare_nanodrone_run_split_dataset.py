"""Prepare the run-grouped in-distribution Nano-Drone benchmark split."""

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Constants as const
from Scripts.DataPreparation.prepare_nanodrone_dataset import _load_frame
from src.datasets.NonlinearBenchmarks import NanoDroneRunSplit


RAW_ROOT = const.RAW_DATA_PATH / "NonlinearBenchmarks/_repos/nanodrone-sysid-benchmark/data"
OUT_DIR = NanoDroneRunSplit.DATA_DIR

MOTOR_COLUMNS = ["m1_rads", "m2_rads", "m3_rads", "m4_rads"]
RESPONSE_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "rotvec_x", "rotvec_y", "rotvec_z",
    "omega_x", "omega_y", "omega_z",
]
SCALED_COLUMNS = MOTOR_COLUMNS + RESPONSE_COLUMNS

split_files = {
    "train": NanoDroneRunSplit.TRAIN_FILES,
    "val": NanoDroneRunSplit.VAL_FILES,
    "test": NanoDroneRunSplit.TEST_FILES,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.csv"):
        old.unlink()
    for filename in ("split_files.json", "metadata.json", "scaler_params.json"):
        path = OUT_DIR / filename
        if path.exists():
            path.unlink()

    frames = {}
    for files in split_files.values():
        for filename in files:
            name = filename.removesuffix("_scaled.csv")
            trajectory, run_idx = name.split("_run")
            raw_split = "test" if trajectory == "melon" else "train"
            path = RAW_ROOT / raw_split / f"{trajectory}_20251017_run{run_idx}.csv"
            frames[filename] = _load_frame(path, trajectory, int(run_idx))

    train = pd.concat([frames[filename][SCALED_COLUMNS] for filename in split_files["train"]], ignore_index=True)
    details = {}
    for col in SCALED_COLUMNS:
        data_min = float(train[col].min())
        data_max = float(train[col].max())
        data_range = data_max - data_min
        scale = 1.0 / data_range if data_range != 0.0 else 1.0
        min_offset = -data_min * scale
        details[col] = {
            "data_min": data_min,
            "data_max": data_max,
            "data_range": data_range,
            "min": min_offset,
            "scale": scale,
            "min_": min_offset,
            "scale_": scale,
        }

    counts = {}
    for split, files in split_files.items():
        counts[split] = sum(len(frames[filename]) for filename in files)
        for filename in files:
            scaled = frames[filename].copy()
            for col, params in details.items():
                scaled[col] = scaled[col] * params["scale_"] + params["min_"]
            scaled.to_csv(OUT_DIR / filename, index=False)

    metadata = {
        "dataset": "NanoDroneRunSplit",
        "source": "Datasets/NonlinearBenchmarks/_repos/nanodrone-sysid-benchmark",
        "paper": "Nonlinear System Identification for a Nano-drone Benchmark",
        "sampling_time_s": 0.01,
        "task": "50-step open-loop state prediction from motor angular velocities",
        "split": {split: ", ".join(files) for split, files in split_files.items()},
        "channels": {
            "time": ["time_s"],
            "excitation": MOTOR_COLUMNS,
            "response": RESPONSE_COLUMNS,
            "auxiliary": [
                "quat_x_raw",
                "quat_y_raw",
                "quat_z_raw",
                "quat_w_raw",
                "acc_body_x",
                "acc_body_y",
                "acc_body_z",
                "trajectory_code",
                "run_idx",
            ],
        },
        "notes": [
            "Splits are grouped by complete CSV runs, not by random windows or temporal chopping.",
            "Min-max scaling is fitted on the training runs only.",
        ],
    }

    (OUT_DIR / "split_files.json").write_text(json.dumps(split_files, indent=2, sort_keys=True), encoding="utf-8")
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    scaler = {
        "scaler_type": "minmax",
        "exclude_columns": [
            "time_s",
            "quat_x_raw",
            "quat_y_raw",
            "quat_z_raw",
            "quat_w_raw",
            "acc_body_x",
            "acc_body_y",
            "acc_body_z",
            "trajectory_code",
            "run_idx",
        ],
        "scaled_columns": SCALED_COLUMNS,
        "scaling_details": details,
    }
    (OUT_DIR / "scaler_params.json").write_text(json.dumps(scaler, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Prepared NanoDroneRunSplit under {OUT_DIR}")
    print(json.dumps({"rows": counts, "files": {k: len(v) for k, v in split_files.items()}}, indent=2))


if __name__ == "__main__":
    main()
