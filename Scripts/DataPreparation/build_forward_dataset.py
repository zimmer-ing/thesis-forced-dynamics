"""Prepare the Industrial Robot forward dataset with a train-only min-max scaler."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Constants as const
from src.datasets.IndustrialRobot import IndustrialRobotForward

MAT_PATH = const.RAW_DATA_PATH / "IndustrialRobot/Rohdaten/forward_identification_without_raw_data.mat"
OUTPUT_DIR = IndustrialRobotForward.DATA_DIR

TIME_COLUMN = "time_s"
EXCITATION_COLUMNS = [f"motor_torque_{idx}" for idx in range(1, 7)]
RESPONSE_COLUMNS = [f"axis_position_{idx}" for idx in range(1, 7)]
MODEL_COLUMNS = [TIME_COLUMN] + EXCITATION_COLUMNS + RESPONSE_COLUMNS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-path", default=str(MAT_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--val-fraction", type=float, default=IndustrialRobotForward.VAL_FRACTION)
    args = parser.parse_args(argv)

    mat_path = Path(args.mat_path)
    output_dir = Path(args.output_dir)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    source = {
        "time_train": np.asarray(data["time_train"], dtype=float).reshape(-1),
        "time_test": np.asarray(data["time_test"], dtype=float).reshape(-1),
        "u_train": np.asarray(data["u_train"], dtype=float),
        "u_test": np.asarray(data["u_test"], dtype=float),
        "y_train": np.asarray(data["y_train"], dtype=float),
        "y_test": np.asarray(data["y_test"], dtype=float),
    }
    official = {}
    for split in ("train", "test"):
        time = source[f"time_{split}"]
        frame = pd.DataFrame({TIME_COLUMN: time - time[0]})
        for idx, column in enumerate(EXCITATION_COLUMNS):
            frame[column] = source[f"u_{split}"][idx]
        for idx, column in enumerate(RESPONSE_COLUMNS):
            frame[column] = source[f"y_{split}"][idx]
        official[split] = frame[MODEL_COLUMNS]

    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")
    n_val = max(1, int(round(len(official["train"]) * args.val_fraction)))
    n_train = len(official["train"]) - n_val
    if n_train <= 0:
        raise ValueError("Validation split would consume all training samples.")
    raw_files = {
        "train_official_part_scaled.csv": official["train"].iloc[:n_train].copy().reset_index(drop=True),
        "val_from_train_tail_scaled.csv": official["train"].iloc[n_train:].copy().reset_index(drop=True),
        "test_official_scaled.csv": official["test"],
    }
    for filename in ("train_official_part_scaled.csv", "val_from_train_tail_scaled.csv"):
        frame = raw_files[filename]
        frame[TIME_COLUMN] = frame[TIME_COLUMN] - frame[TIME_COLUMN].iloc[0]
    split_files = {
        "train": IndustrialRobotForward.TRAIN_FILES,
        "val": IndustrialRobotForward.VAL_FILES,
        "test": IndustrialRobotForward.TEST_FILES,
    }

    train_for_scaler = pd.concat([raw_files[name] for name in IndustrialRobotForward.TRAIN_FILES], ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(train_for_scaler[MODEL_COLUMNS[1:]])
    scaled_files = {}
    for files in split_files.values():
        for filename in files:
            frame = raw_files[filename]
            values = scaler.transform(frame[MODEL_COLUMNS[1:]])
            scaled = pd.DataFrame(values, columns=MODEL_COLUMNS[1:])
            scaled.insert(0, TIME_COLUMN, frame[TIME_COLUMN].to_numpy(dtype=float))
            scaled_files[filename] = scaled[MODEL_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "scaler_type": "minmax",
        "exclude_columns": [TIME_COLUMN],
        "scaled_columns": MODEL_COLUMNS[1:],
        "scaling_details": {},
    }
    for col, min_value, scale, data_min, data_max, data_range in zip(
        MODEL_COLUMNS[1:], scaler.min_, scaler.scale_,
        scaler.data_min_, scaler.data_max_, scaler.data_range_,
    ):
        details["scaling_details"][col] = {
            "min": float(min_value),
            "scale": float(scale),
            "min_": float(min_value),
            "scale_": float(scale),
            "data_min": float(data_min),
            "data_max": float(data_max),
            "data_range": float(data_range),
        }
    (output_dir / "scaler_params.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    for old_file in output_dir.glob("*.csv"):
        old_file.unlink()
    for filename, frame in scaled_files.items():
        frame.to_csv(output_dir / filename, index=False)

    metadata = {
        "dataset": "IndustrialRobotForward",
        "source_mat": str(mat_path),
        "official_interface": {
            "excitation": "u_train/u_test, measured motor torques in Nm",
            "response": "y_train/y_test, measured real axis positions in deg",
            "time": "time_train/time_test in seconds",
        },
        "sampling": {
            "train_dt_mean_s": float(np.mean(np.diff(source["time_train"]))),
            "test_dt_mean_s": float(np.mean(np.diff(source["time_test"]))),
        },
        "split_policy": (
            "The official training trace is divided into a prefix and a tail. The official "
            "test trace remains a separate file. Dataset class file lists assign the files "
            "to training, validation and test."
        ),
        "val_fraction_from_official_train": float(args.val_fraction),
        "split_files": split_files,
        "rows": {split: sum(len(raw_files[name]) for name in files) for split, files in split_files.items()},
        "time_axis": "time_s resets at zero for each generated split file.",
        "excitation": EXCITATION_COLUMNS,
        "response": RESPONSE_COLUMNS,
    }
    (output_dir / "split_files.json").write_text(json.dumps(split_files, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote IndustrialRobotForward dataset to {output_dir}")
    print(json.dumps(metadata["rows"], indent=2))


if __name__ == "__main__":
    main()
