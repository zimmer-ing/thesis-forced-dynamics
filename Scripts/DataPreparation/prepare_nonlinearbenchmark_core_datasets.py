"""Prepare CascadedTanks; fit min-max scaling on the training split only."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Constants as const
from src.datasets.NonlinearBenchmarks import CascadedTanks

mat_path = const.RAW_DATA_PATH / "NonlinearBenchmarks/_cache/Cascaded_Tanks/CascadedTanksFiles/dataBenchmark.mat"
output_dir = CascadedTanks.DATA_DIR
scaled_columns = ["pump_voltage", "lower_tank_level"]
exclude_columns = ["time_s", "source_id"]


def main():
    mat = loadmat(mat_path, squeeze_me=True)
    ts = float(mat["Ts"])

    u = np.asarray(mat["uEst"], dtype=float).reshape(-1)
    y = np.asarray(mat["yEst"], dtype=float).reshape(-1)
    estimation = pd.DataFrame({
        "time_s": np.arange(u.size, dtype=float) * ts,
        "pump_voltage": u,
        "lower_tank_level": y,
        "source_id": float(0),
    })
    split_idx = int(len(estimation) * CascadedTanks.TRAIN_FRACTION)
    train = estimation.iloc[:split_idx].copy()
    val = estimation.iloc[split_idx:].copy()
    val["time_s"] = val["time_s"] - float(val["time_s"].iloc[0])

    u = np.asarray(mat["uVal"], dtype=float).reshape(-1)
    y = np.asarray(mat["yVal"], dtype=float).reshape(-1)
    test = pd.DataFrame({
        "time_s": np.arange(u.size, dtype=float) * ts,
        "pump_voltage": u,
        "lower_tank_level": y,
        "source_id": float(1),
    })
    raw_files = {
        "train_estimation_scaled.csv": train,
        "val_estimation_tail_scaled.csv": val,
        "test_official_validation_scaled.csv": test,
    }
    split_files = {
        "train": CascadedTanks.TRAIN_FILES,
        "val": CascadedTanks.VAL_FILES,
        "test": CascadedTanks.TEST_FILES,
    }

    train_for_scaler = pd.concat([raw_files[name] for name in CascadedTanks.TRAIN_FILES], ignore_index=True)
    scaling_details = {}
    for column in scaled_columns:
        data_min = float(train_for_scaler[column].min())
        data_max = float(train_for_scaler[column].max())
        data_range = data_max - data_min
        scale = 1.0 / data_range if data_range != 0.0 else 1.0
        min_offset = -data_min * scale
        scaling_details[column] = {
            "data_min": data_min,
            "data_max": data_max,
            "data_range": data_range,
            "min": min_offset,
            "scale": scale,
            "min_": min_offset,
            "scale_": scale,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.csv"):
        old.unlink()
    for filename in ["split_files.json", "metadata.json", "scaler_params.json"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    for files in split_files.values():
        for filename in files:
            scaled = raw_files[filename].copy()
            for column, params in scaling_details.items():
                scaled[column] = scaled[column] * params["scale_"] + params["min_"]
            scaled.to_csv(output_dir / filename, index=False)

    metadata = {
        "dataset": "CascadedTanks",
        "source": "Datasets/NonlinearBenchmarks/_cache/Cascaded_Tanks/CascadedTanksFiles",
        "sampling_time_s": ts,
        "channels": {
            "time": ["time_s"],
            "excitation": ["pump_voltage"],
            "response": ["lower_tank_level"],
            "auxiliary": ["source_id"],
        },
        "notes": [
            f"The official estimation sequence is divided after the first {CascadedTanks.TRAIN_FRACTION * 100:g} percent.",
            "The official validation sequence remains a separate file; dataset class file lists assign files to each split.",
            "Min-max scaling is fitted only on the files selected in TRAIN_FILES.",
        ],
    }
    scaler = {
        "scaler_type": "minmax",
        "exclude_columns": exclude_columns,
        "scaled_columns": scaled_columns,
        "scaling_details": scaling_details,
    }
    (output_dir / "split_files.json").write_text(json.dumps(split_files, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "scaler_params.json").write_text(json.dumps(scaler, indent=2, sort_keys=True), encoding="utf-8")
    print("Prepared CascadedTanks.")


if __name__ == "__main__":
    main()
