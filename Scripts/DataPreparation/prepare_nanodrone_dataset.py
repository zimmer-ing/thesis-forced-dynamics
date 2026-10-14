"""Prepare NanoDroneOOD: train on random/square/chirp, test on melon."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Constants as const
from src.datasets.NonlinearBenchmarks import NanoDroneOOD

RAW_ROOT = const.RAW_DATA_PATH / "NonlinearBenchmarks/_repos/nanodrone-sysid-benchmark/data"
OUT_DIR = NanoDroneOOD.DATA_DIR

MOTOR_COLUMNS = ["m1_rads", "m2_rads", "m3_rads", "m4_rads"]
RESPONSE_COLUMNS = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "rotvec_x", "rotvec_y", "rotvec_z",
    "omega_x", "omega_y", "omega_z",
]
SCALED_COLUMNS = MOTOR_COLUMNS + RESPONSE_COLUMNS

split_files = {
    "train": NanoDroneOOD.TRAIN_FILES,
    "val": NanoDroneOOD.VAL_FILES,
    "test": NanoDroneOOD.TEST_FILES,
}
TRAJECTORY_CODES = {"random": 0, "square": 1, "chirp": 2, "melon": 3}


def _quat_xyzw_to_rotvec(quat):
    """Convert scalar-last unit quaternions to shortest SO(3) rotation vectors."""
    # Busetto et al., Sec. 6.1, Eq. (13): https://arxiv.org/html/2512.14450v1#S6.SS1
    q = np.asarray(quat, dtype=float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    q = q / norms

    # q and -q describe the same rotation; choose the shorter representation.
    negative_w = q[:, 3] < 0.0
    q[negative_w] *= -1.0

    vector = q[:, :3]
    scalar = np.clip(q[:, 3], -1.0, 1.0)
    vector_norm = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(vector_norm, scalar)

    rotvec = np.zeros((q.shape[0], 3), dtype=float)
    regular = vector_norm > 1e-12
    rotvec[regular] = vector[regular] / vector_norm[regular, None] * angle[regular, None]
    rotvec[~regular] = 2.0 * vector[~regular]
    return rotvec


def _load_frame(path, trajectory, run_idx):
    raw = pd.read_csv(path)
    quat = raw[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    rotvec = _quat_xyzw_to_rotvec(quat)
    time_s = pd.to_numeric(raw["t"], errors="coerce").to_numpy(dtype=float)
    time_s = time_s - float(time_s[0])

    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "m1_rads": pd.to_numeric(raw["m1_rads"], errors="coerce"),
            "m2_rads": pd.to_numeric(raw["m2_rads"], errors="coerce"),
            "m3_rads": pd.to_numeric(raw["m3_rads"], errors="coerce"),
            "m4_rads": pd.to_numeric(raw["m4_rads"], errors="coerce"),
            "pos_x": pd.to_numeric(raw["x"], errors="coerce"),
            "pos_y": pd.to_numeric(raw["y"], errors="coerce"),
            "pos_z": pd.to_numeric(raw["z"], errors="coerce"),
            "vel_x": pd.to_numeric(raw["vx"], errors="coerce"),
            "vel_y": pd.to_numeric(raw["vy"], errors="coerce"),
            "vel_z": pd.to_numeric(raw["vz"], errors="coerce"),
            "rotvec_x": rotvec[:, 0],
            "rotvec_y": rotvec[:, 1],
            "rotvec_z": rotvec[:, 2],
            "omega_x": pd.to_numeric(raw["wx"], errors="coerce"),
            "omega_y": pd.to_numeric(raw["wy"], errors="coerce"),
            "omega_z": pd.to_numeric(raw["wz"], errors="coerce"),
            "quat_x_raw": pd.to_numeric(raw["qx"], errors="coerce"),
            "quat_y_raw": pd.to_numeric(raw["qy"], errors="coerce"),
            "quat_z_raw": pd.to_numeric(raw["qz"], errors="coerce"),
            "quat_w_raw": pd.to_numeric(raw["qw"], errors="coerce"),
            "acc_body_x": pd.to_numeric(raw["ax_body"], errors="coerce"),
            "acc_body_y": pd.to_numeric(raw["ay_body"], errors="coerce"),
            "acc_body_z": pd.to_numeric(raw["az_body"], errors="coerce"),
            "trajectory_code": float(TRAJECTORY_CODES[trajectory]),
            "run_idx": float(run_idx),
        }
    )
    if frame.isna().any().any():
        raise ValueError(f"NaN values while preparing {path}")
    return frame


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
        "dataset": "NanoDroneOOD",
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
            "Responses follow the paper's black-box representation: position, linear velocity, SO(3) log rotation vector, and angular velocity.",
            "Raw quaternions and body accelerations are retained only as auxiliary metadata.",
            "Min-max scaling is fitted on the training trajectories only.",
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

    print(f"Prepared NanoDroneOOD under {OUT_DIR}")
    print(json.dumps({"rows": counts, "files": {k: len(v) for k, v in split_files.items()}}, indent=2))


if __name__ == "__main__":
    main()
