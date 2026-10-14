from pathlib import Path
from typing import Dict, Optional

import Constants as CONST
import pandas as pd

from src.datasets.dataset_base import MultiFileTimeSeriesDataset


class _PreparedNonlinearBenchmark(MultiFileTimeSeriesDataset):
    DATA_DIR: Path

    def __init__(self, config: dict, data_split: str = "train", device: Optional[str] = "cpu"):
        if data_split == "train":
            files = self.TRAIN_FILES
        elif data_split == "val":
            files = self.VAL_FILES
        elif data_split == "test":
            files = self.TEST_FILES
        else:
            raise ValueError(f"Invalid data split: {data_split}. Must be one of ['train', 'val', 'test']")
        super().__init__(
            directory=self.DATA_DIR,
            files=files,
            config=config,
            device=device,
            data_split=data_split,
        )

    def load_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)


class CascadedTanks(_PreparedNonlinearBenchmark):
    """Prepared Cascaded Tanks benchmark dataset.

    The excitation is the pump voltage and the response is the measured lower
    tank water level. The official estimation sequence is split into train and
    validation parts; the official validation sequence is used as test data.
    The configuration uses 10 history samples and 150 forecast samples.
    """

    DATA_DIR = Path(CONST.DATA_PATH, "NonlinearBenchmarks", "CascadedTanks", "minmax")
    TRAIN_FRACTION = 0.8
    TRAIN_FILES = ["train_estimation_scaled.csv"]
    VAL_FILES = ["val_estimation_tail_scaled.csv"]
    TEST_FILES = ["test_official_validation_scaled.csv"]

    @staticmethod
    def get_channels() -> Dict[str, list[str]]:
        return {
            "time": ["time_s"],
            "excitation": ["pump_voltage"],
            "response": ["lower_tank_level"],
            "auxiliary": ["source_id"],
        }


class NanoDroneOOD(_PreparedNonlinearBenchmark):
    """Prepared Nano-Drone system-identification benchmark dataset.

    Inputs are the four measured motor angular velocities. Responses follow
    the benchmark paper's learning representation: position, linear velocity,
    SO(3) rotation vector, and angular velocity. The split keeps the Melon
    trajectory family as a held-out out-of-distribution test set.
    """

    DATA_DIR = Path(CONST.DATA_PATH, "NonlinearBenchmarks", "NanoDroneOOD", "minmax")
    TRAIN_FILES = [
        "random_run1_scaled.csv",
        "random_run2_scaled.csv",
        "random_run3_scaled.csv",
        "square_run1_scaled.csv",
        "square_run2_scaled.csv",
        "square_run3_scaled.csv",
        "chirp_run1_scaled.csv",
        "chirp_run2_scaled.csv",
        "chirp_run3_scaled.csv",
    ]
    VAL_FILES = [
        "random_run4_scaled.csv",
        "square_run4_scaled.csv",
        "chirp_run4_scaled.csv",
    ]
    TEST_FILES = [
        "melon_run1_scaled.csv",
        "melon_run2_scaled.csv",
        "melon_run3_scaled.csv",
    ]

    @staticmethod
    def get_channels() -> Dict[str, list[str]]:
        return {
            "time": ["time_s"],
            "excitation": ["m1_rads", "m2_rads", "m3_rads", "m4_rads"],
            "response": [
                "pos_x",
                "pos_y",
                "pos_z",
                "vel_x",
                "vel_y",
                "vel_z",
                "rotvec_x",
                "rotvec_y",
                "rotvec_z",
                "omega_x",
                "omega_y",
                "omega_z",
            ],
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
        }


class NanoDroneRunSplit(NanoDroneOOD):
    """Nano-Drone benchmark with all trajectory families in all splits.

    This split keeps whole CSV runs grouped and disjoint, but it does not hold
    out an entire trajectory family. It is useful to distinguish ordinary
    in-distribution evaluation from the NanoDroneOOD unseen-family split.
    """

    DATA_DIR = Path(CONST.DATA_PATH, "NonlinearBenchmarks", "NanoDroneRunSplit", "minmax")
    TRAIN_FILES = [
        "random_run1_scaled.csv",
        "random_run2_scaled.csv",
        "square_run1_scaled.csv",
        "square_run2_scaled.csv",
        "chirp_run1_scaled.csv",
        "chirp_run2_scaled.csv",
        "melon_run1_scaled.csv",
    ]
    VAL_FILES = [
        "random_run3_scaled.csv",
        "square_run3_scaled.csv",
        "chirp_run3_scaled.csv",
        "melon_run2_scaled.csv",
    ]
    TEST_FILES = [
        "random_run4_scaled.csv",
        "square_run4_scaled.csv",
        "chirp_run4_scaled.csv",
        "melon_run3_scaled.csv",
    ]
