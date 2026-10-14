from pathlib import Path
from typing import Dict, Optional

import Constants as CONST
import pandas as pd

from src.datasets.dataset_base import MultiFileTimeSeriesDataset


class IndustrialRobotForward(MultiFileTimeSeriesDataset):
    """Prepared KUKA KR300 forward-identification benchmark dataset.

    The official prepared file contains measured motor torques as inputs and
    measured real-axis positions as outputs. The generated CSV files are scaled
    and keep a strict 10 Hz time grid.
    """

    DATA_DIR = Path(CONST.DATA_PATH, "IndustrialRobot", "forward_minmax")
    VAL_FRACTION = 0.10
    TRAIN_FILES = ["train_official_part_scaled.csv"]
    VAL_FILES = ["val_from_train_tail_scaled.csv"]
    TEST_FILES = ["test_official_scaled.csv"]

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

    @staticmethod
    def get_channels() -> Dict[str, list[str]]:
        return {
            "time": ["time_s"],
            "excitation": [
                "motor_torque_1",
                "motor_torque_2",
                "motor_torque_3",
                "motor_torque_4",
                "motor_torque_5",
                "motor_torque_6",
            ],
            "response": [
                "axis_position_1",
                "axis_position_2",
                "axis_position_3",
                "axis_position_4",
                "axis_position_5",
                "axis_position_6",
            ],
            "auxiliary": [],
        }
