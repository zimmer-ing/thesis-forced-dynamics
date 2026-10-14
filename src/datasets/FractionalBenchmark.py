from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import Constants as CONST
from src.datasets.dataset_base import MultiFileTimeSeriesDataset


class FractionalBenchmarkBase(MultiFileTimeSeriesDataset):
    """
    Base dataset for benchmark trajectories generated from MATLAB/FOMCON-style NFDE experiments.

    Expected folder layout:
        data/Fractional_Data/fractional_benchmark_data/<task>/{train,val,test}/*.csv

    Expected CSV columns:
        - t : time
        - u : forcing / excitation
        - y : response
    """

    TASK_SUBDIR: str = ""
    DEFAULT_WINDOW_TASK: str = "forecasting"

    def __init__(self, config: dict, data_split: str = "train", device: Optional[str] = "cpu"):
        if data_split not in {"train", "val", "test"}:
            raise ValueError("data_split must be 'train' | 'val' | 'test'")

        cfg = dict(config)
        cfg.setdefault("task", self.DEFAULT_WINDOW_TASK)

        if "data_root" in cfg:
            root = Path(cfg["data_root"])
        else:
            task_subdir = cfg.get("task_data_subdir", self.TASK_SUBDIR)
            root = Path(CONST.DATA_PATH, "Fractional_Data", "fractional_benchmark_data", task_subdir)

        split_dir = root / data_split
        files = sorted(split_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found for split '{data_split}' in {split_dir}")

        super().__init__(
            directory=root,
            files=files,
            config=cfg,
            device=device,
            data_split=data_split,
        )

    def load_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def get_channels() -> Dict[str, list | str]:
        return {
            "time": "t",
            "excitation": ["u"],
            "response": ["y"],
            "auxiliary": []
        }


class FractionalBenchmarkReconstruction(FractionalBenchmarkBase):
    TASK_SUBDIR = "reconstruction"
    DEFAULT_WINDOW_TASK = "reconstruction"


class FractionalBenchmarkExtrapolation(FractionalBenchmarkBase):
    TASK_SUBDIR = "extrapolation"
    DEFAULT_WINDOW_TASK = "forecasting"


class FractionalBenchmarkForcing(FractionalBenchmarkBase):
    TASK_SUBDIR = "forcing"
    DEFAULT_WINDOW_TASK = "forecasting"
