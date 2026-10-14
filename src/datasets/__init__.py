from .SMD_Dataset import SMDSystem
from .IndPenSim import IndPenSimMinMax
from .RLC_Datasets import Order_4_Dataset as RLC_Dataset
from .IndustrialRobot import IndustrialRobotForward
from .NonlinearBenchmarks import (
    CascadedTanks,
    NanoDroneOOD,
    NanoDroneRunSplit,
)
from .FractionalBenchmark import (
    FractionalBenchmarkReconstruction,
    FractionalBenchmarkExtrapolation,
    FractionalBenchmarkForcing,
)

DATASET_REGISTRY = {
    "SMDSystem": SMDSystem,
    "IndPenSim": IndPenSimMinMax,
    "RLCDataset": RLC_Dataset,
    "IndustrialRobotForward": IndustrialRobotForward,
    "CascadedTanks": CascadedTanks,
    "NanoDroneOOD": NanoDroneOOD,
    "NanoDroneRunSplit": NanoDroneRunSplit,
    "FractionalBenchmarkReconstruction": FractionalBenchmarkReconstruction,
    "FractionalBenchmarkExtrapolation": FractionalBenchmarkExtrapolation,
    "FractionalBenchmarkForcing": FractionalBenchmarkForcing,
}

__all__ = [
    "DATASET_REGISTRY",
    "SMDSystem",
    "IndPenSimMinMax",
    "RLC_Dataset",
    "IndustrialRobotForward",
    "CascadedTanks",
    "NanoDroneOOD",
    "NanoDroneRunSplit",
    "FractionalBenchmarkReconstruction",
    "FractionalBenchmarkExtrapolation",
    "FractionalBenchmarkForcing",
]
