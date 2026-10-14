from .PredictionPerformace import PredictionPerformance
from .PredictionPerformace import TestPredictionPerformance
from .MeasureResources import MeasureResources

EXPERIMENTS_REGISTRY = {
    "PredictionPerformance": PredictionPerformance,
    "TestPredictionPerformance": TestPredictionPerformance,
    "MeasureResources": MeasureResources,
}

__all__ = list(EXPERIMENTS_REGISTRY.keys())
