"""MORA: Correcting Time Series Forecasts with Coarse Future Signals."""

from .config import MORAConfig
from .data import ForecastBatch, OOFForecasts
from .coarse_future import CoarseFutureEstimation, ChannelWiseCoarseEstimator
from .factorized_estimator import FactorizedCoarseEstimator
from .stagewise import StagewiseResidualAdaptation
from .model import MORA

__version__ = "0.1.0"
__all__ = ["MORA", "MORAConfig", "ForecastBatch", "OOFForecasts",
           "CoarseFutureEstimation", "ChannelWiseCoarseEstimator",
           "FactorizedCoarseEstimator", "StagewiseResidualAdaptation"]
