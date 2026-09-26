"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .coarse import coarse_features
from .coarse import decode_target
from .coarse import normalization_state
from .coarse import segment_means
from .history import RidgeMap
from .history import predict_ridge_map
from .periodic import select_template_periods
from dataclasses import dataclass
import numpy as np

def _harmonic_design(time: np.ndarray, periods: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=np.float64)
    columns = [np.ones_like(time, dtype=np.float64)]
    for period in np.asarray(periods, dtype=np.float64):
        angle = 2.0 * np.pi * time / float(period)
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.stack(columns, axis=-1)


@dataclass(frozen=True)
class PeriodicTemplateEstimator:
    """Train-only multi-period atlas with X-only local level registration."""

    periods: np.ndarray
    coefficient: np.ndarray
    half_life: float = 16.0

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        train_end: int,
        daily_period: int,
        period_count: int = 8,
        known_harmonics: int = 1,
        half_life: float = 16.0,
        atlas_ridge: float = 1e-3,
    ) -> "SpectralAtlasLevelObserver":
        values = np.asarray(values, dtype=np.float32)
        periods = select_template_periods(
            values,
            int(train_end),
            int(daily_period),
            int(period_count),
            known_harmonics=int(known_harmonics),
        )
        design = _harmonic_design(
            np.arange(int(train_end), dtype=np.float64), periods
        )
        gram = design.T @ design
        penalty = np.eye(gram.shape[0], dtype=np.float64) * (
            float(atlas_ridge) * max(float(train_end), 1.0)
        )
        penalty[0, 0] = 1e-8
        coefficient = np.linalg.solve(
            gram + penalty,
            design.T @ values[: int(train_end)].astype(np.float64),
        )
        return cls(
            periods=np.asarray(periods, dtype=np.float64),
            coefficient=np.asarray(coefficient, dtype=np.float64),
            half_life=float(half_life),
        )

    @property
    def channels(self) -> int:
        return int(self.coefficient.shape[1])

    def predict(
        self,
        history: np.ndarray,
        index: np.ndarray,
        horizon: int,
        aggregate: np.ndarray,
    ) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        index = np.asarray(index, dtype=np.int64)
        if history.ndim != 3 or history.shape[0] != len(index):
            raise ValueError((history.shape, index.shape))
        if history.shape[2] != self.channels:
            raise ValueError((history.shape[2], self.channels))

        history_length = int(history.shape[1])
        history_positions = (
            index[:, None]
            + np.arange(history_length, dtype=np.int64)[None]
        )
        atlas_history = (
            _harmonic_design(history_positions, self.periods)
            @ self.coefficient
        )
        recency = np.exp(
            -np.log(2.0)
            * (history_length - 1 - np.arange(history_length, dtype=np.float64))
            / max(float(self.half_life), 1.0)
        )
        recency /= recency.sum()
        anomaly_level = np.einsum(
            "l,nlc->nc",
            recency,
            history.astype(np.float64) - atlas_history,
            optimize=True,
        )

        future_positions = (
            index[:, None]
            + history_length
            + np.arange(int(horizon), dtype=np.int64)[None]
        )
        atlas_future = (
            _harmonic_design(future_positions, self.periods)
            @ self.coefficient
        ).astype(np.float32)
        atlas_moment = segment_means(atlas_future, aggregate).astype(np.float64)
        return (atlas_moment + anomaly_level[:, :, None]).astype(np.float32)


@dataclass(frozen=True)
class ChannelWisePrediction:
    period: int
    feature_set: str
    absolute: RidgeMap
    centered: RidgeMap

    def predict(self, history: np.ndarray, index: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        index = np.asarray(index, dtype=np.int64)
        outputs: list[np.ndarray] = []
        for mode, model in (
            ("absolute", self.absolute),
            ("last_centered", self.centered),
        ):
            normalized, location, scale = normalization_state(history, mode)
            features = coarse_features(
                normalized, index, self.period, self.feature_set
            )
            outputs.append(
                decode_target(
                    predict_ridge_map(model, features), location, scale
                )
            )
        return (0.5 * (outputs[0] + outputs[1])).astype(np.float32)
