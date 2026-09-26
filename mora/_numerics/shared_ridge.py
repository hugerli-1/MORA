"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class GlobalRidge:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weight: np.ndarray


def fit_global(features: np.ndarray, target: np.ndarray, alpha: float) -> GlobalRidge:
    feature_mean = features.mean(axis=0, dtype=np.float64)
    feature_scale = np.maximum(features.std(axis=0, dtype=np.float64), 1e-5)
    flat_target = target.reshape(len(target), -1)
    target_mean = flat_target.mean(axis=0, dtype=np.float64)
    target_scale = np.maximum(flat_target.std(axis=0, dtype=np.float64), 1e-4)
    x = (features.astype(np.float64) - feature_mean) / feature_scale
    y = (flat_target.astype(np.float64) - target_mean) / target_scale
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    weight = np.linalg.solve(gram, x.T @ y)
    return GlobalRidge(
        feature_mean.astype(np.float32),
        feature_scale.astype(np.float32),
        target_mean.astype(np.float32),
        target_scale.astype(np.float32),
        weight.astype(np.float32),
    )


def predict_global(
    model: GlobalRidge, features: np.ndarray, channels: int, blocks: int
) -> np.ndarray:
    x = (
        features.astype(np.float64) - model.feature_mean.astype(np.float64)
    ) / model.feature_scale.astype(np.float64)
    output = x @ model.weight.astype(np.float64)
    output = output * model.target_scale.astype(np.float64) + model.target_mean
    return output.reshape(len(features), channels, blocks).astype(np.float32)
