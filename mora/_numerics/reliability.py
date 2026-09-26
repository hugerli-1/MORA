"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from typing import Any
import json
import numpy as np

EPS = 1e-12


def coefficient_gain(residual: np.ndarray, correction: np.ndarray) -> float:
    residual64 = residual.astype(np.float64, copy=False)
    correction64 = correction.astype(np.float64, copy=False)
    before = float(np.square(residual64).sum())
    after = float(np.square(residual64 - correction64).sum())
    return 100.0 * (before - after) / max(before, EPS)


def coordinate_reliability(
    correction: np.ndarray,
    residual: np.ndarray,
    environment: np.ndarray,
    cap: float,
    dynamic: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    correction64 = correction.astype(np.float64, copy=False)
    residual64 = residual.astype(np.float64, copy=False)

    def optimum(mask: np.ndarray | None = None) -> np.ndarray:
        current_correction = (
            correction64 if mask is None else correction64[mask]
        )
        current_residual = residual64 if mask is None else residual64[mask]
        denominator = np.square(current_correction).sum(axis=(0, 1))
        numerator = (current_correction * current_residual).sum(axis=(0, 1))
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > EPS,
        )

    values = np.unique(environment)
    pooled = optimum()
    fold_optima = np.stack(tuple(optimum(environment == value) for value in values))
    fold_denominator = np.stack(
        tuple(
            np.square(correction64[environment == value]).sum(axis=(0, 1))
            for value in values
        )
    )
    fold_numerator = fold_optima * fold_denominator
    minimum = np.min(fold_optima, axis=0)
    support = minimum > 0.0
    pooled_gain = np.clip(pooled, 0.0, cap) * support
    common_gain = np.minimum(
        np.clip(pooled, 0.0, cap), np.maximum(minimum, 0.0)
    ) * support
    gain_maps = {
        "pooled": pooled_gain.astype(np.float32),
        "common": common_gain.astype(np.float32),
    }
    recent_gain = np.zeros_like(pooled)
    trend_gain = np.zeros_like(pooled)
    if dynamic and len(values) >= 3:
        recency = np.power(0.5, np.arange(len(values) - 1, -1, -1))
        weighted_denominator = (fold_denominator * recency[:, None]).sum(axis=0)
        recent_optimum = np.divide(
            (fold_numerator * recency[:, None]).sum(axis=0),
            weighted_denominator,
            out=np.zeros_like(pooled),
            where=weighted_denominator > EPS,
        )
        recent_support = np.all(fold_optima[-2:] > 0.0, axis=0)
        recent_gain = np.clip(recent_optimum, 0.0, cap) * recent_support

        # Each mode's risk is quadratic in its gain.  Weighted least squares
        # therefore estimates a slowly drifting optimum directly in risk
        # space; a slope penalty prevents a four-environment fit from making
        # an unstable long extrapolation.
        time = np.arange(len(values), dtype=np.float64) - (len(values) - 1)
        design = np.stack((np.ones_like(time), time), axis=1)
        for mode in range(correction.shape[2]):
            weight = fold_denominator[:, mode] * recency
            if np.count_nonzero(weight > EPS) < 3 or not recent_support[mode]:
                continue
            gram = design.T @ (design * weight[:, None])
            gram[1, 1] += 0.5 * float(weight.sum())
            linear = design.T @ (weight * fold_optima[:, mode])
            try:
                coefficient = np.linalg.solve(gram, linear)
            except np.linalg.LinAlgError:
                continue
            extrapolated = float(coefficient[0] + coefficient[1])
            # The recency estimate is a data-dependent trust region.  It
            # permits genuine growth while excluding explosive extrapolation.
            trend_gain[mode] = np.clip(
                extrapolated,
                0.0,
                min(float(cap), 2.0 * max(float(recent_gain[mode]), 0.0)),
            )
        gain_maps["recent"] = recent_gain.astype(np.float32)
        gain_maps["temporal_trend"] = trend_gain.astype(np.float32)
    return (
        gain_maps,
        {
            "active_modes": int(np.count_nonzero(support)),
            "active_mode_fraction": float(np.mean(support)),
            "fold_mode_optima": json.dumps(fold_optima.tolist()),
            "pooled_mode_optima": json.dumps(pooled.tolist()),
            "minimum_mode_optima": json.dumps(minimum.tolist()),
            "recent_mode_gains": json.dumps(recent_gain.tolist()),
            "temporal_trend_mode_gains": json.dumps(trend_gain.tolist()),
        },
    )


def coordinate_gain(
    correction: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    correction64 = correction.astype(np.float64, copy=False)
    residual64 = residual.astype(np.float64, copy=False)
    denominator = np.square(correction64).sum(axis=(0, 1))
    numerator = (correction64 * residual64).sum(axis=(0, 1))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )
