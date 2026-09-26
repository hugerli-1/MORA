"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .history import time_features
from .shared_ridge import fit_global
from .shared_ridge import predict_global
from typing import Any
import json
import numpy as np

def fit_periodic_baseline(
    values: np.ndarray, fit_mask: np.ndarray, period: int
) -> np.ndarray:
    positions = np.flatnonzero(fit_mask)
    phase = positions % int(period)
    count = np.bincount(phase, minlength=period).astype(np.float64)
    total = np.zeros((period, values.shape[1]), dtype=np.float64)
    np.add.at(total, phase, values[positions].astype(np.float64))
    if np.any(count == 0):
        raise RuntimeError("empty train-only phase bin")
    return (total / count[:, None]).astype(np.float32)


def factor_history_features(
    history: np.ndarray, index: np.ndarray, period: int
) -> np.ndarray:
    """Compact joint state used to predict all latent factors together."""

    current = np.asarray(history, dtype=np.float32)
    last = current[:, -1]
    pieces = [last]
    for width in (6, 12, 24, 48, 96):
        realized = min(int(width), current.shape[1])
        pieces.append(current[:, -realized:].mean(axis=1))
    for lag in (6, 12, 24):
        realized = min(int(lag), current.shape[1] - 1)
        pieces.append(last - current[:, -1 - realized])
    recent = current[:, -min(24, current.shape[1]) :]
    axis = np.linspace(-1.0, 1.0, recent.shape[1], dtype=np.float32)
    denominator = max(float(np.square(axis).sum()), 1e-8)
    pieces.append(
        np.einsum("t,ntr->nr", axis, recent, optimize=True) / denominator
    )
    flattened = [value.reshape(len(current), -1) for value in pieces]
    calendar = time_features(index, period, current.shape[1], 1)[:, 0]
    return np.concatenate((*flattened, calendar), axis=1).astype(np.float32)


def factor_future_moments(
    future: np.ndarray, aggregate: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "nhr,qh->nrq",
        np.asarray(future, dtype=np.float64),
        np.asarray(aggregate, dtype=np.float64),
        optimize=True,
    ).astype(np.float32)


def select_factor_predictor(
    train: dict[str, np.ndarray],
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    aggregate: np.ndarray,
    period: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features = factor_history_features(train["history"], train["index"], period)
    first_features = factor_history_features(
        first["history"], first["index"], period
    )
    second_features = factor_history_features(
        second["history"], second["index"], period
    )
    absolute = factor_future_moments(train["true"], aggregate)
    delta = absolute - train["history"][:, -1, :, None]
    joint_target = np.concatenate((absolute, delta), axis=2)
    blocks = absolute.shape[2]
    candidate_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, float, str]] = []

    for alpha in alphas:
        predictions: dict[str, list[np.ndarray]] = {
            "absolute": [],
            "last_delta": [],
            "balanced": [],
        }
        truths: list[np.ndarray] = []
        fold_errors: dict[str, list[float]] = {
            name: [] for name in predictions
        }
        for fit, held in folds:
            model = fit_global(features[fit], joint_target[fit], float(alpha))
            current = predict_global(
                model,
                features[held],
                absolute.shape[1],
                2 * blocks,
            )
            direct = current[:, :, :blocks]
            last_delta = (
                current[:, :, blocks:]
                + train["history"][held, -1, :, None]
            )
            modes = {
                "absolute": direct,
                "last_delta": last_delta,
                "balanced": 0.5 * (direct + last_delta),
            }
            truths.append(absolute[held])
            for name, prediction in modes.items():
                predictions[name].append(prediction)
                fold_errors[name].append(
                    float(
                        np.square(
                            prediction.astype(np.float64)
                            - absolute[held].astype(np.float64)
                        ).mean()
                    )
                )
        truth = np.concatenate(truths)
        for name, values in predictions.items():
            prediction = np.concatenate(values)
            error = float(
                np.square(
                    prediction.astype(np.float64) - truth.astype(np.float64)
                ).mean()
            )
            candidates.append((error, float(alpha), name))
            candidate_rows.append(
                {
                    "branch": "global_factor_state",
                    "mode": name,
                    "alpha": float(alpha),
                    "aggregate_oof_moment_mse": error,
                    "fold_mse": json.dumps(fold_errors[name]),
                    "feature_dimension": int(features.shape[1]),
                }
            )

    _, selected_alpha, selected_mode = min(candidates)
    model = fit_global(features, joint_target, selected_alpha)

    def predict(current: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
        joint = predict_global(
            model, values, absolute.shape[1], 2 * blocks
        )
        direct = joint[:, :, :blocks]
        last_delta = joint[:, :, blocks:] + current["history"][:, -1, :, None]
        if selected_mode == "absolute":
            return direct
        if selected_mode == "last_delta":
            return last_delta
        return 0.5 * (direct + last_delta)

    candidate_rows.append(
        {
            "branch": "global_factor_state_selected",
            "mode": selected_mode,
            "alpha": selected_alpha,
            "feature_dimension": int(features.shape[1]),
        }
    )
    return (
        predict(first, first_features).astype(np.float32),
        predict(second, second_features).astype(np.float32),
        candidate_rows,
    )
