"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from dataclasses import dataclass
from scipy.fft import dct
from typing import List
from typing import Tuple
import numpy as np

@dataclass
class RidgeMap:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights: List[np.ndarray]
    cap: float | None


def time_features(index: np.ndarray, period: int, history: int, channels: int) -> np.ndarray:
    future_start = np.asarray(index, dtype=np.float64) + int(history)
    values = []
    for current, harmonic in ((period, 1), (period, 2), (7 * period, 1)):
        angle = 2.0 * np.pi * harmonic * future_start / float(current)
        values.extend((np.sin(angle), np.cos(angle)))
    output = np.stack(values, axis=1).astype(np.float32)
    return np.repeat(output[:, None, :], channels, axis=1)


def slope(values: np.ndarray) -> np.ndarray:
    length = values.shape[1]
    axis = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    denominator = max(float(np.square(axis).sum()), 1e-8)
    return np.sum(values * axis[None, :, None], axis=1) / denominator


def stats(values: np.ndarray) -> np.ndarray:
    difference = np.diff(values, axis=1)
    output = np.stack(
        (
            values.mean(axis=1),
            values.std(axis=1),
            values[:, 0],
            values[:, -1],
            values[:, -1] - values[:, 0],
            slope(values),
            difference.mean(axis=1),
            difference.std(axis=1),
        ),
        axis=2,
    )
    return output.astype(np.float32)


def block_stat(values: np.ndarray, blocks: int, kind: str) -> np.ndarray:
    edges = np.linspace(0, values.shape[1], blocks + 1, dtype=int)
    pooled = []
    for left, right in zip(edges[:-1], edges[1:]):
        current = values[:, left:right]
        if kind == "mean":
            pooled.append(current.mean(axis=1))
        elif kind == "std":
            pooled.append(current.std(axis=1))
        elif kind == "slope":
            pooled.append(slope(current))
        else:
            raise ValueError(kind)
    return np.stack(pooled, axis=2).astype(np.float32)


def history_dct(values: np.ndarray, count: int) -> np.ndarray:
    transformed = dct(values, type=2, axis=1, norm="ortho", workers=1)
    return transformed[:, :count].transpose(0, 2, 1).astype(np.float32)


def paired_recent(values: np.ndarray, length: int = 24) -> np.ndarray:
    recent = values[:, -min(length, values.shape[1]) :]
    pairs = min(12, recent.shape[1])
    means = block_stat(recent, pairs, "mean")
    deltas = block_stat(np.diff(recent, axis=1), pairs, "mean")
    summaries = []
    for current in (8, 16, 24):
        window = recent[:, -min(current, recent.shape[1]) :]
        summaries.extend(
            (window.mean(axis=1), window.std(axis=1), slope(window), window[:, -1] - window[:, 0])
        )
    return np.concatenate((means, deltas, np.stack(summaries, axis=2)), axis=2).astype(np.float32)


def cross_channel_features(history: np.ndarray) -> np.ndarray:
    channels = history.shape[2]
    global_mean = history.mean(axis=2, keepdims=True)
    global_std = history.std(axis=2, keepdims=True)
    mean_blocks = block_stat(global_mean, 12, "mean")
    std_blocks = block_stat(global_std, 12, "mean")
    target_blocks = block_stat(history, 12, "mean")
    repeated_mean = np.repeat(mean_blocks, channels, axis=1)
    repeated_std = np.repeat(std_blocks, channels, axis=1)
    relative = target_blocks - repeated_mean
    latest_mean = global_mean[:, -1]
    latest_std = np.maximum(global_std[:, -1], 1e-4)
    latest_z = (history[:, -1] - latest_mean) / latest_std
    global_summary = np.stack(
        (
            global_mean.mean(axis=1)[:, 0],
            global_mean.std(axis=1)[:, 0],
            slope(global_mean)[:, 0],
            global_mean[:, -1, 0],
            global_std.mean(axis=1)[:, 0],
        ),
        axis=1,
    )
    global_summary = np.repeat(global_summary[:, None], channels, axis=1)
    return np.concatenate(
        (repeated_mean, repeated_std, relative, latest_z[:, :, None], global_summary), axis=2
    ).astype(np.float32)


def fit_ridge_map(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
    cap: float | None,
) -> RidgeMap:
    feature_mean = features.mean(axis=0, dtype=np.float64)
    feature_scale = np.maximum(features.std(axis=0, dtype=np.float64), 1e-5)
    target_mean = target.mean(axis=0, dtype=np.float64)
    target_scale = np.maximum(target.std(axis=0, dtype=np.float64), 1e-4)
    weights: List[np.ndarray] = []
    for channel in range(features.shape[1]):
        current_x = (
            features[:, channel].astype(np.float64) - feature_mean[channel][None]
        ) / feature_scale[channel][None]
        current_y = (
            target[:, channel].astype(np.float64) - target_mean[channel][None]
        ) / target_scale[channel][None]
        gram = current_x.T @ current_x
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        weights.append(np.linalg.solve(gram, current_x.T @ current_y))
    return RidgeMap(
        feature_mean.astype(np.float32),
        feature_scale.astype(np.float32),
        target_mean.astype(np.float32),
        target_scale.astype(np.float32),
        weights,
        cap,
    )


def predict_ridge_map(model: RidgeMap, features: np.ndarray) -> np.ndarray:
    output = np.empty(
        (len(features), features.shape[1], model.target_mean.shape[1]), dtype=np.float32
    )
    for channel, weight in enumerate(model.weights):
        current = (
            features[:, channel].astype(np.float64) - model.feature_mean[channel][None]
        ) / model.feature_scale[channel][None]
        normalized = current @ weight
        if model.cap is not None:
            normalized = np.clip(normalized, -model.cap, model.cap)
        output[:, channel] = (
            normalized * model.target_scale[channel][None]
            + model.target_mean[channel][None]
        ).astype(np.float32)
    return output


def rolling_origin_folds(index: np.ndarray, blocks: int, purge: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(index)
    pieces = [part for part in np.array_split(order, blocks) if len(part)]
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for block_id in range(1, len(pieces)):
        held = pieces[block_id]
        cutoff = int(np.min(index[held])) - int(purge)
        fit = order[index[order] < cutoff]
        if len(fit) < 96:
            continue
        folds.append((fit.astype(np.int64), held.astype(np.int64)))
    if not folds:
        raise RuntimeError("no rolling-origin folds")
    return folds
