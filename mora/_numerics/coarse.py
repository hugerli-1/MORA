"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .history import block_stat
from .history import cross_channel_features
from .history import fit_ridge_map
from .history import history_dct
from .history import paired_recent
from .history import predict_ridge_map
from .history import stats
from .history import time_features
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
import json
import numpy as np

@dataclass
class CandidateModel:
    blocks: int
    alpha: float
    model: object


def segment_mean_operator(horizon: int, blocks: int) -> np.ndarray:
    if blocks < 1 or blocks >= horizon:
        raise ValueError((horizon, blocks))
    edges = np.linspace(0, horizon, blocks + 1, dtype=int)
    matrix = np.zeros((blocks, horizon), dtype=np.float64)
    for row, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        matrix[row, left:right] = 1.0 / float(right - left)
    return matrix


def segment_means(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("qh,nhc->ncq", matrix, values, optimize=True).astype(np.float32)


def normalization_state(
    history: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    channels = history.shape[2]
    if mode == "absolute":
        location = np.zeros((len(history), channels), dtype=np.float32)
        scale = np.ones_like(location)
    else:
        location = history[:, -1].astype(np.float32)
        if mode == "last_centered":
            scale = np.ones_like(location)
        elif mode == "last_scaled":
            scale = np.maximum(history.std(axis=1), 1e-3).astype(np.float32)
        else:
            raise ValueError(mode)
    normalized = (
        (history - location[:, None]) / scale[:, None]
    ).astype(np.float32)
    return normalized, location, scale


def encode_target(
    target: np.ndarray, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return ((target - location[:, :, None]) / scale[:, :, None]).astype(np.float32)


def decode_target(
    target: np.ndarray, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (target * scale[:, :, None] + location[:, :, None]).astype(np.float32)


def coarse_features(
    history: np.ndarray, index: np.ndarray, period: int, feature_set: str
) -> np.ndarray:
    channels = history.shape[2]
    calendar = time_features(index, period, history.shape[1], channels)
    if feature_set == "calendar_only":
        return calendar.astype(np.float32)
    minimal = (
        stats(history),
        block_stat(history, 12, "mean"),
        calendar,
    )
    if feature_set == "minimal":
        return np.concatenate(minimal, axis=2).astype(np.float32)
    compact = (
        history_dct(history, 16),
        *minimal,
        paired_recent(history),
    )
    if feature_set == "compact":
        return np.concatenate(compact, axis=2).astype(np.float32)
    phase_interaction = np.einsum(
        "ncs,ncp->ncsp",
        np.concatenate(
            (
                history_dct(history, 8),
                stats(history),
            ),
            axis=2,
        ),
        time_features(index, period, history.shape[1], channels),
        optimize=True,
    ).reshape(len(history), channels, -1)
    if feature_set == "compact_phase_interaction":
        return np.concatenate((*compact, phase_interaction), axis=2).astype(
            np.float32
        )
    if feature_set not in {"full", "full_no_time", "full_phase_interaction"}:
        raise ValueError(feature_set)
    difference = np.diff(history, axis=1)
    full = (
        (
            history.transpose(0, 2, 1),
            history_dct(history, 16),
            stats(history),
            block_stat(history, 12, "mean"),
            block_stat(history, 12, "std"),
            paired_recent(history),
            cross_channel_features(history),
            calendar,
            difference[:, -min(24, difference.shape[1]) :].transpose(0, 2, 1),
        )
    )
    if feature_set == "full_no_time":
        full = tuple(value for value in full if value is not calendar)
    if feature_set == "full_phase_interaction":
        full = (*full, phase_interaction)
    return np.concatenate(full, axis=2).astype(np.float32)


def select_ridges(
    features: np.ndarray,
    targets: Dict[int, np.ndarray],
    index: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    alphas: Sequence[float],
) -> Tuple[Dict[int, CandidateModel], List[Dict[str, object]]]:
    slices: Dict[int, slice] = {}
    offset = 0
    pieces = []
    for blocks, target in targets.items():
        slices[blocks] = slice(offset, offset + target.shape[2])
        offset += target.shape[2]
        pieces.append(target)
    joint_target = np.concatenate(pieces, axis=2)
    rows: List[Dict[str, object]] = []
    best: Dict[int, Tuple[float, float]] = {}
    for alpha in alphas:
        predictions: Dict[int, List[np.ndarray]] = {blocks: [] for blocks in targets}
        truths: Dict[int, List[np.ndarray]] = {blocks: [] for blocks in targets}
        fold_mse: Dict[int, List[float]] = {blocks: [] for blocks in targets}
        for fit, held in folds:
            model = fit_ridge_map(features[fit], joint_target[fit], alpha, None)
            prediction = predict_ridge_map(model, features[held])
            for blocks, target in targets.items():
                current = prediction[:, :, slices[blocks]]
                current_mse = float(
                    np.square(current.astype(np.float64) - target[held]).mean()
                )
                predictions[blocks].append(current)
                truths[blocks].append(target[held])
                fold_mse[blocks].append(current_mse)
        for blocks in targets:
            aggregate = float(
                np.square(
                    np.concatenate(predictions[blocks]).astype(np.float64)
                    - np.concatenate(truths[blocks])
                ).mean()
            )
            rows.append(
                {
                    "blocks": blocks,
                    "alpha": alpha,
                    "rolling_moment_mse": aggregate,
                    "fold_mse": json.dumps(fold_mse[blocks]),
                }
            )
            if blocks not in best or aggregate < best[blocks][0]:
                best[blocks] = (aggregate, alpha)
    models = {
        blocks: CandidateModel(
            blocks,
            float(best[blocks][1]),
            fit_ridge_map(features, target, float(best[blocks][1]), None),
        )
        for blocks, target in targets.items()
    }
    return models, rows
