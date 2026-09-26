"""Explicit query indices and prediction-array contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


def integer_indices(value, name="sample_index"):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = raw.astype(np.int64)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def windows(series, index, length, offset=0):
    series = np.asarray(series, dtype=np.float32)
    index = integer_indices(index)
    positions = index[:, None] + offset + np.arange(length, dtype=np.int64)
    if series.ndim != 2 or not len(index) or positions.max() >= len(series):
        raise ValueError("Window indices exceed the supplied series")
    return series[positions]


@dataclass
class ForecastBatch:
    history: np.ndarray
    backbone_prediction: np.ndarray
    sample_index: np.ndarray
    y_true: np.ndarray | None = None

    def validate(self, config, require_targets=False):
        self.sample_index = integer_indices(self.sample_index)
        self.history = np.asarray(self.history, dtype=np.float32)
        self.backbone_prediction = np.asarray(self.backbone_prediction, dtype=np.float32)
        if self.history.ndim != 3 or not len(self.sample_index):
            raise ValueError("history must have nonempty shape [N,L,C]")
        n, length, channels = self.history.shape
        if length != config.history_length or len(self.sample_index) != n:
            raise ValueError("History shape and sample indices disagree")
        if self.backbone_prediction.shape != (n, config.horizon, channels):
            raise ValueError("backbone_prediction must have shape [N,H,C]")
        if len(np.unique(self.sample_index)) != n or np.any(np.diff(self.sample_index) <= 0):
            raise ValueError("Fitting/evaluation batches require sorted, unique query origins")
        if require_targets and self.y_true is None:
            raise ValueError("Targets are required for fitting or evaluation")
        if self.y_true is not None:
            self.y_true = np.asarray(self.y_true, dtype=np.float32)
            if self.y_true.shape != self.backbone_prediction.shape:
                raise ValueError("Target and backbone shapes differ")
        for array in (self.history, self.backbone_prediction, self.y_true):
            if array is not None and not np.isfinite(array).all():
                raise ValueError("Inputs must be finite; apply the benchmark preprocessing first")
        return self

    @classmethod
    def load(cls, path, targets=True):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["history"], data["backbone_prediction"], data["sample_index"],
                       data["y_true"] if targets and "y_true" in data.files else None)


@dataclass
class OOFForecasts:
    batch: ForecastBatch
    fold_id: np.ndarray
    held_raw_intervals: np.ndarray

    def validate(self, config, training_series):
        batch = self.batch.validate(config, require_targets=True)
        self.fold_id = integer_indices(self.fold_id, "fold_id")
        intervals = np.asarray(self.held_raw_intervals)
        if intervals.ndim != 2 or intervals.shape[1] != 2 or intervals.dtype.kind not in "iu":
            raise ValueError("held_raw_intervals must have shape [folds,2], integer half-open bounds")
        self.held_raw_intervals = intervals.astype(np.int64)
        if len(self.fold_id) != len(batch.sample_index):
            raise ValueError("fold_id and OOF queries must align exactly")
        if not np.array_equal(np.unique(self.fold_id), np.arange(len(intervals))) or len(intervals) < 2:
            raise ValueError("Folds must be nonempty and numbered consecutively from zero")
        if np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] > len(training_series)) or np.any(intervals[:, 0] >= intervals[:, 1]):
            raise ValueError("Held intervals must be contained in the training split")
        if np.any(intervals[1:, 0] < intervals[:-1, 0]):
            raise ValueError("Held intervals must be chronological")
        span = config.history_length + config.horizon
        for fold, (left, right) in enumerate(intervals):
            origins = batch.sample_index[self.fold_id == fold]
            if np.any(origins < left) or np.any(origins + span > right):
                raise ValueError("Held interval does not contain all held histories and futures")
        for start in range(0, len(batch.sample_index), config.batch_size):
            section = slice(start, start + config.batch_size)
            idx = batch.sample_index[section]
            x = windows(training_series, idx, config.history_length)
            y = windows(training_series, idx, config.horizon, config.history_length)
            if not np.allclose(x, batch.history[section], rtol=2e-5, atol=2e-5):
                raise ValueError("OOF histories do not match training indices/normalization")
            if not np.allclose(y, batch.y_true[section], rtol=2e-5, atol=2e-5):
                raise ValueError("OOF targets do not match training indices/normalization")
        return self

    @classmethod
    def load(cls, path):
        batch = ForecastBatch.load(path)
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(batch, data["fold_id"], data["held_raw_intervals"])


def blocked_oof_geometry(training_rows, history_length, horizon, stride=6, blocks=4):
    """Four held blocks in the latter half of training-window origins."""
    count = training_rows - history_length - horizon + 1
    if count <= blocks * stride or stride < 1:
        raise ValueError("Not enough training windows for blocked OOF")
    edges = np.linspace(int(round(count * 0.5)), count, blocks + 1, dtype=np.int64)
    indices, labels, intervals = [], [], []
    for fold, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = np.arange(left, right, dtype=np.int64)
        selected = selected[selected % stride == 0]
        if not len(selected):
            raise ValueError("An OOF block is empty")
        indices.append(selected)
        labels.append(np.full(len(selected), fold, dtype=np.int64))
        intervals.append((left, right - 1 + history_length + horizon))
    return np.concatenate(indices), np.concatenate(labels), np.asarray(intervals, dtype=np.int64)


def legal_training_origins(training_rows, history_length, horizon, held_interval=None, stride=1):
    span = history_length + horizon
    index = np.arange(0, training_rows - span + 1, stride, dtype=np.int64)
    if held_interval is not None:
        left, right = held_interval
        index = index[(index + span <= left) | (index >= right)]
    if not len(index):
        raise ValueError("No non-overlapping training windows remain")
    return index
