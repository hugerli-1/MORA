"""Coarse Future Estimation: extrapolation and learned segment-wise means."""

from __future__ import annotations

import numpy as np
from .config import MORAConfig
from .data import legal_training_origins, windows
from ._numerics.coarse import (segment_mean_operator, segment_means,
    coarse_features, normalization_state, encode_target, select_ridges)
from ._numerics.history import rolling_origin_folds, RidgeMap
from ._numerics.coarse_inference import (
    ChannelWisePrediction, PeriodicTemplateEstimator, _harmonic_design)
from ._numerics.periodic import select_template_periods


def ridge_arrays(model, prefix):
    result = {prefix + name: np.asarray(getattr(model, name)) for name in (
        "feature_mean", "feature_scale", "target_mean", "target_scale")}
    result[prefix + "weights"] = np.stack(model.weights)
    return result


def load_ridge(data, prefix):
    return RidgeMap(*(np.asarray(data[prefix + name], dtype=np.float32) for name in (
        "feature_mean", "feature_scale", "target_mean", "target_scale")),
        [row for row in np.asarray(data[prefix + "weights"])], None)


class ChannelWiseCoarseEstimator:
    """Two ridge predictions of A Y, in absolute and last-centered coordinates."""

    def __init__(self, config):
        self.config = config
        self.predictor = None

    def fit(self, series, origins, raw_fit_mask=None):
        config = self.config
        self.location = np.zeros(series.shape[1], dtype=np.float64)
        self.scale = np.ones(series.shape[1], dtype=np.float64)
        # Fold-specific normalization follows the frozen OOF implementation.
        if raw_fit_mask is not None:
            selected = series[raw_fit_mask]
            self.location = selected.mean(axis=0, dtype=np.float64)
            self.scale = np.maximum(selected.std(axis=0, dtype=np.float64), 1e-5)
        normalized = ((series.astype(np.float64) - self.location) / self.scale).astype(np.float32)
        history = windows(normalized, origins, config.history_length)
        future = windows(normalized, origins, config.horizon, config.history_length)
        aggregate = segment_mean_operator(config.horizon, config.intervals)
        target = segment_means(future, aggregate)
        folds = rolling_origin_folds(origins, config.rolling_blocks, config.history_length + config.horizon - 1)
        fitted, self.selection = [], {}
        for mode in ("absolute", "last_centered"):
            x, location, scale = normalization_state(history, mode)
            features = coarse_features(x, origins, config.base_period, config.feature_set)
            targets = encode_target(target, location, scale)
            models, _ = select_ridges(features, {config.intervals: targets}, origins, folds, config.ridge_alphas)
            fitted.append(models[config.intervals].model)
            self.selection[mode] = float(models[config.intervals].alpha)
        self.predictor = ChannelWisePrediction(config.base_period, config.feature_set, *fitted)
        return self

    def predict(self, history, index):
        if self.predictor is None:
            raise RuntimeError("Fit the coarse estimator before prediction")
        x = ((history.astype(np.float64) - self.location) / self.scale).astype(np.float32)
        predicted = self.predictor.predict(x, index)
        return (predicted.astype(np.float64) * self.scale[None, :, None] + self.location[None, :, None]).astype(np.float32)

    def state_arrays(self):
        arrays = {"learned_location": self.location, "learned_scale": self.scale}
        arrays.update(ridge_arrays(self.predictor.absolute, "absolute_"))
        arrays.update(ridge_arrays(self.predictor.centered, "relative_"))
        return arrays

    @classmethod
    def from_arrays(cls, config, arrays):
        result = cls(config)
        result.location, result.scale = arrays["learned_location"], arrays["learned_scale"]
        result.predictor = ChannelWisePrediction(config.base_period, config.feature_set,
            load_ridge(arrays, "absolute_"), load_ridge(arrays, "relative_"))
        result.selection = {}
        return result


class CoarseFutureEstimation:
    """The first MORA component; never receives a backbone forecast to fit."""

    def __init__(self, config: MORAConfig):
        self.config = config
        self.A = segment_mean_operator(config.horizon, config.intervals)
        self.periodic_template = None
        self.learned = None

    @property
    def extrapolation_names(self):
        base = ("current_level", "recent_pattern")
        return base + (("periodic_template",) if self.config.use_periodic_template else ())

    def fit(self, training_series, held_interval=None):
        series = np.asarray(training_series, dtype=np.float32)
        config = self.config
        mask = np.ones(len(series), dtype=bool)
        if held_interval is not None:
            mask[slice(*held_interval)] = False
        origins = legal_training_origins(len(series), config.history_length, config.horizon,
                                         held_interval, config.stride)
        if config.coarse_estimator == "factorized":
            from .factorized_estimator import FactorizedCoarseEstimator
            self.learned = FactorizedCoarseEstimator(config).fit(series, origins, mask)
        else:
            self.learned = ChannelWiseCoarseEstimator(config).fit(
                series, origins, mask if held_interval is not None else None)
        if config.use_periodic_template:
            periods = select_template_periods(series, len(series), config.base_period,
                                             config.template_period_count, train_fit_mask=mask)
            design = _harmonic_design(np.arange(len(series), dtype=np.float64), periods)[mask]
            gram = design.T @ design
            penalty = np.eye(gram.shape[0]) * (1e-3 * max(float(mask.sum()), 1.0))
            penalty[0, 0] = 1e-8
            coefficients = np.linalg.solve(gram + penalty, design.T @ series[mask].astype(np.float64))
            self.periodic_template = PeriodicTemplateEstimator(periods, coefficients, config.template_half_life)
        return self

    def predict(self, history, sample_index):
        history = np.asarray(history, dtype=np.float32)
        config = self.config
        current_level = np.repeat(history[:, -1:], config.horizon, axis=1)
        period = min(config.base_period, history.shape[1])
        continuation = history[:, -period:][:, np.arange(config.horizon) % period]
        extrapolations = [segment_means(current_level, self.A), segment_means(continuation, self.A)]
        if config.use_periodic_template:
            extrapolations.append(self.periodic_template.predict(history, sample_index, config.horizon, self.A))
        return np.stack(extrapolations, axis=2), self.learned.predict(history, sample_index)

    def predict_batched(self, history, sample_index):
        first, second = [], []
        for start in range(0, len(history), self.config.batch_size):
            stop = start + self.config.batch_size
            extrapolated, learned = self.predict(history[start:stop], sample_index[start:stop])
            first.append(extrapolated)
            second.append(learned)
        return np.concatenate(first), np.concatenate(second)

    def state_arrays(self):
        result = self.learned.state_arrays()
        if self.periodic_template is not None:
            result.update(template_periods=self.periodic_template.periods,
                          template_coefficients=self.periodic_template.coefficient)
        return result

    @classmethod
    def from_arrays(cls, config, arrays):
        result = cls(config)
        if config.coarse_estimator == "factorized":
            from .factorized_estimator import FactorizedCoarseEstimator
            result.learned = FactorizedCoarseEstimator.from_arrays(config, arrays)
        else:
            result.learned = ChannelWiseCoarseEstimator.from_arrays(config, arrays)
        if config.use_periodic_template:
            result.periodic_template = PeriodicTemplateEstimator(
                arrays["template_periods"], arrays["template_coefficients"], config.template_half_life)
        return result
