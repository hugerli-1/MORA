"""Factorized coarse estimator for series with many variables."""

from __future__ import annotations

import numpy as np
from .data import windows
from ._numerics.factor_space import fit_factor_basis, project_factors, reconstruct_coarse_states
from ._numerics.factor_features import fit_periodic_baseline, factor_history_features, select_factor_predictor
from ._numerics.coarse import segment_mean_operator, segment_means
from ._numerics.shared_ridge import GlobalRidge, fit_global, predict_global
from ._numerics.history import rolling_origin_folds


class FactorizedCoarseEstimator:
    def __init__(self, config):
        self.config = config
        self.A = segment_mean_operator(config.horizon, config.intervals)
        self.regressor = None

    def fit(self, training_series, origins, raw_fit_mask):
        config = self.config
        self.periodic_baseline = fit_periodic_baseline(training_series, raw_fit_mask, config.phase_period)
        anomaly = training_series - self.periodic_baseline[np.arange(len(training_series)) % config.phase_period]
        self.location, self.scale, self.loadings, _ = fit_factor_basis(anomaly, raw_fit_mask, config.factor_rank)
        factors = project_factors(anomaly, self.location, self.scale, self.loadings)
        history = windows(factors, origins, config.history_length)
        future = windows(factors, origins, config.horizon, config.history_length)
        train = {"history": history, "true": future, "index": origins}
        folds = rolling_origin_folds(origins, config.rolling_blocks, config.history_length + config.horizon - 1)
        probe = {"history": history[:1], "index": origins[:1]}
        _, _, selection = select_factor_predictor(train, probe, probe, self.A, config.base_period, folds, config.ridge_alphas)
        chosen = selection[-1]
        self.prediction_rule = chosen["mode"]
        self.ridge_alpha = float(chosen["alpha"])
        absolute = segment_means(future, self.A)
        relative = absolute - history[:, -1, :, None]
        features = factor_history_features(history, origins, config.base_period)
        self.regressor = fit_global(features, np.concatenate((absolute, relative), axis=2), self.ridge_alpha)
        return self

    def predict(self, history, sample_index):
        config = self.config
        position = sample_index[:, None] + np.arange(history.shape[1], dtype=np.int64)
        anomaly = history - self.periodic_baseline[position % config.phase_period]
        factors = project_factors(anomaly, self.location, self.scale, self.loadings)
        features = factor_history_features(factors, sample_index, config.base_period)
        rank, blocks = self.loadings.shape[1], config.intervals
        joint = predict_global(self.regressor, features, rank, 2 * blocks)
        absolute = joint[:, :, :blocks]
        relative = joint[:, :, blocks:] + factors[:, -1, :, None]
        if self.prediction_rule == "absolute":
            predicted = absolute
        elif self.prediction_rule == "last_delta":
            predicted = relative
        elif self.prediction_rule == "balanced":
            predicted = 0.5 * (absolute + relative)
        else:
            raise ValueError("Unknown fixed factor prediction rule")
        restored = reconstruct_coarse_states(predicted, self.location, self.scale, self.loadings)
        future_position = sample_index[:, None] + history.shape[1] + np.arange(config.horizon, dtype=np.int64)
        return restored + segment_means(self.periodic_baseline[future_position % config.phase_period], self.A)

    def state_arrays(self):
        result = {"factor_periodic_baseline": self.periodic_baseline,
                  "factor_location": self.location, "factor_scale": self.scale,
                  "factor_loadings": self.loadings, "factor_rule": np.asarray(self.prediction_rule)}
        for name in ("feature_mean", "feature_scale", "target_mean", "target_scale", "weight"):
            result["factor_" + name] = np.asarray(getattr(self.regressor, name))
        return result

    @classmethod
    def from_arrays(cls, config, arrays):
        result = cls(config)
        for name in ("periodic_baseline", "location", "scale", "loadings"):
            setattr(result, name, arrays["factor_" + name])
        result.prediction_rule = str(arrays["factor_rule"].item())
        result.regressor = GlobalRidge(*(arrays["factor_" + name] for name in (
            "feature_mean", "feature_scale", "target_mean", "target_scale", "weight")))
        return result
