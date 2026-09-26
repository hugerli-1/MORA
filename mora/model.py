"""Frozen-backbone Model-Output Residual Adaptation (MORA)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .config import MORAConfig
from .coarse_future import CoarseFutureEstimation
from .stagewise import StagewiseResidualAdaptation
from .data import ForecastBatch, OOFForecasts, integer_indices


class MORA:
    def __init__(self, config: MORAConfig):
        self.config = config
        self.coarse_future = None
        self.residual_adapter = None
        self.fit_summary = {}

    @classmethod
    def fit(cls, config, training_series, oof: OOFForecasts,
            validation: ForecastBatch, progress=None):
        """Fit from training-only OOF forecasts and official validation targets.

        `training_series` contains only normalized training rows. Neither this
        method nor any fitted coarse estimator accepts a test split.
        """
        series = np.asarray(training_series, dtype=np.float32)
        if series.ndim != 2 or not np.isfinite(series).all():
            raise ValueError("training_series must be a finite [T_train,C] array")
        oof.validate(config, series)
        validation.validate(config, require_targets=True)
        if validation.history.shape[2] != series.shape[1]:
            raise ValueError("Validation and training channels differ")
        if np.min(validation.sample_index + config.history_length) < len(series):
            raise ValueError("Validation target origins must be outside the training split")
        model = cls(config)
        q = config.intervals
        count = 3 if config.use_periodic_template else 2
        n, _, c = oof.batch.history.shape
        extrapolated = np.empty((n, c, count, q), dtype=np.float32)
        learned = np.empty((n, c, q), dtype=np.float32)
        for fold, interval in enumerate(oof.held_raw_intervals):
            if progress:
                progress(f"Fitting coarse estimates for training block {fold + 1}/{len(oof.held_raw_intervals)}")
            estimator = CoarseFutureEstimation(config).fit(series, tuple(int(x) for x in interval))
            held = oof.fold_id == fold
            extrapolated[held], learned[held] = estimator.predict_batched(
                oof.batch.history[held], oof.batch.sample_index[held])
        if progress:
            progress("Refitting coarse estimators on the full training split")
        model.coarse_future = CoarseFutureEstimation(config).fit(series)
        val_extrapolated, val_learned = model.coarse_future.predict_batched(
            validation.history, validation.sample_index)
        if progress:
            progress("Fitting stagewise residual maps and validation calibration")
        model.residual_adapter = StagewiseResidualAdaptation(config).fit(
            oof, extrapolated, learned, validation, val_extrapolated, val_learned)
        model.fit_summary = dict(model.residual_adapter.fit_summary,
                                 coarse_estimator=config.coarse_estimator,
                                 training_rows=len(series), test_targets_used=False)
        return model

    def predict(self, history, backbone_prediction, sample_index):
        """Refine one frozen forecast, using no future targets or extra history."""
        if self.coarse_future is None or self.residual_adapter is None:
            raise RuntimeError("Fit or load MORA before prediction")
        x = np.asarray(history, dtype=np.float32)
        prediction = np.asarray(backbone_prediction, dtype=np.float32)
        index = integer_indices(sample_index)
        c = self.residual_adapter.first_map.weight.shape[0]
        if x.shape != (len(index), self.config.history_length, c) or prediction.shape != (len(index), self.config.horizon, c):
            raise ValueError("Expected history [N,L,C], forecast [N,H,C], and origins [N]")
        if not np.isfinite(x).all() or not np.isfinite(prediction).all():
            raise ValueError("History and backbone prediction must be finite")
        result = np.empty_like(prediction)
        for start in range(0, len(index), self.config.batch_size):
            end = start + self.config.batch_size
            first, second = self.coarse_future.predict(x[start:end], index[start:end])
            result[start:end] = self.residual_adapter.predict(prediction[start:end], first, second)
        return result

    def save(self, path):
        if self.coarse_future is None or self.residual_adapter is None:
            raise RuntimeError("Cannot save an unfitted MORA model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.coarse_future.state_arrays()
        arrays.update(self.residual_adapter.state_arrays())
        arrays["metadata"] = np.asarray(json.dumps({"format_version": 1,
            "method": "Model-Output Residual Adaptation", "config": self.config.to_dict()}, allow_nan=False))
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        metadata = json.loads(str(arrays.pop("metadata").item()))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported MORA state format")
        config = MORAConfig.from_dict(metadata["config"])
        result = cls(config)
        result.coarse_future = CoarseFutureEstimation.from_arrays(config, arrays)
        result.residual_adapter = StagewiseResidualAdaptation.from_arrays(config, arrays)
        return result
