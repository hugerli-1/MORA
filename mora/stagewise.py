"""Stagewise Residual Adaptation in the paper's A, U, K, G, S notation."""

from __future__ import annotations

import numpy as np
from ._numerics.basis import boundary_basis
from ._numerics.coarse import segment_mean_operator, segment_means
from ._numerics.residual_mapping import ResidualMap, predict_residual_map
from ._numerics.stagewise import (fit_stage, stage_gain, fit_final_calibration,
    contiguous_labels, stack_coarse_operator, coarse_discrepancy, measured_correction)
from ._numerics.projection import reconstruct


class StagewiseResidualAdaptation:
    def __init__(self, config):
        self.config = config
        self.A = segment_mean_operator(config.horizon, config.intervals)
        self.basis = boundary_basis(config.horizon, config.rank, config.taper)
        self.first_map = self.second_map = None
        self.G_e = self.G_l = self.S = None
        self.fit_summary = {}

    @property
    def U(self):
        """H x k correction basis; internal routines store its transpose."""
        return self.basis.T

    @property
    def K_e(self):
        return self.first_map.weight.transpose(0, 2, 1)

    @property
    def K_l(self):
        return self.second_map.weight.transpose(0, 2, 1)

    def fit(self, oof, oof_extrapolated, oof_learned, validation,
            validation_extrapolated, validation_learned):
        config = self.config
        residual = (
            np.einsum("rh,nhc->ncr", self.basis, oof.batch.y_true, optimize=True).astype(np.float32)
            - np.einsum("rh,nhc->ncr", self.basis, oof.batch.backbone_prediction, optimize=True).astype(np.float32)
        )
        base_coarse = segment_means(oof.batch.backbone_prediction, self.A)
        expected = (len(residual), residual.shape[1], config.intervals)
        if oof_learned.shape != expected or oof_extrapolated.shape[:2] != expected[:2] or oof_extrapolated.shape[-1] != expected[-1]:
            raise ValueError("Coarse estimates must align with OOF queries and channels")
        measurement = self.A @ self.U.astype(np.float64)
        extrapolation_measurement = stack_coarse_operator(measurement, oof_extrapolated.shape[2])
        first_discrepancy = coarse_discrepancy(oof_extrapolated, base_coarse)
        second_discrepancy = oof_learned - base_coarse
        shared = dict(cap=config.coefficient_cap, compute_backend=config.compute_backend,
                      cuda_device=config.cuda_device)
        first = fit_stage(("extrapolation",), first_discrepancy, residual,
            extrapolation_measurement, oof.fold_id, *config.mapping_grid, **shared)
        self.first_map = first.model
        self.G_e = stage_gain(first, residual, "pooled", config.coefficient_cap)
        first_crossfit = first.oof_raw * self.G_e[None]
        remaining = residual - first_crossfit
        adjusted = second_discrepancy - measured_correction(first_crossfit, measurement)
        second = fit_stage(("learned",), adjusted, remaining, measurement,
            oof.fold_id, *config.mapping_grid, **shared)
        self.second_map = second.model
        self.G_l = stage_gain(second, remaining, "common", config.coefficient_cap)
        self.S = np.ones(residual.shape[1:], dtype=np.float32)
        validation_coefficients = self.coefficients(validation.backbone_prediction,
            validation_extrapolated, validation_learned, calibrated=False)
        if len(validation.sample_index) < 4:
            raise ValueError("Calibration requires at least four validation queries")
        self.S = fit_final_calibration(validation_coefficients, None, self.basis,
            contiguous_labels(len(validation.sample_index)), config.coefficient_cap,
            config.calibration_mse_weight, config.batch_size, config.channel_chunk_size,
            anchor=validation.backbone_prediction, target=validation.y_true)
        self.fit_summary = {"extrapolation_stage": first.parameters,
                            "learned_stage": second.parameters,
                            "training_queries": len(residual),
                            "validation_queries": len(validation.sample_index)}
        return self

    def coefficients(self, backbone_prediction, extrapolated, learned, calibrated=True):
        if self.first_map is None or self.second_map is None:
            raise RuntimeError("Fit residual mappings before inference")
        baseline = segment_means(backbone_prediction, self.A)
        d_e = coarse_discrepancy(extrapolated, baseline)
        z_e = predict_residual_map(self.first_map, d_e) * self.G_e[None]
        measurement = self.A @ self.U.astype(np.float64)
        d_l = learned - baseline - measured_correction(z_e, measurement)
        z_l = predict_residual_map(self.second_map, d_l) * self.G_l[None]
        result = z_e + z_l
        if calibrated:
            result = result * self.S[None]
        return result.astype(np.float32)

    def predict(self, backbone_prediction, extrapolated, learned):
        coefficients = self.coefficients(backbone_prediction, extrapolated, learned)
        return backbone_prediction + reconstruct(coefficients, self.basis)

    def state_arrays(self):
        result = {"correction_basis": self.basis, "G_e": self.G_e,
                  "G_l": self.G_l, "S": self.S}
        for name, model in (("extrapolation", self.first_map), ("learned", self.second_map)):
            result[name + "_map_weight"] = model.weight
            floor = model.sample_scale_floor
            result[name + "_map_floor"] = floor if floor is not None else np.empty(0, dtype=np.float32)
            result[name + "_map_power"] = np.asarray(model.sample_scale_power)
        return result

    @classmethod
    def from_arrays(cls, config, arrays):
        result = cls(config)
        result.basis = np.asarray(arrays["correction_basis"], dtype=np.float32)
        if result.basis.shape != (config.rank, config.horizon):
            raise ValueError("Saved correction basis has the wrong shape")
        if not np.allclose(result.basis @ result.basis.T, np.eye(config.rank), atol=2e-5) or np.max(np.abs(result.basis[:, 0])) > 1e-7:
            raise ValueError("Saved correction basis violates the orthogonality/boundary contract")
        result.G_e, result.G_l, result.S = arrays["G_e"], arrays["G_l"], arrays["S"]
        for name, attribute in (("extrapolation", "first_map"), ("learned", "second_map")):
            floor = arrays[name + "_map_floor"]
            model = ResidualMap(arrays[name + "_map_weight"], floor if floor.size else None,
                                float(arrays[name + "_map_power"]), "numpy", 0)
            setattr(result, attribute, model)
        return result
