"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .reliability import coefficient_gain
from .reliability import coordinate_gain
from .reliability import coordinate_reliability
from dataclasses import dataclass
from typing import Any
import json
import numpy as np

EPS = 1e-12


@dataclass
class ResidualMap:
    weight: np.ndarray
    sample_scale_floor: np.ndarray | None
    sample_scale_power: float
    compute_backend: str = "numpy"
    cuda_device: int = 0


def sample_scale_state(
    innovation: np.ndarray,
    power: float,
    floor_quantile: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if power <= 0.0:
        return np.ones(innovation.shape[:2], dtype=np.float64), None
    rms = np.sqrt(np.mean(np.square(innovation.astype(np.float64)), axis=2))
    floor = np.maximum(np.quantile(rms, floor_quantile, axis=0), 1e-5)
    return np.power(np.maximum(rms, floor[None]), power), floor


def shrink_to_diagonal(matrix: np.ndarray, amount: float) -> np.ndarray:
    diagonal = np.diag(np.diag(matrix))
    return (1.0 - amount) * matrix + amount * diagonal


def shrink_discrepancy_covariance(
    matrix: np.ndarray,
    diagonal_amount: float,
    block_sizes: tuple[int, ...] | None = None,
    cross_block_scale: float = 1.0,
) -> np.ndarray:
    """Shrink a coarse-discrepancy covariance toward block independence and diagonality.

    The block target preserves covariance within a declared mechanism group
    while removing covariance between groups.  Convexly interpolating the
    empirical covariance, its block-diagonal target, and its diagonal target
    keeps the result positive semidefinite.  With no blocks (or a cross-block
    scale of one), this is exactly the historical diagonal shrinkage.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(matrix.shape)
    if not 0.0 <= float(diagonal_amount) <= 1.0:
        raise ValueError(diagonal_amount)
    if not 0.0 <= float(cross_block_scale) <= 1.0:
        raise ValueError(cross_block_scale)
    if block_sizes is None:
        if float(cross_block_scale) != 1.0:
            raise ValueError("cross-block shrinkage requires block sizes")
        return shrink_to_diagonal(matrix, diagonal_amount)
    if len(block_sizes) < 2 or any(int(size) <= 0 for size in block_sizes):
        raise ValueError(block_sizes)
    if sum(int(size) for size in block_sizes) != matrix.shape[0]:
        raise ValueError((block_sizes, matrix.shape))

    block_diagonal = np.zeros_like(matrix)
    start = 0
    for size in block_sizes:
        stop = start + int(size)
        block_diagonal[start:stop, start:stop] = matrix[start:stop, start:stop]
        start = stop
    block_shrunk = (
        float(cross_block_scale) * matrix
        + (1.0 - float(cross_block_scale)) * block_diagonal
    )
    diagonal = np.diag(np.diag(matrix))
    return (
        (1.0 - float(diagonal_amount)) * block_shrunk
        + float(diagonal_amount) * diagonal
    )


def fit_residual_map(
    innovation: np.ndarray,
    residual: np.ndarray,
    measurement: np.ndarray,
    covariance_shrinkage: float,
    noise_scale: float,
    empirical_coupling: float,
    sample_scale_power: float,
    sample_scale_floor_quantile: float,
    covariance_sharing: str = "per_channel",
    compute_backend: str = "numpy",
    cuda_device: int = 0,
    observation_block_sizes: tuple[int, ...] | None = None,
    cross_block_scale: float = 1.0,
) -> ResidualMap:
    if innovation.shape[:2] != residual.shape[:2]:
        raise ValueError((innovation.shape, residual.shape))
    if measurement.shape != (innovation.shape[2], residual.shape[2]):
        raise ValueError((measurement.shape, innovation.shape, residual.shape))
    scale, floor = sample_scale_state(
        innovation, sample_scale_power, sample_scale_floor_quantile
    )
    scaled_innovation = innovation.astype(np.float64) / scale[:, :, None]
    scaled_residual = residual.astype(np.float64) / scale[:, :, None]
    channels = innovation.shape[1]
    if covariance_sharing not in {"per_channel", "pooled_channel"}:
        raise ValueError(covariance_sharing)
    if compute_backend not in {"numpy", "torch_cuda"}:
        raise ValueError(compute_backend)
    weight = np.empty(
        (channels, innovation.shape[2], residual.shape[2]), dtype=np.float32
    )
    measurement64 = measurement.astype(np.float64)
    if covariance_sharing == "pooled_channel":
        pooled_residual = scaled_residual.reshape(-1, scaled_residual.shape[2])
        pooled_innovation = scaled_innovation.reshape(
            -1, scaled_innovation.shape[2]
        )
        coarse_estimation_noise = (
            pooled_innovation - pooled_residual @ measurement64.T
        )
        denominator = max(len(pooled_residual), 1)
        prior = pooled_residual.T @ pooled_residual / denominator
        noise = coarse_estimation_noise.T @ coarse_estimation_noise / denominator
        prior = shrink_to_diagonal(prior, covariance_shrinkage)
        noise = shrink_discrepancy_covariance(
            noise,
            covariance_shrinkage,
            observation_block_sizes,
            cross_block_scale,
        )
        structured_system = (
            measurement64 @ prior @ measurement64.T
            + float(noise_scale) * noise
        )
        structured_right = measurement64 @ prior
        empirical_system = pooled_innovation.T @ pooled_innovation / denominator
        empirical_system = shrink_discrepancy_covariance(
            empirical_system,
            0.0,
            observation_block_sizes,
            cross_block_scale,
        )
        empirical_right = pooled_innovation.T @ pooled_residual / denominator
        coupling = float(empirical_coupling)
        system = (
            (1.0 - coupling) * structured_system
            + coupling * empirical_system
        )
        right = (
            (1.0 - coupling) * structured_right
            + coupling * empirical_right
        )
        ridge = max(float(np.trace(system)) / max(system.shape[0], 1), EPS) * 1e-6
        system.flat[:: system.shape[0] + 1] += ridge
        try:
            pooled_weight = np.linalg.solve(system, right).astype(np.float32)
        except np.linalg.LinAlgError:
            pooled_weight = (np.linalg.pinv(system) @ right).astype(np.float32)
        weight[:] = pooled_weight[None]
        return ResidualMap(
            weight=weight,
            sample_scale_floor=(
                floor.astype(np.float32) if floor is not None else None
            ),
            sample_scale_power=float(sample_scale_power),
            compute_backend=compute_backend,
            cuda_device=int(cuda_device),
        )
    if compute_backend == "torch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda backend requested without CUDA")
        device = torch.device(f"cuda:{int(cuda_device)}")
        dtype = torch.float64
        current_residual = torch.as_tensor(
            scaled_residual, dtype=dtype, device=device
        )
        current_innovation = torch.as_tensor(
            scaled_innovation, dtype=dtype, device=device
        )
        current_measurement = torch.as_tensor(
            measurement64, dtype=dtype, device=device
        )
        coarse_estimation_noise = current_innovation - torch.einsum(
            "ncr,fr->ncf", current_residual, current_measurement
        )
        denominator = max(len(residual), 1)
        prior = torch.einsum(
            "ncr,ncs->crs", current_residual, current_residual
        ) / float(denominator)
        noise = torch.einsum(
            "ncf,ncg->cfg", coarse_estimation_noise, coarse_estimation_noise
        ) / float(denominator)
        shrinkage = float(covariance_shrinkage)
        if shrinkage > 0.0:
            prior = (1.0 - shrinkage) * prior + shrinkage * torch.diag_embed(
                torch.diagonal(prior, dim1=1, dim2=2)
            )
        if observation_block_sizes is not None:
            if (
                len(observation_block_sizes) < 2
                or any(int(size) <= 0 for size in observation_block_sizes)
                or sum(int(size) for size in observation_block_sizes)
                != noise.shape[1]
            ):
                raise ValueError((observation_block_sizes, tuple(noise.shape)))
            if not 0.0 <= float(cross_block_scale) <= 1.0:
                raise ValueError(cross_block_scale)
            block_mask = torch.zeros(
                (noise.shape[1], noise.shape[2]),
                dtype=torch.bool,
                device=device,
            )
            start = 0
            for size in observation_block_sizes:
                stop = start + int(size)
                block_mask[start:stop, start:stop] = True
                start = stop
            block_noise = torch.where(block_mask[None], noise, torch.zeros_like(noise))
            noise = (
                float(cross_block_scale) * noise
                + (1.0 - float(cross_block_scale)) * block_noise
            )
        elif float(cross_block_scale) != 1.0:
            raise ValueError("cross-block shrinkage requires block sizes")
        if shrinkage > 0.0:
            noise = (1.0 - shrinkage) * noise + shrinkage * torch.diag_embed(
                torch.diagonal(noise, dim1=1, dim2=2)
            )
        structured_system = torch.einsum(
            "fr,crs,gs->cfg", current_measurement, prior, current_measurement
        ) + float(noise_scale) * noise
        structured_right = torch.einsum(
            "fr,crs->cfs", current_measurement, prior
        )
        coupling = float(empirical_coupling)
        if coupling > 0.0:
            empirical_system = torch.einsum(
                "ncf,ncg->cfg", current_innovation, current_innovation
            ) / float(denominator)
            if observation_block_sizes is not None:
                block_empirical = torch.where(
                    block_mask[None], empirical_system, torch.zeros_like(empirical_system)
                )
                empirical_system = (
                    float(cross_block_scale) * empirical_system
                    + (1.0 - float(cross_block_scale)) * block_empirical
                )
            empirical_right = torch.einsum(
                "ncf,ncr->cfr", current_innovation, current_residual
            ) / float(denominator)
            system = (
                (1.0 - coupling) * structured_system
                + coupling * empirical_system
            )
            right = (
                (1.0 - coupling) * structured_right
                + coupling * empirical_right
            )
        else:
            system = structured_system
            right = structured_right
        ridge = torch.clamp(
            torch.diagonal(system, dim1=1, dim2=2).sum(dim=1)
            / max(system.shape[1], 1),
            min=EPS,
        ) * 1e-6
        system = system + ridge[:, None, None] * torch.eye(
            system.shape[1], dtype=dtype, device=device
        )[None]
        try:
            solved = torch.linalg.solve(system, right)
        except RuntimeError:
            solved = torch.linalg.pinv(system) @ right
        weight[:] = solved.detach().cpu().numpy().astype(np.float32)
        return ResidualMap(
            weight=weight,
            sample_scale_floor=(
                floor.astype(np.float32) if floor is not None else None
            ),
            sample_scale_power=float(sample_scale_power),
            compute_backend=compute_backend,
            cuda_device=int(cuda_device),
        )
    for channel in range(channels):
        current_residual = scaled_residual[:, channel]
        current_innovation = scaled_innovation[:, channel]
        coarse_estimation_noise = current_innovation - current_residual @ measurement64.T
        prior = current_residual.T @ current_residual / max(len(residual), 1)
        noise = coarse_estimation_noise.T @ coarse_estimation_noise / max(len(residual), 1)
        prior = shrink_to_diagonal(prior, covariance_shrinkage)
        noise = shrink_discrepancy_covariance(
            noise,
            covariance_shrinkage,
            observation_block_sizes,
            cross_block_scale,
        )
        structured_system = (
            measurement64 @ prior @ measurement64.T
            + float(noise_scale) * noise
        )
        structured_right = measurement64 @ prior
        empirical_system = (
            current_innovation.T @ current_innovation / max(len(residual), 1)
        )
        empirical_system = shrink_discrepancy_covariance(
            empirical_system,
            0.0,
            observation_block_sizes,
            cross_block_scale,
        )
        empirical_right = (
            current_innovation.T @ current_residual / max(len(residual), 1)
        )
        coupling = float(empirical_coupling)
        system = (
            (1.0 - coupling) * structured_system
            + coupling * empirical_system
        )
        right = (
            (1.0 - coupling) * structured_right
            + coupling * empirical_right
        )
        ridge = max(float(np.trace(system)) / max(system.shape[0], 1), EPS) * 1e-6
        system.flat[:: system.shape[0] + 1] += ridge
        try:
            weight[channel] = np.linalg.solve(system, right).astype(np.float32)
        except np.linalg.LinAlgError:
            weight[channel] = (np.linalg.pinv(system) @ right).astype(np.float32)
    return ResidualMap(
        weight=weight,
        sample_scale_floor=(floor.astype(np.float32) if floor is not None else None),
        sample_scale_power=float(sample_scale_power),
        compute_backend=compute_backend,
        cuda_device=int(cuda_device),
    )


def predict_residual_map(
    model: ResidualMap,
    innovation: np.ndarray,
) -> np.ndarray:
    if model.sample_scale_power > 0.0:
        if model.sample_scale_floor is None:
            raise ValueError("missing sample-scale floor")
        rms = np.sqrt(np.mean(np.square(innovation.astype(np.float64)), axis=2))
        scale = np.power(
            np.maximum(rms, model.sample_scale_floor.astype(np.float64)[None]),
            model.sample_scale_power,
        )
    else:
        scale = np.ones(innovation.shape[:2], dtype=np.float64)
    normalized = innovation.astype(np.float64) / scale[:, :, None]
    if model.compute_backend == "torch_cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch_cuda backend requested without CUDA")
        device = torch.device(f"cuda:{int(model.cuda_device)}")
        with torch.no_grad():
            prediction = torch.einsum(
                "ncf,cfr->ncr",
                torch.as_tensor(normalized, dtype=torch.float64, device=device),
                torch.as_tensor(
                    model.weight, dtype=torch.float64, device=device
                ),
            )
            prediction = prediction * torch.as_tensor(
                scale[:, :, None], dtype=torch.float64, device=device
            )
            return prediction.cpu().numpy().astype(np.float32)
    prediction = np.einsum(
        "ncf,cfr->ncr", normalized, model.weight.astype(np.float64), optimize=True
    )
    return (prediction * scale[:, :, None]).astype(np.float32)


def cuda_candidate_statistics(
    correction: np.ndarray,
    residual: np.ndarray,
    environment: np.ndarray,
    gain_cap: float,
    cuda_device: int,
) -> dict[str, Any]:
    """Evaluate OOF mode gains and environment risks without CPU copies."""
    import torch

    device = torch.device(f"cuda:{int(cuda_device)}")
    with torch.no_grad():
        current_correction = torch.as_tensor(
            correction, dtype=torch.float64, device=device
        )
        current_residual = torch.as_tensor(
            residual, dtype=torch.float64, device=device
        )
        labels = torch.as_tensor(environment, dtype=torch.int64, device=device)

        def optimum(mask: Any | None = None) -> tuple[Any, Any]:
            selected_correction = (
                current_correction
                if mask is None
                else current_correction[mask]
            )
            selected_residual = (
                current_residual if mask is None else current_residual[mask]
            )
            denominator = torch.square(selected_correction).sum(dim=(0, 1))
            numerator = (selected_correction * selected_residual).sum(
                dim=(0, 1)
            )
            value = torch.where(
                denominator > EPS,
                numerator / torch.clamp(denominator, min=EPS),
                torch.zeros_like(numerator),
            )
            return value, denominator

        pooled, _ = optimum()
        values = torch.unique(labels, sorted=True)
        fold_optima = []
        for value in values:
            current, _ = optimum(labels == value)
            fold_optima.append(current)
        fold_optima_tensor = torch.stack(fold_optima)
        minimum = torch.min(fold_optima_tensor, dim=0).values
        support = minimum > 0.0
        clipped_pooled = torch.clamp(pooled, min=0.0, max=float(gain_cap))
        common_gain = (
            torch.minimum(clipped_pooled, torch.clamp(minimum, min=0.0))
            * support
        )
        pooled_oof_gain = clipped_pooled

        def coefficient(current: Any, target: Any, mask: Any | None = None) -> float:
            if mask is not None:
                current = current[mask]
                target = target[mask]
            before = torch.square(target).sum()
            after = torch.square(target - current).sum()
            return float(
                (100.0 * (before - after) / torch.clamp(before, min=EPS))
                .detach()
                .cpu()
            )

        certified = current_correction * common_gain[None, None]
        pooled_certified = current_correction * pooled_oof_gain[None, None]
        fold_gains = [
            coefficient(certified, current_residual, labels == value)
            for value in values
        ]
        pooled_fold_gains = [
            coefficient(pooled_certified, current_residual, labels == value)
            for value in values
        ]
        raw_fold_gains = [
            coefficient(current_correction, current_residual, labels == value)
            for value in values
        ]
        common_numpy = common_gain.detach().cpu().numpy().astype(np.float32)
        pooled_numpy = pooled_oof_gain.detach().cpu().numpy().astype(np.float32)
        fold_optima_numpy = fold_optima_tensor.detach().cpu().numpy()
        pooled_numpy64 = pooled.detach().cpu().numpy()
        minimum_numpy = minimum.detach().cpu().numpy()
        support_numpy = support.detach().cpu().numpy()
        zeros = np.zeros_like(pooled_numpy64)
        return {
            "common_gain": common_numpy,
            "pooled_oof_gain": pooled_numpy,
            "fold_gains": fold_gains,
            "overall": coefficient(certified, current_residual),
            "pooled_fold_gains": pooled_fold_gains,
            "pooled_overall": coefficient(pooled_certified, current_residual),
            "raw_fold_gains": raw_fold_gains,
            "raw_overall": coefficient(current_correction, current_residual),
            "certificate": {
                "active_modes": int(np.count_nonzero(support_numpy)),
                "active_mode_fraction": float(np.mean(support_numpy)),
                "fold_mode_optima": json.dumps(fold_optima_numpy.tolist()),
                "pooled_mode_optima": json.dumps(pooled_numpy64.tolist()),
                "minimum_mode_optima": json.dumps(minimum_numpy.tolist()),
                "recent_mode_gains": json.dumps(zeros.tolist()),
                "temporal_trend_mode_gains": json.dumps(zeros.tolist()),
            },
        }


def crossfit_candidates(
    innovation: np.ndarray,
    residual: np.ndarray,
    measurement: np.ndarray,
    environment: np.ndarray,
    shrinkages: tuple[float, ...],
    noise_scales: tuple[float, ...],
    empirical_couplings: tuple[float, ...],
    sample_scale_powers: tuple[float, ...],
    sample_scale_floor_quantile: float,
    gain_cap: float,
    selection_strategy: str = "worst_environment",
    covariance_sharing: str = "per_channel",
    compute_backend: str = "numpy",
    cuda_device: int = 0,
    cross_block_scales: tuple[float, ...] = (1.0,),
    observation_block_sizes: tuple[int, ...] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if selection_strategy not in (
        "worst_environment",
        "pooled_oof",
        "raw_worst_environment",
        "raw_pooled_oof",
    ):
        raise ValueError(selection_strategy)
    environments = tuple(sorted(int(value) for value in np.unique(environment)))
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, float], dict[str, float], np.ndarray, np.ndarray] | None = None
    for power in sample_scale_powers:
        for shrinkage in shrinkages:
            for noise_scale in noise_scales:
                for empirical_coupling in empirical_couplings:
                    for cross_block_scale in cross_block_scales:
                        correction = np.zeros_like(residual)
                        for current in environments:
                            held = environment == current
                            model = fit_residual_map(
                                innovation[~held],
                                residual[~held],
                                measurement,
                                shrinkage,
                                noise_scale,
                                empirical_coupling,
                                power,
                                sample_scale_floor_quantile,
                                covariance_sharing,
                                compute_backend,
                                cuda_device,
                                observation_block_sizes,
                                cross_block_scale,
                            )
                            correction[held] = predict_residual_map(
                                model, innovation[held]
                            )
                        if compute_backend == "torch_cuda":
                            statistics = cuda_candidate_statistics(
                                correction,
                                residual,
                                environment,
                                gain_cap,
                                cuda_device,
                            )
                            common_gain = statistics["common_gain"]
                            certificate = statistics["certificate"]
                            fold_gains = statistics["fold_gains"]
                            overall = statistics["overall"]
                            pooled_oof_gain = statistics["pooled_oof_gain"]
                            pooled_fold_gains = statistics["pooled_fold_gains"]
                            pooled_overall = statistics["pooled_overall"]
                            raw_fold_gains = statistics["raw_fold_gains"]
                            raw_overall = statistics["raw_overall"]
                        else:
                            gain_maps, certificate = coordinate_reliability(
                                correction, residual, environment, gain_cap
                            )
                            common_gain = gain_maps["common"]
                            certified = correction * common_gain[None, None]
                            fold_gains = [
                                coefficient_gain(
                                    residual[environment == current],
                                    certified[environment == current],
                                )
                                for current in environments
                            ]
                            overall = coefficient_gain(residual, certified)
                            pooled_oof_gain = np.clip(
                                coordinate_gain(correction, residual),
                                0.0,
                                gain_cap,
                            ).astype(np.float32)
                            pooled_certified = (
                                correction * pooled_oof_gain[None, None]
                            )
                            pooled_fold_gains = [
                                coefficient_gain(
                                    residual[environment == current],
                                    pooled_certified[environment == current],
                                )
                                for current in environments
                            ]
                            pooled_overall = coefficient_gain(
                                residual, pooled_certified
                            )
                            raw_fold_gains = [
                                coefficient_gain(
                                    residual[environment == current],
                                    correction[environment == current],
                                )
                                for current in environments
                            ]
                            raw_overall = coefficient_gain(residual, correction)
                        row = {
                            "sample_scale_power": power,
                            "covariance_shrinkage": shrinkage,
                            "noise_scale": noise_scale,
                            "empirical_coupling": empirical_coupling,
                            "cross_block_scale": cross_block_scale,
                            "oof_projected_mse_gain_pct": overall,
                            "worst_environment_gain_pct": float(min(fold_gains)),
                            "mean_environment_gain_pct": float(np.mean(fold_gains)),
                            "environment_gains_pct": json.dumps(fold_gains),
                            "pooled_oof_projected_mse_gain_pct": pooled_overall,
                            "pooled_oof_worst_environment_gain_pct": float(
                                min(pooled_fold_gains)
                            ),
                            "pooled_oof_mean_environment_gain_pct": float(
                                np.mean(pooled_fold_gains)
                            ),
                            "pooled_oof_environment_gains_pct": json.dumps(
                                pooled_fold_gains
                            ),
                            "raw_oof_projected_mse_gain_pct": raw_overall,
                            "raw_worst_environment_gain_pct": float(
                                min(raw_fold_gains)
                            ),
                            "raw_mean_environment_gain_pct": float(
                                np.mean(raw_fold_gains)
                            ),
                            "raw_environment_gains_pct": json.dumps(
                                raw_fold_gains
                            ),
                            "active_modes": certificate["active_modes"],
                            "active_mode_fraction": certificate["active_mode_fraction"],
                            "selection_strategy": selection_strategy,
                            "covariance_sharing": covariance_sharing,
                            "compute_backend": compute_backend,
                        }
                        rows.append(row)
                        parameters = {
                            "sample_scale_power": float(power),
                            "covariance_shrinkage": float(shrinkage),
                            "noise_scale": float(noise_scale),
                            "empirical_coupling": float(empirical_coupling),
                            "cross_block_scale": float(cross_block_scale),
                        }
                        if selection_strategy == "raw_pooled_oof":
                            key = (
                                float(raw_overall),
                                float(np.mean(raw_fold_gains)),
                                float(min(raw_fold_gains)),
                                -float(empirical_coupling),
                                -abs(float(noise_scale) - 1.0),
                                -abs(float(cross_block_scale) - 1.0),
                            )
                        elif selection_strategy == "raw_worst_environment":
                            key = (
                                float(min(raw_fold_gains)),
                                float(np.mean(raw_fold_gains)),
                                float(raw_overall),
                                -float(empirical_coupling),
                                -abs(float(noise_scale) - 1.0),
                                -abs(float(cross_block_scale) - 1.0),
                            )
                        elif selection_strategy == "pooled_oof":
                            key = (
                                float(pooled_overall),
                                float(np.mean(pooled_fold_gains)),
                                float(min(pooled_fold_gains)),
                                -float(empirical_coupling),
                                -abs(float(noise_scale) - 1.0),
                                -abs(float(cross_block_scale) - 1.0),
                            )
                        else:
                            key = (
                                float(min(fold_gains)),
                                float(np.mean(fold_gains)),
                                float(overall),
                                -float(empirical_coupling),
                                -abs(float(noise_scale) - 1.0),
                                -abs(float(cross_block_scale) - 1.0),
                            )
                        if best is None or key > best[0]:
                            best = (key, parameters, correction, common_gain)
    if best is None:
        raise RuntimeError("empty residual-map candidate grid")
    for row in rows:
        row["selected"] = all(
            float(row[key]) == float(best[1][key])
            for key in (
                "sample_scale_power",
                "covariance_shrinkage",
                "noise_scale",
                "empirical_coupling",
                "cross_block_scale",
            )
        )
    return best[1], best[2], best[3], rows
