"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .calibration import fit_multi_environment_balanced
from .calibration_solver import band_trajectory_directions
from .calibration_solver import expand_band_coefficients
from .calibration_solver import solve_channel_box
from .projection import reconstruct
from .reliability import coordinate_gain
from .residual_mapping import ResidualMap
from .residual_mapping import crossfit_candidates
from .residual_mapping import fit_residual_map
from dataclasses import dataclass
from typing import Any
import numpy as np

EPS = 1e-12


@dataclass
class FittedStage:
    names: tuple[str, ...]
    parameters: dict[str, float]
    oof_raw: np.ndarray
    common_gain: np.ndarray
    model: ResidualMap
    candidate_rows: list[dict[str, Any]]


def stack_coarse_operator(base: np.ndarray, count: int) -> np.ndarray:
    return np.concatenate(tuple(base for _ in range(count)), axis=0)


def coarse_discrepancy(
    stack: np.ndarray, anchor_moment: np.ndarray
) -> np.ndarray:
    return (stack - anchor_moment[:, :, None, :]).reshape(
        stack.shape[0], stack.shape[1], -1
    )


def measured_correction(
    correction: np.ndarray, measurement: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "ncr,qr->ncq",
        correction.astype(np.float64),
        measurement.astype(np.float64),
        optimize=True,
    ).astype(np.float32)


def fit_stage(
    names: tuple[str, ...],
    innovation: np.ndarray,
    residual: np.ndarray,
    measurement: np.ndarray,
    labels: np.ndarray,
    shrinkages: tuple[float, ...],
    noise_scales: tuple[float, ...],
    empirical_couplings: tuple[float, ...],
    scale_powers: tuple[float, ...],
    cap: float,
    selection_strategy: str = "worst_environment",
    covariance_sharing: str = "per_channel",
    compute_backend: str = "numpy",
    cuda_device: int = 0,
    cross_block_scales: tuple[float, ...] = (1.0,),
    observation_block_sizes: tuple[int, ...] | None = None,
) -> FittedStage:
    parameters, oof_raw, common_gain, candidates = crossfit_candidates(
        innovation,
        residual,
        measurement,
        labels,
        shrinkages,
        noise_scales,
        empirical_couplings,
        scale_powers,
        0.1,
        cap,
        selection_strategy,
        covariance_sharing,
        compute_backend,
        cuda_device,
        cross_block_scales,
        observation_block_sizes,
    )
    model = fit_residual_map(
        innovation,
        residual,
        measurement,
        parameters["covariance_shrinkage"],
        parameters["noise_scale"],
        parameters["empirical_coupling"],
        parameters["sample_scale_power"],
        0.1,
        covariance_sharing,
        compute_backend,
        cuda_device,
        observation_block_sizes,
        parameters["cross_block_scale"],
    )
    return FittedStage(names, parameters, oof_raw, common_gain, model, candidates)


def stage_gain(
    branch: FittedStage,
    target: np.ndarray,
    reliability: str,
    cap: float,
) -> np.ndarray:
    if reliability == "raw":
        return np.ones(branch.oof_raw.shape[1:], dtype=np.float32)
    if reliability == "nonexpansive":
        return np.clip(
            coordinate_gain(branch.oof_raw, target), 0.0, 1.0
        ).astype(np.float32)
    if reliability == "common":
        return branch.common_gain.astype(np.float32)
    if reliability == "pooled":
        return np.clip(
            coordinate_gain(branch.oof_raw, target), 0.0, cap
        ).astype(np.float32)
    raise ValueError(reliability)


def chunk_bounds(length: int, chunk_size: int):
    if chunk_size < 1:
        raise ValueError(f"chunk size must be positive, received {chunk_size}")
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def use_streaming(channels: int, channel_chunk_size: int | None) -> bool:
    if channel_chunk_size is None:
        return False
    if channel_chunk_size < 1:
        raise ValueError(
            "trajectory channel chunk size must be positive, received "
            f"{channel_chunk_size}"
        )
    return channels > channel_chunk_size


def trajectory_chunk(
    values: np.ndarray,
    sample_index: np.ndarray,
    channel_start: int,
    channel_stop: int,
) -> np.ndarray:
    """Read only the requested N x H x C block, including from memmaps."""

    current = np.asarray(sample_index, dtype=np.int64)
    if len(current) == 0:
        return np.empty(
            (0, values.shape[1], channel_stop - channel_start),
            dtype=values.dtype,
        )
    if len(current) == 1 or np.all(np.diff(current) == 1):
        return np.asarray(
            values[
                slice(int(current[0]), int(current[-1]) + 1),
                :,
                channel_start:channel_stop,
            ]
        )
    return np.asarray(
        values[
            np.ix_(
                current,
                np.arange(values.shape[1], dtype=np.int64),
                np.arange(channel_start, channel_stop, dtype=np.int64),
            )
        ]
    )


def coefficient_chunk(
    values: np.ndarray,
    sample_index: np.ndarray,
    channel_start: int,
    channel_stop: int,
) -> np.ndarray:
    """Read only the requested N x C x R coefficient block."""

    current = np.asarray(sample_index, dtype=np.int64)
    if len(current) == 0:
        return np.empty(
            (0, channel_stop - channel_start, values.shape[2]),
            dtype=values.dtype,
        )
    if len(current) == 1 or np.all(np.diff(current) == 1):
        return np.asarray(
            values[
                slice(int(current[0]), int(current[-1]) + 1),
                channel_start:channel_stop,
                :,
            ]
        )
    return np.asarray(
        values[
            np.ix_(
                current,
                np.arange(channel_start, channel_stop, dtype=np.int64),
                np.arange(values.shape[2], dtype=np.int64),
            )
        ]
    )


def fit_final_calibration(
    correction: np.ndarray,
    error: np.ndarray | None,
    basis: np.ndarray,
    labels: np.ndarray,
    cap: float,
    mse_weight: float,
    sample_chunk_size: int = 128,
    channel_chunk_size: int | None = None,
    *,
    anchor: np.ndarray | None = None,
    target: np.ndarray | None = None,
    error_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    sources = sum(
        (
            error is not None,
            anchor is not None or target is not None,
            error_coefficients is not None,
        )
    )
    if sources != 1 or (anchor is None) != (target is None):
        raise ValueError(
            "provide exactly one balanced-calibration error source: error, "
            "anchor+target, or error_coefficients"
        )
    samples, channels, rank = correction.shape
    if len(labels) != samples or basis.shape[0] != rank:
        raise ValueError((correction.shape, basis.shape, labels.shape))

    if error is not None:
        if error.shape != (samples, basis.shape[1], channels):
            raise ValueError((error.shape, correction.shape, basis.shape))
    elif anchor is not None and target is not None:
        if anchor.shape != target.shape or anchor.shape != (
            samples,
            basis.shape[1],
            channels,
        ):
            raise ValueError((anchor.shape, target.shape, correction.shape))
    elif error_coefficients is not None:
        if error_coefficients.shape != correction.shape:
            raise ValueError((error_coefficients.shape, correction.shape))

    def read_error(
        sample_index: np.ndarray, channel_start: int, channel_stop: int
    ) -> np.ndarray:
        if error is not None:
            current = trajectory_chunk(
                error, sample_index, channel_start, channel_stop
            )
        elif anchor is not None and target is not None:
            current = trajectory_chunk(
                anchor, sample_index, channel_start, channel_stop
            ) - trajectory_chunk(
                target, sample_index, channel_start, channel_stop
            )
        else:
            current = reconstruct(
                coefficient_chunk(
                    error_coefficients,
                    sample_index,
                    channel_start,
                    channel_stop,
                ),
                basis,
            )
        return current.astype(np.float64, copy=False)

    if not use_streaming(channels, channel_chunk_size):
        if error is not None:
            dense_error = error
        elif anchor is not None and target is not None:
            dense_error = anchor - target
        else:
            dense_error = reconstruct(error_coefficients, basis)
        environments = []
        for label in np.unique(labels):
            selected = labels == label
            environments.append((dense_error[selected], correction[selected]))
        initial = np.ones((channels, 4), dtype=np.float32)
        band, _ = fit_multi_environment_balanced(
            environments,
            basis,
            initial,
            cap,
            8,
            sample_chunk_size,
            mse_weight,
        )
        return expand_band_coefficients(band, rank)

    environment_indices = tuple(
        np.flatnonzero(labels == label).astype(np.int64, copy=False)
        for label in np.unique(labels)
    )
    environment_stats: list[tuple[float, float, int]] = []
    for indices in environment_indices:
        sum_squared = 0.0
        sum_absolute = 0.0
        points = max(len(indices) * basis.shape[1] * channels, 1)
        for sample_start, sample_stop in chunk_bounds(
            len(indices), sample_chunk_size
        ):
            sample_index = indices[sample_start:sample_stop]
            for channel_start, channel_stop in chunk_bounds(
                channels, int(channel_chunk_size)
            ):
                current = read_error(
                    sample_index, channel_start, channel_stop
                )
                sum_squared += float(np.square(current).sum())
                sum_absolute += float(np.abs(current).sum())
        environment_stats.append(
            (
                max(sum_squared / points, EPS),
                max(sum_absolute / points, EPS),
                points,
            )
        )

    coefficient = np.ones((channels, 4), dtype=np.float64)
    epsilon = max(
        1e-3 * float(np.mean([item[1] for item in environment_stats])),
        1e-7,
    )
    for _ in range(8):
        gram = np.zeros((channels, 4, 4), dtype=np.float64)
        rhs = np.zeros((channels, 4), dtype=np.float64)
        for indices, (base_mse, base_mae, points) in zip(
            environment_indices, environment_stats
        ):
            for sample_start, sample_stop in chunk_bounds(
                len(indices), sample_chunk_size
            ):
                sample_index = indices[sample_start:sample_stop]
                for channel_start, channel_stop in chunk_bounds(
                    channels, int(channel_chunk_size)
                ):
                    current_error = read_error(
                        sample_index, channel_start, channel_stop
                    )
                    proposal = coefficient_chunk(
                        correction,
                        sample_index,
                        channel_start,
                        channel_stop,
                    )
                    directions = band_trajectory_directions(
                        proposal, basis, 4
                    ).astype(np.float64, copy=False)
                    residual = current_error + np.einsum(
                        "nhcb,cb->nhc",
                        directions,
                        coefficient[channel_start:channel_stop],
                        optimize=True,
                    )
                    smooth_abs = np.sqrt(
                        np.square(residual) + epsilon * epsilon
                    )
                    weight = (
                        2.0 * float(mse_weight) / base_mse
                        + (1.0 - float(mse_weight))
                        / (base_mae * smooth_abs)
                    ) / points
                    gram[channel_start:channel_stop] += np.einsum(
                        "nhc,nhcb,nhcd->cbd",
                        weight,
                        directions,
                        directions,
                        optimize=True,
                    )
                    rhs[channel_start:channel_stop] -= np.einsum(
                        "nhc,nhcb,nhc->cb",
                        weight,
                        directions,
                        current_error,
                        optimize=True,
                    )
        updated = solve_channel_box(gram, rhs, cap)
        change = float(np.max(np.abs(updated - coefficient)))
        coefficient = updated
        if change < 1e-5:
            break
    return expand_band_coefficients(coefficient.astype(np.float32), rank)


def contiguous_labels(length: int, count: int = 4) -> np.ndarray:
    labels = np.empty(length, dtype=np.int64)
    for label, indices in enumerate(np.array_split(np.arange(length), count)):
        labels[indices] = label
    return labels
