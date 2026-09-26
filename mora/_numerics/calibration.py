"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .calibration_solver import band_trajectory_directions
from .calibration_solver import solve_channel_box
import numpy as np

def fit_multi_environment_balanced(
    environments: list[tuple[np.ndarray, np.ndarray]],
    basis: np.ndarray,
    initial: np.ndarray,
    cap: float,
    iterations: int,
    batch_size: int,
    mse_weight: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Fit one channel-band map under equal normalized environment risks."""

    coefficient = initial.astype(np.float64, copy=True)
    channels, bands = coefficient.shape
    environment_stats = []
    for error, _ in environments:
        current = error.astype(np.float64, copy=False)
        environment_stats.append(
            (
                max(float(np.square(current).mean()), 1e-12),
                max(float(np.abs(current).mean()), 1e-12),
                max(int(current.size), 1),
            )
        )
    epsilon = max(
        1e-3 * float(np.mean([item[1] for item in environment_stats])),
        1e-7,
    )
    trace: list[dict[str, float]] = []
    for iteration in range(iterations):
        gram = np.zeros((channels, bands, bands), dtype=np.float64)
        rhs = np.zeros((channels, bands), dtype=np.float64)
        objective = 0.0
        for (error, proposal), (base_mse, base_mae, points) in zip(
            environments, environment_stats
        ):
            environment_sse = 0.0
            environment_sae = 0.0
            for start in range(0, len(error), batch_size):
                stop = min(start + batch_size, len(error))
                current_error = error[start:stop].astype(np.float64, copy=False)
                directions = band_trajectory_directions(
                    proposal[start:stop], basis, bands
                ).astype(np.float64, copy=False)
                residual = current_error + np.einsum(
                    "nhcb,cb->nhc", directions, coefficient, optimize=True
                )
                smooth_abs = np.sqrt(np.square(residual) + epsilon * epsilon)
                weight = (
                    2.0 * float(mse_weight) / base_mse
                    + (1.0 - float(mse_weight)) / (base_mae * smooth_abs)
                ) / points
                gram += np.einsum(
                    "nhc,nhcb,nhcd->cbd",
                    weight,
                    directions,
                    directions,
                    optimize=True,
                )
                rhs -= np.einsum(
                    "nhc,nhcb,nhc->cb",
                    weight,
                    directions,
                    current_error,
                    optimize=True,
                )
                environment_sse += float(np.square(residual).sum())
                environment_sae += float(np.abs(residual).sum())
            objective += (
                float(mse_weight) * environment_sse / points / base_mse
                + (1.0 - float(mse_weight))
                * environment_sae
                / points
                / base_mae
            )
        updated = solve_channel_box(gram, rhs, cap)
        change = float(np.max(np.abs(updated - coefficient)))
        trace.append(
            {
                "iteration": float(iteration),
                "mean_normalized_risk": objective / len(environments),
                "maximum_change": change,
                "mse_weight": float(mse_weight),
            }
        )
        coefficient = updated
        if change < 1e-5:
            break
    return coefficient.astype(np.float32), trace
