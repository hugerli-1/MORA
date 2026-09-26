"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from .projection import reconstruct
import itertools
import numpy as np

def band_trajectory_directions(
    proposal: np.ndarray,
    basis: np.ndarray,
    bands: int = 4,
) -> np.ndarray:
    rank = proposal.shape[2]
    output = []
    for band in range(bands):
        left = band * rank // bands
        right = (band + 1) * rank // bands
        current = np.zeros_like(proposal)
        current[:, :, left:right] = proposal[:, :, left:right]
        output.append(reconstruct(current, basis))
    return np.stack(output, axis=-1).astype(np.float32, copy=False)


def expand_band_coefficients(
    coefficient: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Expand channel-by-band gains to channel-by-mode gains."""

    channels, bands = coefficient.shape
    output = np.empty((channels, rank), dtype=np.float32)
    for band in range(bands):
        left = band * rank // bands
        right = (band + 1) * rank // bands
        output[:, left:right] = coefficient[:, band, None]
    return output


def solve_box_quadratic(
    gram: np.ndarray,
    rhs: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Solve a tiny box-constrained PSD quadratic by active-set enumeration."""

    dimensions = len(rhs)
    best = np.full(dimensions, lower, dtype=np.float64)
    best_objective = float(best @ gram @ best - 2.0 * best @ rhs)
    # -1 is lower bound, 0 is free, and +1 is upper bound.
    for state in itertools.product((-1, 0, 1), repeat=dimensions):
        candidate = np.zeros(dimensions, dtype=np.float64)
        fixed = [index for index, value in enumerate(state) if value != 0]
        free = [index for index, value in enumerate(state) if value == 0]
        for index in fixed:
            candidate[index] = lower if state[index] < 0 else upper
        if free:
            free_index = np.asarray(free, dtype=np.int64)
            adjusted = rhs[free_index].copy()
            if fixed:
                fixed_index = np.asarray(fixed, dtype=np.int64)
                adjusted -= gram[np.ix_(free_index, fixed_index)] @ candidate[
                    fixed_index
                ]
            candidate[free_index] = (
                np.linalg.pinv(
                    gram[np.ix_(free_index, free_index)], rcond=1e-9
                )
                @ adjusted
            )
            if np.any(candidate[free_index] < lower - 1e-9) or np.any(
                candidate[free_index] > upper + 1e-9
            ):
                continue
        candidate = np.clip(candidate, lower, upper)
        objective = float(candidate @ gram @ candidate - 2.0 * candidate @ rhs)
        if objective < best_objective:
            best = candidate
            best_objective = objective
    return best


def solve_channel_box(
    gram: np.ndarray, rhs: np.ndarray, cap: float
) -> np.ndarray:
    return np.stack(
        [
            solve_box_quadratic(gram[channel], rhs[channel], 0.0, cap)
            for channel in range(len(rhs))
        ]
    )
