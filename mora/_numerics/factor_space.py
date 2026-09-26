"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

import numpy as np

def fit_factor_basis(
    values: np.ndarray,
    fit_mask: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = values[fit_mask].astype(np.float64)
    location = selected.mean(axis=0)
    scale = np.maximum(selected.std(axis=0), 1e-5)
    standardized = (selected - location[None]) / scale[None]
    covariance = standardized.T @ standardized / max(len(standardized), 1)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    realized = min(max(int(rank), 1), values.shape[1])
    order = np.argsort(eigenvalue)[-realized:][::-1]
    loadings = eigenvector[:, order]
    explained = np.maximum(eigenvalue[order], 0.0)
    return (
        location.astype(np.float32),
        scale.astype(np.float32),
        loadings.astype(np.float32),
        explained.astype(np.float32),
    )


def project_factors(values: np.ndarray, location: np.ndarray, scale: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    return np.einsum(
        "...c,cr->...r",
        (values.astype(np.float64) - location.astype(np.float64))
        / scale.astype(np.float64),
        loadings.astype(np.float64),
        optimize=True,
    ).astype(np.float32)


def reconstruct_coarse_states(
    factor_moments: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    loadings: np.ndarray,
) -> np.ndarray:
    standardized = np.einsum(
        "nrq,cr->ncq",
        factor_moments.astype(np.float64),
        loadings.astype(np.float64),
        optimize=True,
    )
    return (
        standardized * scale.astype(np.float64)[None, :, None]
        + location.astype(np.float64)[None, :, None]
    ).astype(np.float32)
