"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

import numpy as np

def reconstruct(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("qh,ncq->nhc", basis, coefficients, optimize=True).astype(np.float32)
