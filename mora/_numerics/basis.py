"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

from scipy.fft import idct
import numpy as np

def boundary_basis(
    horizon: int, rank: int = 16, taper_length: int = 13
) -> np.ndarray:
    rank = min(int(rank), int(horizon))
    taper_length = min(int(taper_length), int(horizon))
    if rank < 1 or taper_length < 2:
        raise ValueError((horizon, rank, taper_length))
    raw = idct(
        np.eye(horizon, dtype=np.float64)[:rank],
        type=2,
        axis=1,
        norm="ortho",
    )
    taper = np.ones(horizon, dtype=np.float64)
    taper[:taper_length] = (
        np.sin(
            0.5
            * np.pi
            * np.arange(taper_length, dtype=np.float64)
            / (taper_length - 1)
        )
        ** 2
    )
    raw *= taper[None]
    orthogonal, _ = np.linalg.qr(raw.T, mode="reduced")
    result = orthogonal.T.astype(np.float32)
    if float(np.max(np.abs(result[:, 0]))) > 1e-7:
        raise RuntimeError("boundary basis does not preserve the first forecast lead")
    return result
