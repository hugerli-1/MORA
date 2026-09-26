"""Frozen numerical routines; see docs/METHOD.md for notation."""
from __future__ import annotations

import numpy as np

def select_template_periods(
    values: np.ndarray,
    train_end: int,
    daily_period: int,
    count: int = 8,
    train_fit_mask: np.ndarray | None = None,
    known_harmonics: int = 1,
) -> np.ndarray:
    """Discover a separated long-range period dictionary from train only."""

    train = values[:train_end].astype(np.float64)
    if train_fit_mask is None:
        fit_mask = np.ones(train_end, dtype=bool)
    else:
        fit_mask = np.asarray(train_fit_mask, dtype=bool)
        if fit_mask.shape != (train_end,):
            raise ValueError((fit_mask.shape, train_end))
        if int(fit_mask.sum()) < 16:
            raise ValueError("too few legal rows for spectral period discovery")
    time = np.linspace(-1.0, 1.0, train_end, dtype=np.float64)
    design = np.stack((np.ones_like(time), time), axis=1)
    coefficient = np.linalg.lstsq(design[fit_mask], train[fit_mask], rcond=None)[0]
    residual = train - design @ coefficient
    scale = np.maximum(residual[fit_mask].std(axis=0, keepdims=True), 1e-6)
    residual /= scale
    # Zero filling is exactly the masked DFT on the regular Fourier grid.  It
    # keeps blocked-OOF period discovery leakage-free without an O(NF) masked
    # sinusoid scan.
    residual[~fit_mask] = 0.0
    spectrum = np.fft.rfft(residual, axis=0)
    power = np.median(np.log1p(np.square(np.abs(spectrum))), axis=1)
    frequencies = np.fft.rfftfreq(train_end)
    valid = (frequencies >= 2.0 / float(train_end)) & (
        frequencies <= 1.0 / 8.0
    )
    local_peak = np.zeros_like(valid)
    local_peak[1:-1] = (power[1:-1] >= power[:-2]) & (
        power[1:-1] >= power[2:]
    )
    candidates = np.flatnonzero(valid & local_peak)
    ranked = candidates[np.argsort(power[candidates])[::-1]]

    selected: list[float] = []

    def append_frequency(frequency: float) -> None:
        if not (frequencies[1] <= frequency <= frequencies[-1]):
            return
        if any(
            abs(np.log(frequency / existing)) < np.log(1.04)
            for existing in selected
        ):
            return
        selected.append(float(frequency))

    # Calendar sampling supplies only the base period, not future values.  A
    # compact harmonic scaffold lets non-sinusoidal daily/weekly shapes enter
    # the same train-only atlas while the remaining slots stay data-driven.
    for harmonic in range(1, max(int(known_harmonics), 1) + 1):
        for known_period in (
            float(daily_period) / float(harmonic),
            7.0 * float(daily_period) / float(harmonic),
        ):
            if known_period >= 8.0:
                append_frequency(1.0 / known_period)
    for current in ranked:
        frequency = float(frequencies[current])
        if any(
            abs(np.log(frequency / existing)) < np.log(1.08)
            for existing in selected
        ):
            continue
        append_frequency(frequency)
        if len(selected) >= int(count):
            break
    if not selected:
        selected = [1.0 / float(daily_period)]
    return np.asarray(
        sorted((1.0 / value for value in selected[: int(count)])),
        dtype=np.float64,
    )
