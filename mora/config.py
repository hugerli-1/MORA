"""Configuration for Model-Output Residual Adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MORAConfig:
    horizon: int
    history_length: int = 96
    base_period: int = 96
    coarse_estimator: str = "channel_wise"
    use_periodic_template: bool = False
    feature_set: str = "compact"
    correction_rank: int | None = None
    boundary_taper: int | None = None
    max_coarse_intervals: int = 64
    min_interval_length: int = 4
    factor_rank: int = 32
    factor_period: int | None = None
    template_period_count: int = 8
    template_half_life: float = 16.0
    estimator_stride: int | None = None
    rolling_blocks: int = 4
    ridge_alphas: tuple[float, ...] = (1, 10, 100, 1000, 10000)
    covariance_shrinkages: tuple[float, ...] = (0.5, 1.0)
    noise_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    empirical_couplings: tuple[float, ...] = (0, 0.25, 0.5, 1.0)
    sample_scale_powers: tuple[float, ...] = (0, 0.5)
    coefficient_cap: float = 1.5
    calibration_mse_weight: float = 0.1
    compute_backend: str = "numpy"
    cuda_device: int = 0
    batch_size: int = 128
    channel_chunk_size: int = 32

    def __post_init__(self):
        if self.horizon < 12 or self.history_length < 24 or self.base_period < 1:
            raise ValueError("Require H >= 12, L >= 24, and a positive base period")
        if self.coarse_estimator not in {"channel_wise", "factorized"}:
            raise ValueError("coarse_estimator must be channel_wise or factorized")
        if self.feature_set not in {"compact", "full", "compact_phase_interaction", "full_phase_interaction"}:
            raise ValueError("Unsupported history feature set")
        if self.min_interval_length < 1 or self.max_coarse_intervals < 1:
            raise ValueError("Coarse interval dimensions must be positive")
        if self.template_period_count < 1 or self.template_half_life <= 0:
            raise ValueError("Periodic-template dimensions must be positive")
        if self.coarse_estimator == "factorized" and self.use_periodic_template:
            raise ValueError("The paper's factorized configuration uses two extrapolations")
        if not 4 <= self.rank < self.horizon or not 2 <= self.taper <= self.horizon:
            raise ValueError("Invalid correction rank or boundary taper")
        if not 1 <= self.intervals < self.horizon:
            raise ValueError("Invalid coarse interval count")
        if self.stride < 1 or self.factor_rank < 1 or self.phase_period < 1:
            raise ValueError("Stride, factor rank, and period must be positive")
        if not 2 <= self.rolling_blocks <= 4:
            raise ValueError("Use two to four training-internal rolling blocks")
        if not self.ridge_alphas or min(self.ridge_alphas) <= 0:
            raise ValueError("Ridge strengths must be positive")
        for values, low, high in ((self.covariance_shrinkages, 0, 1),
                                  (self.empirical_couplings, 0, 1),
                                  (self.sample_scale_powers, 0, 1)):
            if not values or any(not low <= x <= high for x in values):
                raise ValueError("Invalid residual-map parameter grid")
        if not self.noise_scales or min(self.noise_scales) <= 0:
            raise ValueError("Noise scales must be positive")
        if self.compute_backend not in {"numpy", "torch_cuda"}:
            raise ValueError("Unknown numerical backend")
        if self.batch_size < 1 or self.channel_chunk_size < 1 or self.coefficient_cap <= 0:
            raise ValueError("Batch sizes and coefficient cap must be positive")
        if not 0 <= self.calibration_mse_weight <= 1:
            raise ValueError("Invalid calibration loss weight")

    @property
    def rank(self):
        return self.correction_rank if self.correction_rank is not None else {12: 8, 24: 16, 48: 32}.get(self.horizon, min(64, self.horizon - 1))

    @property
    def taper(self):
        return self.boundary_taper if self.boundary_taper is not None else {12: 2, 24: 3, 48: 6}.get(self.horizon, min(13, self.horizon))

    @property
    def intervals(self):
        return min(self.max_coarse_intervals, self.horizon // self.min_interval_length)

    @property
    def stride(self):
        return self.estimator_stride if self.estimator_stride is not None else (1 if self.coarse_estimator == "factorized" else 6)

    @property
    def phase_period(self):
        return self.factor_period or 7 * self.base_period

    @property
    def mapping_grid(self):
        if self.coarse_estimator == "factorized":
            return ((1.0,), (1.0,), (0.0,), (0.0,))
        return (self.covariance_shrinkages, self.noise_scales,
                self.empirical_couplings, self.sample_scale_powers)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("ridge_alphas", "covariance_shrinkages", "noise_scales",
                    "empirical_couplings", "sample_scale_powers"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
