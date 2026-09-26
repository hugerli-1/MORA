"""Exercise both MORA components with synthetic, target-free baseline forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from mora import MORA, MORAConfig, ForecastBatch, OOFForecasts
from mora.data import blocked_oof_geometry, windows
from mora.metrics import paired_metrics


def synthetic_problem(estimator="channel_wise"):
    rng = np.random.RandomState(7)
    time = np.arange(5200, dtype=np.float64)
    channels = 3
    values = np.stack([np.sin(2 * np.pi * time / 24 + c * 0.3)
                       + 0.4 * np.cos(2 * np.pi * time / 168 + c)
                       + 0.08 * rng.normal(size=len(time)) for c in range(channels)], axis=1)
    train_end, validation_end = 3400, 4200
    location, scale = values[:train_end].mean(axis=0), values[:train_end].std(axis=0)
    series = ((values - location) / scale).astype(np.float32)
    config = MORAConfig(horizon=24, history_length=96, base_period=24,
        coarse_estimator=estimator, use_periodic_template=estimator == "channel_wise",
        factor_rank=3, ridge_alphas=(10.0,), covariance_shrinkages=(1.0,),
        noise_scales=(1.0,), empirical_couplings=(0.0,), sample_scale_powers=(0.0,))

    def batch(index):
        x = windows(series, index, config.history_length)
        # This deterministic extrapolator has no fitted parameters and sees X only.
        baseline = 0.7 * x[:, -24:][:, np.arange(config.horizon) % 24]
        y = windows(series, index, config.horizon, config.history_length)
        return ForecastBatch(x, baseline, index, y)

    origins, fold_id, intervals = blocked_oof_geometry(train_end, config.history_length, config.horizon)
    oof = OOFForecasts(batch(origins), fold_id, intervals)
    validation = batch(np.arange(train_end - config.history_length,
                                 validation_end - config.history_length - config.horizon + 1, 6))
    test = batch(np.arange(validation_end - config.history_length,
                           len(series) - config.history_length - config.horizon + 1, 6))
    return config, series[:train_end], oof, validation, test


def save_batch(path, batch, **extra):
    np.savez_compressed(path, history=batch.history, backbone_prediction=batch.backbone_prediction,
                        sample_index=batch.sample_index, y_true=batch.y_true, **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("channel_wise", "factorized"), default="channel_wise")
    parser.add_argument("--output", default="runs/quickstart")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config, train, oof, validation, test = synthetic_problem(args.estimator)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    np.save(output / "training_series.npy", train)
    save_batch(output / "oof.npz", oof.batch, fold_id=oof.fold_id, held_raw_intervals=oof.held_raw_intervals)
    save_batch(output / "validation.npz", validation)
    save_batch(output / "test.npz", test)
    model = MORA.fit(config, train, oof, validation, progress=lambda s: print(s, flush=True))
    prediction = model.predict(test.history, test.backbone_prediction, test.sample_index)
    model.save(output / "mora.npz")
    replay = MORA.load(output / "mora.npz").predict(test.history, test.backbone_prediction, test.sample_index)
    np.testing.assert_array_equal(prediction, replay)
    np.testing.assert_allclose(prediction[:, 0], test.backbone_prediction[:, 0], atol=1e-7)
    np.savez_compressed(output / "predictions.npz", prediction=prediction, sample_index=test.sample_index)
    report = {"scope": "synthetic smoke test, not a forecasting benchmark",
              "coarse_estimator": args.estimator, "save_load_exact": True,
              **paired_metrics(test.backbone_prediction, prediction, test.y_true)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
