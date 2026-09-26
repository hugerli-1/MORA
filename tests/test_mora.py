from dataclasses import replace
import inspect
import tempfile
import unittest
from pathlib import Path
import numpy as np

from mora import MORA, MORAConfig, ForecastBatch, CoarseFutureEstimation
from mora.data import legal_training_origins, windows
from mora.metrics import paired_metrics
from mora._numerics.coarse import segment_mean_operator, segment_means, coarse_features
from mora._numerics.basis import boundary_basis
from mora._numerics.factor_space import fit_factor_basis
from examples.quickstart import synthetic_problem


class GeometryTests(unittest.TestCase):
    def test_segment_means(self):
        a = segment_mean_operator(96, 24)
        np.testing.assert_allclose(a.sum(axis=1), 1)
        x = np.arange(192, dtype=np.float32).reshape(1, 96, 2)
        np.testing.assert_allclose(segment_means(x, a), x.reshape(1, 24, 4, 2).mean(axis=2).transpose(0, 2, 1))

    def test_boundary_and_orthogonality(self):
        for horizon in (12, 24, 48, 96, 192, 336, 720):
            config = MORAConfig(horizon=horizon)
            u = boundary_basis(horizon, config.rank, config.taper)
            np.testing.assert_allclose(u.astype(np.float64) @ u.T.astype(np.float64), np.eye(config.rank), atol=2e-6)
            self.assertLess(float(np.abs(u[:, 0]).max()), 1e-7)

    def test_compact_feature_dimension(self):
        x = np.random.RandomState(1).normal(size=(4, 96, 3)).astype(np.float32)
        self.assertEqual(coarse_features(x, np.arange(4), 96, "compact").shape, (4, 3, 78))

    def test_purged_training_windows(self):
        index = legal_training_origins(4000, 96, 96, (1500, 2000), 6)
        self.assertTrue(np.all((index + 192 <= 1500) | (index >= 2000)))

    def test_factor_rank_is_bounded_by_channels(self):
        x = np.random.RandomState(3).normal(size=(500, 3)).astype(np.float32)
        _, _, loading, _ = fit_factor_basis(x, np.ones(500, dtype=bool), 32)
        self.assertEqual(loading.shape, (3, 3))
        np.testing.assert_allclose(loading.T @ loading, np.eye(3), atol=1e-6)

    def test_invalid_configuration(self):
        for kwargs in ({"horizon": 3}, {"horizon": 96, "correction_rank": 96},
                       {"horizon": 96, "coarse_estimator": "unknown"},
                       {"horizon": 96, "coarse_estimator": "factorized", "use_periodic_template": True}):
            with self.assertRaises(ValueError):
                MORAConfig(**kwargs)

    def test_indices_do_not_wrap(self):
        with self.assertRaises(ValueError):
            windows(np.zeros((100, 2)), np.array([-1]), 96)
        with self.assertRaises(ValueError):
            windows(np.zeros((100, 2)), np.array([10]), 96)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problems, cls.models = {}, {}
        for kind in ("channel_wise", "factorized"):
            problem = synthetic_problem(kind)
            cls.problems[kind] = problem
            cls.models[kind] = MORA.fit(*problem[:4])

    def test_both_realizations_predict(self):
        for kind, model in self.models.items():
            test = self.problems[kind][-1]
            prediction = model.predict(test.history, test.backbone_prediction, test.sample_index)
            self.assertEqual(prediction.shape, test.y_true.shape)
            self.assertTrue(np.isfinite(prediction).all())
            np.testing.assert_allclose(prediction[:, 0], test.backbone_prediction[:, 0], atol=1e-7)

    def test_saved_state_replays_exactly(self):
        for kind, model in self.models.items():
            test = self.problems[kind][-1]
            expected = model.predict(test.history, test.backbone_prediction, test.sample_index)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.npz"
                model.save(path)
                loaded = MORA.load(path)
                actual = loaded.predict(test.history, test.backbone_prediction, test.sample_index)
                np.testing.assert_array_equal(actual, expected)
                with np.load(path, allow_pickle=False) as data:
                    self.assertFalse(any(data[k].dtype.hasobject for k in data.files))

    def test_no_future_target_in_prediction_interface(self):
        self.assertEqual(tuple(inspect.signature(MORA.predict).parameters),
                         ("self", "history", "backbone_prediction", "sample_index"))

    def test_query_independence(self):
        for kind, model in self.models.items():
            test = self.problems[kind][-1]
            batch = model.predict(test.history[:8], test.backbone_prediction[:8], test.sample_index[:8])
            one = model.predict(test.history[3:4], test.backbone_prediction[3:4], test.sample_index[3:4])
            np.testing.assert_allclose(one[0], batch[3], atol=1e-6)

    def test_future_targets_not_loaded_by_inference_reader(self):
        _, _, _, _, test = self.problems["channel_wise"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "queries.npz"
            np.savez(path, history=test.history, backbone_prediction=test.backbone_prediction,
                     sample_index=test.sample_index, y_true=np.asarray({"never": "read"}, dtype=object))
            batch = ForecastBatch.load(path, targets=False)
            self.assertIsNone(batch.y_true)
            with self.assertRaises(ValueError):
                ForecastBatch.load(path, targets=True)

    def test_reordered_indices_rejected_for_fitting(self):
        config, _, _, val, _ = self.problems["channel_wise"]
        bad = ForecastBatch(val.history, val.backbone_prediction, val.sample_index[::-1], val.y_true)
        with self.assertRaises(ValueError):
            bad.validate(config)

    def test_index_normalization_mismatch_rejected(self):
        config, series, oof, _, _ = synthetic_problem()
        oof.batch.history = oof.batch.history + 1
        with self.assertRaises(ValueError):
            oof.validate(config, series)

    def test_fold_exclusion(self):
        for kind in ("channel_wise", "factorized"):
            config, series, oof, _, test = self.problems[kind]
            interval = tuple(oof.held_raw_intervals[1])
            modified = series.copy()
            modified[slice(*interval)] += 1000
            first = CoarseFutureEstimation(config).fit(series, interval)
            second = CoarseFutureEstimation(config).fit(modified, interval)
            for a, b in zip(first.predict(test.history[:3], test.sample_index[:3]),
                            second.predict(test.history[:3], test.sample_index[:3])):
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)

    def test_state_symbols_match_paper_dimensions(self):
        for kind, model in self.models.items():
            config, _, _, validation, _ = self.problems[kind]
            c = validation.history.shape[2]
            adapter = model.residual_adapter
            self.assertEqual(adapter.U.shape, (config.horizon, config.rank))
            self.assertEqual(adapter.K_e.shape, (c, config.rank, config.intervals * len(model.coarse_future.extrapolation_names)))
            self.assertEqual(adapter.K_l.shape, (c, config.rank, config.intervals))
            self.assertEqual(adapter.G_e.shape, (config.rank,))
            self.assertEqual(adapter.G_l.shape, (config.rank,))
            self.assertEqual(adapter.S.shape, (c, config.rank))

    def test_intermediate_update_does_not_use_final_calibration(self):
        model = self.models["channel_wise"]
        test = self.problems["channel_wise"][-1]
        first, second = model.coarse_future.predict(test.history[:4], test.sample_index[:4])
        stage = model.residual_adapter
        before = stage.coefficients(test.backbone_prediction[:4], first, second, calibrated=False)
        old = stage.S
        try:
            stage.S = np.zeros_like(old)
            np.testing.assert_array_equal(before, stage.coefficients(test.backbone_prediction[:4], first, second, calibrated=False))
            np.testing.assert_array_equal(stage.predict(test.backbone_prediction[:4], first, second), test.backbone_prediction[:4])
        finally:
            stage.S = old

    def test_metrics_do_not_call_fitting(self):
        x = np.ones((4, 24, 3))
        metrics = paired_metrics(x, x * 0.5, np.zeros_like(x))
        self.assertEqual(metrics["mse_reduction_percent"], 75)
        self.assertEqual(metrics["mae_reduction_percent"], 50)

    def test_bad_forecast_shape_rejected(self):
        model = self.models["factorized"]
        test = self.problems["factorized"][-1]
        with self.assertRaises(ValueError):
            model.predict(test.history, test.backbone_prediction[:, :-1], test.sample_index)


if __name__ == "__main__":
    unittest.main()
