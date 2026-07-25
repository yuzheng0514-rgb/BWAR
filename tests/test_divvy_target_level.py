from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bwar.paper_jcgs.divvy_target_level import (
    TARGET_COLUMNS,
    evaluate_fixed_split,
    evaluate_target_panel,
    evaluate_local_split,
    holm_adjust,
    method_level_summary,
    moving_block_mean_bootstrap,
    paired_inference,
    target_loss_row,
    test_source_indices,
    validate_complete_panel,
    validation_source_indices,
)


class TargetPanelTests(unittest.TestCase):
    def test_test_sources_map_exactly_to_test_targets(self) -> None:
        sources = test_source_indices(181, 224, horizon=3)

        np.testing.assert_array_equal(sources, np.arange(178, 221))
        np.testing.assert_array_equal(sources + 3, np.arange(181, 224))

    def test_complete_panel_rejects_missing_method_target(self) -> None:
        panel = pd.DataFrame(
            {
                "origin": [0, 0, 0],
                "horizon": [3, 3, 3],
                "target_index": [181, 181, 182],
                "method": ["fixed_bwar", "cholesky", "fixed_bwar"],
                "raw_rmse": [1.0, 1.1, 0.9],
                "w2": [2.0, 2.1, 1.9],
            }
        )

        with self.assertRaisesRegex(ValueError, "complete paired panel"):
            validate_complete_panel(panel, ("fixed_bwar", "cholesky"))

    def test_complete_panel_sorts_a_finite_exact_panel(self) -> None:
        panel = pd.DataFrame(
            {
                "origin": [0, 0, 0, 0],
                "horizon": [3, 3, 3, 3],
                "target_index": [182, 181, 182, 181],
                "method": ["cholesky", "fixed_bwar", "fixed_bwar", "cholesky"],
                "raw_rmse": [1.2, 0.9, 1.0, 1.1],
                "w2": [2.2, 1.9, 2.0, 2.1],
            }
        )

        result = validate_complete_panel(panel, ("fixed_bwar", "cholesky"))

        self.assertEqual(len(result), 4)
        self.assertEqual(result["target_index"].tolist(), [181, 181, 182, 182])

    def test_target_loss_row_reports_raw_w2_and_positive_eigenvalue(self) -> None:
        row = target_loss_row(
            origin=0,
            fit_end=10,
            val_end=14,
            test_end=18,
            horizon=2,
            source_index=12,
            method="fixed_bwar",
            window_length=10,
            ridge=0.1,
            pred_mean=np.array([0.0, 0.0]),
            pred_cov=np.eye(2),
            target_mean=np.array([1.0, 0.0]),
            target_cov=np.eye(2),
            domain_profile={"kind": "mean_rmse"},
        )

        self.assertEqual(tuple(row), TARGET_COLUMNS)
        self.assertAlmostEqual(float(row["raw_rmse"]), np.sqrt(0.5))
        self.assertAlmostEqual(float(row["w2"]), 1.0)
        self.assertAlmostEqual(float(row["min_pred_eig"]), 1.0)
        self.assertEqual(int(row["target_index"]), 14)

    def test_target_loss_row_rejects_non_spd_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite SPD"):
            target_loss_row(
                origin=0,
                fit_end=10,
                val_end=14,
                test_end=18,
                horizon=2,
                source_index=12,
                method="fixed_bwar",
                window_length=10,
                ridge=0.1,
                pred_mean=np.zeros(2),
                pred_cov=np.array([[1.0, 0.0], [0.0, -1.0]]),
                target_mean=np.zeros(2),
                target_cov=np.eye(2),
                domain_profile={"kind": "mean_rmse"},
            )


class DependenceAwareInferenceTests(unittest.TestCase):
    def test_moving_block_bootstrap_is_deterministic(self) -> None:
        differences = {0: np.arange(7.0), 1: np.arange(4.0) + 20.0}

        first = moving_block_mean_bootstrap(
            differences, block_length=3, replicates=200, seed=71
        )
        second = moving_block_mean_bootstrap(
            differences, block_length=3, replicates=200, seed=71
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (200,))
        self.assertTrue(np.isfinite(first).all())

    def test_holm_adjustment_preserves_input_order(self) -> None:
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))

        np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.06]))

    def test_method_summary_uses_all_targets_and_block_bootstrap_intervals(self) -> None:
        rows = []
        for origin, targets in {0: (10, 11), 1: (20, 21)}.items():
            for target in targets:
                for method, raw, w2 in (
                    ("fixed_bwar", 1.0, 2.0),
                    ("local_bwar", 0.8, 1.7),
                ):
                    rows.append(
                        {
                            "origin": origin,
                            "horizon": 3,
                            "target_index": target,
                            "method": method,
                            "raw_rmse": raw,
                            "w2": w2,
                        }
                    )

        summary = method_level_summary(
            pd.DataFrame(rows),
            methods=("fixed_bwar", "local_bwar"),
            block_length=2,
            replicates=200,
            seed=71,
        )

        self.assertEqual(len(summary), 4)
        self.assertEqual(set(summary["metric"]), {"raw_rmse", "w2"})
        self.assertTrue((summary["n_targets"] == 4).all())
        self.assertTrue((summary["n_origins"] == 2).all())
        local_raw = summary.loc[
            summary["method"].eq("local_bwar")
            & summary["metric"].eq("raw_rmse")
        ].iloc[0]
        self.assertAlmostEqual(float(local_raw["mean"]), 0.8)
        self.assertAlmostEqual(float(local_raw["ci_low"]), 0.8)
        self.assertAlmostEqual(float(local_raw["ci_high"]), 0.8)

    def test_paired_inference_reports_fixed_and_local_differences(self) -> None:
        rows = []
        for origin, targets in {0: (10, 11), 1: (20, 21)}.items():
            for target in targets:
                for method, raw, w2 in (
                    ("fixed_bwar", 1.0, 2.0),
                    ("local_bwar", 0.8, 1.7),
                    ("cholesky", 1.4, 2.3),
                ):
                    rows.append(
                        {
                            "origin": origin,
                            "horizon": 3,
                            "target_index": target,
                            "method": method,
                            "raw_rmse": raw,
                            "w2": w2,
                        }
                    )
        panel = pd.DataFrame(rows)

        result = paired_inference(
            panel,
            targets=("fixed_bwar", "local_bwar"),
            comparators=("cholesky",),
            horizon=3,
            block_length=2,
            sensitivity_block_length=3,
            replicates=200,
            seed=17,
        )

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["metric"]), {"raw_rmse", "w2"})
        self.assertEqual(set(result["target_method"]), {"fixed_bwar", "local_bwar"})
        self.assertTrue((result["mean_difference"] < 0.0).all())
        self.assertTrue((result["n_targets"] == 4).all())
        self.assertTrue((result["n_origins"] == 2).all())
        self.assertTrue((result["p_holm"] >= result["p_value"]).all())


class LocalReferenceSplitTests(unittest.TestCase):
    def test_validation_sources_end_before_test_block(self) -> None:
        sources = validation_source_indices(
            127, 181, horizon=3, max_window_length=72
        )

        self.assertEqual(int(sources[0]), 124)
        self.assertEqual(int(sources[-1]), 177)
        self.assertTrue(np.all(sources + 3 < 181))

    def test_local_test_rows_use_validation_selected_frozen_setting(self) -> None:
        rng = np.random.default_rng(20260714)
        means = rng.normal(scale=0.1, size=(65, 2))
        covariances = np.asarray(
            [
                matrix @ matrix.T + 0.5 * np.eye(2)
                for matrix in rng.normal(size=(65, 2, 2))
            ]
        )

        test_rows, tuning, selected = evaluate_local_split(
            means,
            covariances,
            origin=0,
            fit_end=40,
            val_end=55,
            test_end=65,
            horizon=3,
            window_grid=(12, 18),
            ridge_grid=(0.1, 1.0),
            domain_profile={"kind": "mean_rmse"},
        )

        self.assertEqual(len(tuning), 4)
        self.assertEqual(test_rows["window_length"].nunique(), 1)
        self.assertEqual(test_rows["ridge"].nunique(), 1)
        self.assertEqual(int(test_rows["window_length"].iloc[0]), selected["window_length"])
        self.assertEqual(float(test_rows["ridge"].iloc[0]), selected["ridge"])
        self.assertTrue(
            np.array_equal(
                test_rows["source_index"].to_numpy() + 3,
                test_rows["target_index"].to_numpy(),
            )
        )
        self.assertTrue((test_rows["target_index"] >= 55).all())
        self.assertTrue((test_rows["target_index"] < 65).all())
        self.assertTrue((test_rows["min_pred_eig"] > 0.0).all())
        self.assertEqual(int(tuning["validation_target_stop"].max()), 55)


class FixedMethodSplitTests(unittest.TestCase):
    def test_fixed_methods_share_exact_test_targets(self) -> None:
        rng = np.random.default_rng(20260714)
        starts = np.arange(65, dtype=int) * 2
        raw_series = rng.normal(size=(int(starts[-1] + 6), 2))
        raw_windows = np.asarray([raw_series[start : start + 6] for start in starts])
        means = raw_windows.mean(axis=1)
        covariances = np.asarray(
            [np.cov(window, rowvar=False) + 0.1 * np.eye(2) for window in raw_windows]
        )

        rows = evaluate_fixed_split(
            means,
            covariances,
            raw_series=raw_series,
            raw_windows=raw_windows,
            window_starts=starts,
            window_size=6,
            seasonal_period_windows=1,
            origin=0,
            fit_end=40,
            val_end=55,
            test_end=65,
            horizon=3,
            domain_profile={"kind": "mean_rmse"},
            ridge_grid=(0.1, 1.0),
        )

        methods = {
            "persistence",
            "raw_var_window_ar",
            "seasonal_window_naive",
            "euclidean_gaussian_ar",
            "cholesky_gaussian_ar",
            "log_euclidean_gaussian_ar",
            "fixed_bwar",
        }
        self.assertEqual(set(rows["method"]), methods)
        self.assertEqual(len(rows), 10 * len(methods))
        for method in methods:
            method_targets = rows.loc[
                rows["method"].eq(method), "target_index"
            ].to_numpy()
            np.testing.assert_array_equal(method_targets, np.arange(55, 65))
        self.assertTrue((rows["min_pred_eig"] > 0.0).all())

    def test_target_panel_adds_local_on_the_same_targets(self) -> None:
        rng = np.random.default_rng(8)
        starts = np.arange(65, dtype=int) * 2
        raw_series = rng.normal(size=(int(starts[-1] + 6), 2))
        raw_windows = np.asarray([raw_series[start : start + 6] for start in starts])
        means = raw_windows.mean(axis=1)
        covariances = np.asarray(
            [np.cov(window, rowvar=False) + 0.1 * np.eye(2) for window in raw_windows]
        )

        panel, tuning, selected = evaluate_target_panel(
            means,
            covariances,
            raw_series=raw_series,
            raw_windows=raw_windows,
            window_starts=starts,
            window_size=6,
            seasonal_period_windows=1,
            splits=((40, 55, 65),),
            horizons=(3,),
            domain_profile={"kind": "mean_rmse"},
            ridge_grid=(0.1, 1.0),
            local_window_grid=(12, 18),
        )

        self.assertEqual(panel["method"].nunique(), 8)
        self.assertEqual(len(panel), 10 * 8)
        self.assertEqual(len(tuning), 4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(set(panel["target_index"]), set(range(55, 65)))


if __name__ == "__main__":
    unittest.main()
