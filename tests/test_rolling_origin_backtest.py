from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.rolling_origin_backtest import make_rolling_origin_splits, run_rolling_origin_series, summarize_rolling


class RollingOriginBacktestTest(unittest.TestCase):
    def test_make_rolling_origin_splits_moves_forward_in_time(self) -> None:
        splits = make_rolling_origin_splits(
            100,
            initial_fit_frac=0.30,
            validation_frac=0.20,
            test_block_frac=0.10,
            min_fit=10,
            min_validation=5,
            min_test_block=5,
            max_origins=3,
        )

        self.assertEqual(splits, [(30, 50, 60), (40, 60, 70), (50, 70, 80)])
        for fit_end, val_end, test_end in splits:
            self.assertLess(fit_end, val_end)
            self.assertLess(val_end, test_end)

    def test_summarize_rolling_compares_to_best_non_bwar_per_origin(self) -> None:
        raw = pd.DataFrame(
            [
                {"dataset": "d", "job": "j", "origin": 0, "horizon": 1, "method": "persistence", "test_w2_mean": 10.0},
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "euclidean_gaussian_ar",
                    "test_w2_mean": 8.0,
                },
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "bwar_selected_ref",
                    "test_w2_mean": 6.0,
                },
                {"dataset": "d", "job": "j", "origin": 1, "horizon": 1, "method": "persistence", "test_w2_mean": 5.0},
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 1,
                    "horizon": 1,
                    "method": "log_euclidean_gaussian_ar",
                    "test_w2_mean": 4.0,
                },
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 1,
                    "horizon": 1,
                    "method": "bwar_selected_ref",
                    "test_w2_mean": 4.4,
                },
            ]
        )

        comp, summary, summary_h = summarize_rolling(raw)

        self.assertEqual(len(comp), 2)
        np.testing.assert_allclose(sorted(comp["gain_vs_best_non_bwar"]), [-0.10, 0.25])
        self.assertEqual(float(summary.loc[0, "positive_rate"]), 0.5)
        self.assertEqual(int(summary.loc[0, "n_jobs"]), 2)
        self.assertEqual(int(summary_h.loc[0, "n_jobs"]), 2)

    def test_summarize_rolling_prefers_domain_metric_when_available(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "dataset": "capital_bikeshare",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "raw_var_window_ar",
                    "test_domain_loss_mean": 2.0,
                    "test_log_score_mean": 1.0,
                    "domain_metric": "station_demand_level_rmse",
                },
                {
                    "dataset": "capital_bikeshare",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "bwar_selected_ref",
                    "test_domain_loss_mean": 1.0,
                    "test_log_score_mean": 4.0,
                    "domain_metric": "station_demand_level_rmse",
                },
            ]
        )

        comp, summary, _summary_h = summarize_rolling(raw)

        self.assertEqual(comp.loc[0, "primary_metric"], "test_domain_loss_mean")
        self.assertEqual(summary.loc[0, "domain_metric"], "station_demand_level_rmse")
        self.assertEqual(comp.loc[0, "best_non_bwar_method"], "raw_var_window_ar")
        self.assertAlmostEqual(float(comp.loc[0, "gain_vs_best_non_bwar"]), 0.5)
        self.assertAlmostEqual(float(summary.loc[0, "mean_gain"]), 0.5)

    def test_summarize_rolling_falls_back_to_log_score_when_domain_metric_missing(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "raw_var_window_ar",
                    "test_w2_mean": 100.0,
                    "test_log_score_mean": 5.0,
                },
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "bwar_selected_ref",
                    "test_w2_mean": 1.0,
                    "test_log_score_mean": 4.0,
                },
            ]
        )

        comp, summary, _summary_h = summarize_rolling(raw)

        self.assertEqual(comp.loc[0, "primary_metric"], "test_log_score_mean")
        self.assertEqual(comp.loc[0, "best_non_bwar_method"], "raw_var_window_ar")
        self.assertAlmostEqual(float(comp.loc[0, "gain_vs_best_non_bwar"]), 0.2)
        self.assertAlmostEqual(float(summary.loc[0, "mean_gain"]), 0.2)

    def test_summarize_rolling_handles_negative_log_scores(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "raw_var_window_ar",
                    "test_log_score_mean": -5.0,
                },
                {
                    "dataset": "d",
                    "job": "j",
                    "origin": 0,
                    "horizon": 1,
                    "method": "bwar_selected_ref",
                    "test_log_score_mean": -6.0,
                },
            ]
        )

        comp, summary, _summary_h = summarize_rolling(raw)

        self.assertAlmostEqual(float(comp.loc[0, "gain_vs_best_non_bwar"]), 0.2)
        self.assertTrue(bool(comp.loc[0, "selected_is_best_non_bwar_or_better"]))
        self.assertAlmostEqual(float(summary.loc[0, "mean_gain"]), 0.2)

    def test_run_rolling_origin_series_adds_raw_time_series_baseline(self) -> None:
        rng = np.random.default_rng(123)
        n_raw = 320
        d = 3
        raw = np.zeros((n_raw, d))
        for t in range(1, n_raw):
            raw[t] = 0.7 * raw[t - 1] + rng.normal(scale=0.2, size=d)
        window = 8
        step = 2
        starts = np.arange(0, n_raw - window + 1, step)
        starts = starts[:120]
        windows = np.asarray([raw[start : start + window] for start in starts])
        means = windows.mean(axis=1)
        covs = np.asarray([np.cov(W, rowvar=False) + 1e-4 * np.eye(d) for W in windows])

        rows, _refs = run_rolling_origin_series(
            job="unit",
            dataset="unit",
            means=means,
            covs=covs,
            meta={"window": window, "step": step},
            horizons=[1],
            raw_windows=windows,
            window_starts=starts,
            window_size=window,
            ar_model="diag",
            max_origins=1,
            min_n=50,
        )

        raw_baseline = rows.loc[rows["method"] == "raw_var_window_ar"]
        self.assertFalse(raw_baseline.empty)
        self.assertTrue(np.isfinite(raw_baseline["test_log_score_mean"]).all())
        self.assertTrue(np.isfinite(rows["test_log_score_mean"]).all())

    def test_run_rolling_origin_series_accepts_domain_profile_override(self) -> None:
        rng = np.random.default_rng(456)
        n = 90
        d = 2
        means = rng.normal(size=(n, d))
        covs = []
        for _ in range(n):
            A = rng.normal(size=(d, d))
            covs.append(np.eye(d) + 0.01 * A @ A.T)
        covs = np.asarray(covs)
        profile = {
            "metric": "unit_physical_level_rmse",
            "label": "unit physical level RMSE",
            "kind": "mean_rmse",
        }

        rows, _refs = run_rolling_origin_series(
            job="unit_profile",
            dataset="unit",
            means=means,
            covs=covs,
            meta={},
            horizons=[1],
            ar_model="diag",
            max_origins=1,
            min_n=50,
            domain_profile_override=profile,
        )

        self.assertFalse(rows.empty)
        self.assertEqual(set(rows["domain_metric"]), {"unit_physical_level_rmse"})
        self.assertEqual(set(rows["domain_metric_label"]), {"unit physical level RMSE"})

    def test_run_rolling_origin_series_uses_predeclared_splits(self) -> None:
        rng = np.random.default_rng(789)
        n = 90
        d = 2
        means = rng.normal(size=(n, d))
        covs = np.asarray([np.eye(d) for _ in range(n)])
        splits = [(40, 60, 75), (50, 75, 90)]

        rows, _refs = run_rolling_origin_series(
            job="unit_splits",
            dataset="unit",
            means=means,
            covs=covs,
            meta={},
            horizons=[1],
            ar_model="diag",
            max_origins=1,
            min_n=50,
            splits_override=splits,
        )

        observed = rows[["origin", "fit_end", "val_end", "test_end"]].drop_duplicates()
        self.assertEqual(observed.to_records(index=False).tolist(), [(0, 40, 60, 75), (1, 50, 75, 90)])


if __name__ == "__main__":
    unittest.main()
