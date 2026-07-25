from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.raw_physical_level_screen import (
    _select_variable_columns,
    _standardized_stream_with_profile,
    bikeshare_matrix,
)
from bwar.paper_jcgs.rolling_origin_backtest import make_rolling_origin_splits
from bwar.paper_jcgs.redo_realdata_application import load_gaussian_stream
from bwar.paper_jcgs.redo_realdata_application import combine_bwar_rows, confirmation_dimension


class RedoRealdataApplicationTest(unittest.TestCase):
    def test_confirmation_reuses_loader_dimension_argument(self) -> None:
        row = pd.Series({"dimension": 40, "dimension_arg": 10})
        legacy_row = pd.Series({"dimension": 30})

        self.assertEqual(confirmation_dimension(row), 10)
        self.assertEqual(confirmation_dimension(legacy_row), 30)

    def test_combine_bwar_rows_keeps_complete_selected_barycenter_metrics(self) -> None:
        raw = pd.DataFrame(
            [
                {"method": "persistence", "test_domain_loss_mean": 2.0},
                {
                    "method": "bwar_selected_ref",
                    "selected_reference": "bw_barycenter",
                    "test_domain_loss_mean": 1.0,
                    "n_test_pairs": 17,
                    "min_pred_eig": 0.125,
                },
            ]
        )
        refs = pd.DataFrame(
            [
                {
                    "reference": "bw_barycenter",
                    "test_domain_loss_mean": 1.0,
                }
            ]
        )

        combined = combine_bwar_rows(raw, refs, candidate="divvy")
        bwar = combined.loc[combined["method"].eq("bwar_barycenter")].iloc[0]

        self.assertEqual(int(bwar["n_test_pairs"]), 17)
        self.assertAlmostEqual(float(bwar["min_pred_eig"]), 0.125)

    def test_divvy_loader_applies_fit_only_station_selection(self) -> None:
        rng = np.random.default_rng(19)
        frame = pd.DataFrame(rng.normal(scale=[4.0, 3.0, 1.0, 0.5], size=(1000, 4)), columns=list("abcd")).abs()
        frame.loc[360:, "d"] = np.tile([0.0, 10_000.0], 320)

        months = ("202401", "202412")
        with patch("bwar.paper_jcgs.raw_physical_level_screen.load_bikeshare_hourly", return_value=frame) as mocked_load:
            actual = load_gaussian_stream(
                "divvy",
                window=20,
                step=5,
                dimension=2,
                max_matrices=900,
                months=months,
            )

        mocked_load.assert_called_once_with("divvy", months)
        expected = _standardized_stream_with_profile(
            frame[["a", "b"]].to_numpy(float),
            window=20,
            step=5,
            max_matrices=900,
            metric="divvy_standardized_raw_mean_rmse",
            label="bike-share station trip-count level standardized raw mean RMSE",
        )
        np.testing.assert_allclose(actual[0], expected[0], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(actual[1], expected[1], atol=0.0, rtol=0.0)

    def test_bikeshare_station_set_is_chosen_from_first_origin_fit_block(self) -> None:
        rng = np.random.default_rng(17)
        frame = pd.DataFrame(rng.normal(scale=[4.0, 3.0, 1.0, 0.5], size=(1000, 4)), columns=list("abcd"))
        frame = frame.abs()
        frame.loc[300:, "d"] = np.tile([0.0, 10_000.0], 350)

        months = ("202401", "202402")
        with patch("bwar.paper_jcgs.raw_physical_level_screen.load_bikeshare_hourly", return_value=frame) as mocked_load:
            selected, station_ids = bikeshare_matrix(
                "divvy",
                2,
                months=months,
                window=20,
                step=5,
                max_matrices=900,
                fit_raw_end=300,
                return_columns=True,
            )

        mocked_load.assert_called_once_with("divvy", months)
        self.assertEqual(station_ids, ("a", "b"))
        np.testing.assert_array_equal(selected, frame[["a", "b"]].to_numpy(float))

    def test_station_selection_ignores_values_after_initial_fit_block(self) -> None:
        raw = np.zeros((200, 4), dtype=float)
        raw[:80, 0] = np.tile([0.0, 3.0], 40)
        raw[:80, 1] = np.tile([0.0, 2.0], 40)
        raw[:80, 2] = np.tile([0.0, 1.0], 40)
        raw[:80, 3] = np.tile([0.0, 0.5], 40)

        contaminated = raw.copy()
        contaminated[80:, 3] = np.tile([0.0, 1000.0], 60)

        selected = _select_variable_columns(raw, 2, early_end=80)
        selected_after_contamination = _select_variable_columns(contaminated, 2, early_end=80)

        np.testing.assert_array_equal(selected, selected_after_contamination)
        np.testing.assert_array_equal(selected, raw[:, [0, 1]])

    def test_standardization_uses_only_first_origin_fit_block(self) -> None:
        rng = np.random.default_rng(20260711)
        raw = rng.normal(size=(1000, 3))
        window = 20
        step = 5

        base = _standardized_stream_with_profile(
            raw,
            window=window,
            step=step,
            max_matrices=900,
            metric="unit_raw_mean_rmse",
            label="unit raw mean RMSE",
        )
        starts = base[3]
        fit_end = make_rolling_origin_splits(len(starts), max_origins=1)[0][0]
        fit_raw_end = int(starts[fit_end - 1] + window)

        contaminated = raw.copy()
        contaminated[fit_raw_end : fit_raw_end + 50] += 10_000.0
        changed = _standardized_stream_with_profile(
            contaminated,
            window=window,
            step=step,
            max_matrices=900,
            metric="unit_raw_mean_rmse",
            label="unit raw mean RMSE",
        )

        np.testing.assert_allclose(base[4]["center"], changed[4]["center"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(base[4]["scale"], changed[4]["scale"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(base[0][:fit_end], changed[0][:fit_end], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(base[1][:fit_end], changed[1][:fit_end], atol=0.0, rtol=0.0)

    def test_standardization_accepts_predeclared_raw_fit_boundary(self) -> None:
        rng = np.random.default_rng(20260712)
        raw = rng.normal(size=(1000, 3))
        contaminated = raw.copy()
        contaminated[300:350] += 10_000.0

        base = _standardized_stream_with_profile(
            raw,
            window=20,
            step=5,
            max_matrices=900,
            metric="unit_raw_mean_rmse",
            label="unit raw mean RMSE",
            fit_raw_end=300,
        )
        changed = _standardized_stream_with_profile(
            contaminated,
            window=20,
            step=5,
            max_matrices=900,
            metric="unit_raw_mean_rmse",
            label="unit raw mean RMSE",
            fit_raw_end=300,
        )

        np.testing.assert_allclose(base[4]["center"], changed[4]["center"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(base[4]["scale"], changed[4]["scale"], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
