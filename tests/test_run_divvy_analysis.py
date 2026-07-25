from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd
import numpy as np

from bwar.paper_jcgs import run_divvy_analysis as run

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.run_divvy_analysis import (
    add_run_metadata,
    build_target_level_evidence,
    fixed_application_case,
    monthly_h2_splits,
    protocol_job_name,
    seasonal_period_windows,
)


class AdvisorRealdataRunnerTest(unittest.TestCase):
    def test_run_metadata_supplies_confirmation_dimension_argument(self) -> None:
        frame = pd.DataFrame({"method": ["fixed_bwar"]})

        result = add_run_metadata(
            frame,
            dimension=30,
            n_matrices=364,
            primary_horizon=3,
        )

        self.assertEqual(result["dimension"].tolist(), [30])
        self.assertEqual(result["dimension_arg"].tolist(), [30])
        self.assertEqual(result["n_matrices"].tolist(), [364])
        self.assertEqual(result["h0"].tolist(), [3])
        self.assertEqual(result["q_dimension"].tolist(), [495])

    def test_target_level_evidence_uses_locked_paired_bootstrap(self) -> None:
        rng = np.random.default_rng(44)
        starts = np.arange(65, dtype=int) * 2
        raw_series = rng.normal(size=(int(starts[-1] + 6), 2))
        raw_windows = np.asarray([raw_series[start : start + 6] for start in starts])
        means = raw_windows.mean(axis=1)
        covariances = np.asarray(
            [np.cov(window, rowvar=False) + 0.1 * np.eye(2) for window in raw_windows]
        )

        panel, tuning, selected, inference = build_target_level_evidence(
            means=means,
            covariances=covariances,
            raw_series=raw_series,
            raw_windows=raw_windows,
            starts=starts,
            window_size=6,
            seasonal_period=1,
            splits=((40, 55, 65),),
            horizons=(3,),
            domain_profile={"kind": "mean_rmse"},
            ridge_grid=(0.1, 1.0),
            local_window_grid=(12, 18),
            bootstrap_replicates=100,
        )

        self.assertEqual(panel["method"].nunique(), 8)
        self.assertEqual(len(tuning), 4)
        self.assertEqual(len(selected), 1)
        self.assertEqual(set(inference["target_method"]), {"fixed_bwar", "local_bwar"})
        self.assertTrue(
            (
                inference["target_method"].eq("local_bwar")
                & inference["comparator"].eq("fixed_bwar")
            ).any()
        )
        self.assertEqual(set(inference["block_length"]), {3})
        self.assertEqual(set(inference["sensitivity_block_length"]), {6})

    def test_authoritative_divvy_protocol_is_pinned(self) -> None:
        self.assertEqual(run.WINDOW, 72)
        self.assertEqual(run.STEP, 24)
        self.assertEqual(run.DIMENSION, 30)
        self.assertEqual(run.HORIZONS, (3, 4, 6))
        self.assertEqual(run.MAX_ORIGINS, 6)

    def test_monthly_h2_splits_reserve_july_through_december_for_testing(self) -> None:
        starts = np.arange(0, 366 * 24 - 24 + 1, 6)

        splits = monthly_h2_splits(starts, window=24)

        self.assertEqual(len(splits), 6)
        self.assertEqual(splits[0], (481, 728, 852))
        self.assertEqual(splits[-1][2], len(starts))
        self.assertTrue(all(a < b < c for a, b, c in splits))

    def test_protocol_metadata_tracks_window_step_and_dimension(self) -> None:
        self.assertEqual(protocol_job_name(window=24, step=6, dimension=30), "divvy_2024_w24_s6_d30")
        self.assertEqual(seasonal_period_windows(step=6), 4)
        self.assertEqual(seasonal_period_windows(step=24), 1)

    def test_fixed_application_case_does_not_rank_across_candidates(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "candidate": "other",
                    "window": 72,
                    "step": 24,
                    "dimension": 40,
                    "horizon": 3,
                    "raw_margin_fraction": 0.90,
                },
                {
                    "candidate": "divvy",
                    "window": 72,
                    "step": 24,
                    "dimension": 30,
                    "horizon": 3,
                    "raw_margin_fraction": -0.10,
                },
            ]
        )

        case = fixed_application_case(summary)

        self.assertEqual(case["candidate"], "divvy")
        self.assertEqual(int(case["horizon"]), 3)

    def test_fixed_application_case_requires_the_registered_protocol(self) -> None:
        summary = pd.DataFrame(
            [{"candidate": "divvy", "window": 24, "step": 6, "dimension": 30, "horizon": 4}]
        )

        with self.assertRaisesRegex(ValueError, "fixed Divvy protocol"):
            fixed_application_case(summary)


if __name__ == "__main__":
    unittest.main()
