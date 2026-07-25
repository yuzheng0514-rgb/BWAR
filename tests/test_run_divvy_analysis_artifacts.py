from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from bwar.paper_jcgs import run_divvy_analysis


class RepresentativeForecastSelectionTests(unittest.TestCase):
    def test_manifest_records_target_inference_and_local_protocol(self) -> None:
        long = pd.DataFrame(
            {
                "origin": [0],
                "fit_end": [127],
                "val_end": [181],
                "test_end": [224],
                "evaluation_protocol": ["fractional_rolling_origin"],
            }
        )
        target_panel = pd.DataFrame(
            {
                "origin": [0, 0],
                "horizon": [3, 3],
                "target_index": [181, 182],
                "method": ["local_bwar", "local_bwar"],
            }
        )
        selected = pd.DataFrame(
            {"origin": [0], "horizon": [3], "window_length": [24], "ridge": [0.1]}
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            run_divvy_analysis.write_manifest(
                path,
                station_ids=("A", "B"),
                starts=np.arange(10),
                profile={"center": np.zeros(2), "scale": np.ones(2)},
                long=long,
                target_panel=target_panel,
                selected_local=selected,
            )
            payload = json.loads(path.read_text())

        self.assertEqual(payload["bootstrap"]["block_length"], 3)
        self.assertEqual(payload["bootstrap"]["sensitivity_block_length"], 6)
        self.assertEqual(payload["target_level_evaluation"]["primary_target_count"], 2)
        self.assertEqual(payload["local_reference"]["window_grid"], [24, 48, 72])

    def test_target_level_writer_exports_inference_and_local_diagnostics(self) -> None:
        panel = pd.DataFrame(
            {
                "origin": [0],
                "horizon": [3],
                "target_index": [20],
                "method": ["local_bwar"],
                "raw_rmse": [0.5],
                "w2": [1.5],
                "min_pred_eig": [0.2],
                "reference_residual": [1e-8],
                "reference_refreshed": [True],
                "reference_fallback": [False],
            }
        )
        tuning = pd.DataFrame(
            {"origin": [0], "horizon": [3], "window_length": [24], "ridge": [0.1]}
        )
        selected = tuning.copy()
        inference = pd.DataFrame(
            {"target_method": ["local_bwar"], "comparator": ["cholesky"]}
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_divvy_analysis.write_target_level_outputs(
                output_dir=output_dir,
                panel=panel,
                tuning=tuning,
                selected=selected,
                inference=inference,
            )

            expected = {
                "target_level_losses.csv",
                "paired_inference.csv",
                "local_tuning.csv",
                "local_selected_settings.csv",
                "local_reference_diagnostics.csv",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            diagnostics = pd.read_csv(output_dir / "local_reference_diagnostics.csv")

        self.assertEqual(diagnostics["method"].tolist(), ["local_bwar"])
        self.assertEqual(diagnostics["reference_fallback"].tolist(), [False])

    def test_first_held_out_forecast_is_fixed_before_viewing_results(self) -> None:
        forecast_origin, target_index = (
            run_divvy_analysis.first_held_out_forecast_indices(
                validation_end=181,
                horizon=3,
            )
        )

        self.assertEqual(forecast_origin, 178)
        self.assertEqual(target_index, 181)

    def test_snapshot_writer_exports_physical_means_and_correlations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_divvy_analysis.write_representative_forecast_snapshot(
                output_dir=output_dir,
                station_ids=("A", "B"),
                center=np.array([10.0, 20.0]),
                scale=np.array([2.0, 4.0]),
                forecasts={
                    "observed": (
                        np.array([1.0, -1.0]),
                        np.array([[4.0, 1.0], [1.0, 9.0]]),
                    ),
                    "cholesky_gaussian_ar": (
                        np.array([0.5, 0.0]),
                        np.array([[1.0, 0.0], [0.0, 1.0]]),
                    ),
                    "bwar_barycenter": (
                        np.array([0.0, 0.5]),
                        np.array([[4.0, -2.0], [-2.0, 4.0]]),
                    ),
                },
                metadata={"forecast_origin": 178, "target_index": 181},
            )

            means = pd.read_csv(output_dir / "representative_forecast_means.csv")
            correlations = pd.read_csv(
                output_dir / "representative_forecast_correlations.csv"
            )

        observed = means.loc[means["method"].eq("observed"), "mean_count"]
        self.assertEqual(observed.tolist(), [12.0, 16.0])
        bwar_corr = correlations.loc[
            correlations["method"].eq("bwar_barycenter")
        ].pivot(index="row", columns="column", values="correlation")
        np.testing.assert_allclose(np.diag(bwar_corr), 1.0)
        self.assertAlmostEqual(float(bwar_corr.iloc[0, 1]), -0.5)

    def test_representative_forecast_fit_returns_observed_and_two_methods(
        self,
    ) -> None:
        rng = np.random.default_rng(20260714)
        means = rng.normal(size=(48, 2))
        covariances = np.asarray(
            [
                matrix @ matrix.T + 0.5 * np.eye(2)
                for matrix in rng.normal(size=(48, 2, 2))
            ]
        )

        forecasts, metadata = (
            run_divvy_analysis.fit_representative_forecasts(
                means=means,
                covariances=covariances,
                fit_end=24,
                validation_end=36,
                horizon=2,
                domain_profile={"metric": "toy", "label": "Toy", "kind": "mean_rmse"},
            )
        )

        self.assertEqual(
            set(forecasts),
            {"observed", "cholesky_gaussian_ar", "bwar_barycenter"},
        )
        for mean, covariance in forecasts.values():
            self.assertEqual(mean.shape, (2,))
            self.assertEqual(covariance.shape, (2, 2))
            self.assertGreater(float(np.linalg.eigvalsh(covariance).min()), 0.0)
        self.assertEqual(metadata["forecast_origin"], 34)
        self.assertEqual(metadata["target_index"], 36)


if __name__ == "__main__":
    unittest.main()
