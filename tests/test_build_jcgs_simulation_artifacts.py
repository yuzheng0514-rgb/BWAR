from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd
from pypdf import PdfReader

import bwar.paper_jcgs.build_jcgs_simulation_artifacts as simulation


class ReferenceDriftSimulationTest(unittest.TestCase):
    def test_shift_design_is_deterministic_causal_and_spd(self) -> None:
        common = dict(n=80, d=3, fit_end=36, phi=0.65, dispersion=0.10, seed=4)
        no_shift = simulation.simulate_reference_drift_gaussians(
            setting="No shift",
            **common,
        )
        covariance_shift = simulation.simulate_reference_drift_gaussians(
            setting="Covariance shift",
            **common,
        )
        repeated = simulation.simulate_reference_drift_gaussians(
            setting="Covariance shift",
            **common,
        )

        for left, right in zip(covariance_shift[:2], repeated[:2], strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(
            no_shift[0][: common["fit_end"]],
            covariance_shift[0][: common["fit_end"]],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            no_shift[1][: common["fit_end"]],
            covariance_shift[1][: common["fit_end"]],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreater(
            np.linalg.norm(no_shift[1][-1] - covariance_shift[1][-1], ord="fro"),
            1e-3,
        )
        self.assertGreater(
            min(np.linalg.eigvalsh(covariance).min() for covariance in covariance_shift[1]),
            0.0,
        )

    def test_small_drift_run_uses_complete_common_method_panel(self) -> None:
        result = simulation.run_reference_drift_setting(
            setting="Joint shift",
            seed=2,
            n=90,
            d=3,
            window_length=12,
            fit_fraction=0.45,
            validation_fraction=0.20,
            ridge_grid=(1e-2, 1e-1),
            refresh_period=6,
        )

        self.assertEqual(set(result["method"]), set(simulation.DRIFT_METHOD_ORDER))
        self.assertEqual(result["method"].nunique(), len(simulation.DRIFT_METHOD_ORDER))
        self.assertTrue(np.isfinite(result["w2_ratio_mean"]).all())
        self.assertTrue(np.isfinite(result["w2_mean"]).all())
        self.assertTrue(result["min_pred_eig"].gt(0.0).all())
        self.assertEqual(result["n_test_origins"].nunique(), 1)
        self.assertTrue(
            result.loc[result["method"].ne("persistence"), "ridge"].isin((1e-2, 1e-1)).all()
        )

    def test_drift_summary_reports_replication_standard_errors(self) -> None:
        frames = [
            simulation.run_reference_drift_setting(
                setting="No shift",
                seed=seed,
                n=80,
                d=3,
                window_length=12,
                fit_fraction=0.45,
                validation_fraction=0.20,
                ridge_grid=(1e-1,),
                refresh_period=6,
            )
            for seed in range(2)
        ]
        raw = simulation.pd.concat(frames, ignore_index=True)
        summary = simulation.summarize_reference_drift(raw)

        self.assertEqual(summary["n_rep"].unique().tolist(), [2])
        self.assertTrue(np.isfinite(summary["w2_ratio_se"]).all())
        self.assertEqual(set(summary["method"]), set(simulation.DRIFT_METHOD_ORDER))

    def test_jcgs_figure_exports_are_opaque_and_drift_axis_is_linear(self) -> None:
        fixed_raw_rows = []
        fixed_summary_rows = []
        fixed_methods = simulation._FIXED_PLOT_METHODS
        designs = (
            "Baseline",
            "Shorter series",
            "Higher dimension",
            "Weaker dynamics",
            "Larger variation",
        )
        for design_index, design in enumerate(designs):
            for method_index, method in enumerate(fixed_methods):
                center = 0.88 + 0.025 * method_index + 0.002 * design_index
                for seed in range(3):
                    fixed_raw_rows.append(
                        {
                            "design": design,
                            "method": method,
                            "seed": seed,
                            "w2_ratio_to_persistence": center + 0.002 * seed,
                        }
                    )
                fixed_summary_rows.append(
                    {
                        "design": design,
                        "method": method,
                        "w2_ratio_mean": center + 0.002,
                        "w2_ratio_se": 0.001,
                        "cov_ratio_mean": center + 0.005,
                        "cov_ratio_se": 0.001,
                    }
                )

        drift_raw_rows = []
        drift_summary_rows = []
        for setting_index, setting in enumerate(simulation.DRIFT_SETTINGS):
            for method_index, method in enumerate(simulation.DRIFT_METHOD_ORDER):
                center = 1.0 if method == "persistence" else 0.92 + 0.012 * method_index
                if method == "local_bwar":
                    center -= 0.0015 * setting_index
                for seed in range(3):
                    drift_raw_rows.append(
                        {
                            "setting": setting,
                            "method": method,
                            "seed": seed,
                            "w2_ratio_mean": center + 0.001 * seed,
                        }
                    )
                drift_summary_rows.append(
                    {
                        "setting": setting,
                        "method": method,
                        "w2_ratio_mean": center + 0.001,
                        "w2_ratio_se": 0.001,
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixed_stem = directory / "fixed"
            drift_stem = directory / "drift"
            simulation.make_fixed_reference_figure(
                pd.DataFrame(fixed_raw_rows),
                pd.DataFrame(fixed_summary_rows),
                fixed_stem,
            )
            simulation.make_reference_drift_figure(
                pd.DataFrame(drift_raw_rows),
                pd.DataFrame(drift_summary_rows),
                drift_stem,
            )

            drift_svg = drift_stem.with_suffix(".svg").read_text(encoding="utf-8")
            for stem in (fixed_stem, drift_stem):
                svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
                self.assertIsNone(
                    re.search(r"(?:fill-|stroke-)?opacity\s*[:=]", svg),
                    stem.name,
                )
                stroke_widths = [
                    float(value)
                    for value in re.findall(r"stroke-width:\s*([0-9.]+)", svg)
                ]
                self.assertTrue(stroke_widths, stem.name)
                self.assertGreaterEqual(min(stroke_widths), 0.3, stem.name)
                reader = PdfReader(stem.with_suffix(".pdf"))
                alpha_states = []
                for page in reader.pages:
                    resources = page["/Resources"].get_object()
                    for state_ref in resources.get("/ExtGState", {}).values():
                        state = state_ref.get_object()
                        alpha_states.append(
                            (float(state.get("/CA", 1)), float(state.get("/ca", 1)))
                        )
                self.assertTrue(
                    all(state == (1.0, 1.0) for state in alpha_states),
                    (stem.name, alpha_states),
                )

        self.assertNotIn("log scale", drift_svg)


if __name__ == "__main__":
    unittest.main()
