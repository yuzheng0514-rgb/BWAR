from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pypdf import PdfReader

from bwar.paper_jcgs import polish_jcgs_figures


class SyntheticFigureExportTests(unittest.TestCase):
    def test_group_palette_uses_recent_jcgs_method_hierarchy(self) -> None:
        self.assertEqual(
            polish_jcgs_figures.METHOD_COLOR["bwar_barycenter"], "#D62728"
        )
        self.assertEqual(
            polish_jcgs_figures.METHOD_COLOR["euclidean_gaussian_ar"], "#111111"
        )
        self.assertEqual(
            polish_jcgs_figures.METHOD_COLOR["cholesky_gaussian_ar"], "#9467BD"
        )
        self.assertEqual(
            polish_jcgs_figures.METHOD_COLOR["log_euclidean_gaussian_ar"],
            "#1F77B4",
        )
        self.assertEqual(polish_jcgs_figures.SYNTHETIC_MAIN_DISPLAY, "mean_se")

    def test_parameter_variations_are_not_connected_as_an_ordered_curve(self) -> None:
        style = polish_jcgs_figures.parameter_variation_style("bwar_barycenter")

        self.assertEqual(style["linestyle"], "none")
        self.assertEqual(style["color"], "#D62728")

    def test_realdata_default_is_the_target_level_local_divvy_run(self) -> None:
        self.assertEqual(
            polish_jcgs_figures.REAL_OUT,
            polish_jcgs_figures.ROOT
            / "results"
            / "reference"
            / "divvy",
        )

    def test_realdata_style_distinguishes_fixed_and_local_bwar(self) -> None:
        self.assertIn("fixed_bwar", polish_jcgs_figures.REAL_METHODS)
        self.assertIn("local_bwar", polish_jcgs_figures.REAL_METHODS)
        self.assertEqual(polish_jcgs_figures.METHOD_LABEL["fixed_bwar"], "Fixed BWAR")
        self.assertEqual(polish_jcgs_figures.METHOD_LABEL["local_bwar"], "Local BWAR")
        self.assertNotEqual(
            polish_jcgs_figures.METHOD_LINESTYLE["fixed_bwar"],
            polish_jcgs_figures.METHOD_LINESTYLE["local_bwar"],
        )

    def test_synthetic_svg_contains_no_transparent_artists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figure_dir = Path(tmp)
            with patch.object(polish_jcgs_figures, "FIGURE_DIR", figure_dir):
                polish_jcgs_figures.make_synthetic_figure()

            svg = (figure_dir / "synthetic_transport_mechanism.svg").read_text(
                encoding="utf-8"
            )

        self.assertNotIn("opacity:", svg)
        self.assertNotIn("opacity=", svg)
        self.assertTrue(all(line == line.rstrip() for line in svg.splitlines()))

    def test_synthetic_pdf_uses_only_opaque_graphics_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figure_dir = Path(tmp)
            with patch.object(polish_jcgs_figures, "FIGURE_DIR", figure_dir):
                polish_jcgs_figures.make_synthetic_figure()

            reader = PdfReader(figure_dir / "synthetic_transport_mechanism.pdf")
            alpha_states = []
            for page in reader.pages:
                resources = page["/Resources"].get_object()
                for state_ref in resources.get("/ExtGState", {}).values():
                    state = state_ref.get_object()
                    alpha_states.append(
                        (float(state.get("/CA", 1)), float(state.get("/ca", 1)))
                    )

        self.assertTrue(
            all(state == (1.0, 1.0) for state in alpha_states), alpha_states
        )

    def test_realdata_exports_use_only_opaque_artists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            figure_dir = Path(tmp)
            with patch.object(polish_jcgs_figures, "FIGURE_DIR", figure_dir):
                polish_jcgs_figures.make_realdata_figure()

            svg = (figure_dir / "redone_realdata_application.svg").read_text(
                encoding="utf-8"
            )
            reader = PdfReader(figure_dir / "redone_realdata_application.pdf")
            alpha_states = []
            for page in reader.pages:
                resources = page["/Resources"].get_object()
                for state_ref in resources.get("/ExtGState", {}).values():
                    state = state_ref.get_object()
                    alpha_states.append(
                        (float(state.get("/CA", 1)), float(state.get("/ca", 1)))
                    )

        self.assertNotIn("opacity:", svg)
        self.assertNotIn("opacity=", svg)
        self.assertTrue(
            all(state == (1.0, 1.0) for state in alpha_states), alpha_states
        )

    def test_representative_forecast_figure_exports_group_style_domain_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            figure_dir = root / "figures"
            data_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "method": method,
                        "station_rank": station,
                        "station_id": f"S{station}",
                        "mean_count": value,
                    }
                    for method, values in {
                        "observed": [2.0, 4.0, 3.0],
                        "cholesky_gaussian_ar": [2.2, 3.5, 3.4],
                        "bwar_barycenter": [2.1, 3.8, 3.1],
                    }.items()
                    for station, value in enumerate(values, start=1)
                ]
            ).to_csv(data_dir / "representative_forecast_means.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "method": method,
                        "row": row,
                        "column": column,
                        "correlation": float(matrix[row, column]),
                    }
                    for method, matrix in {
                        "observed": __import__("numpy").eye(3),
                        "cholesky_gaussian_ar": __import__("numpy").eye(3),
                        "bwar_barycenter": __import__("numpy").eye(3),
                    }.items()
                    for row in range(3)
                    for column in range(3)
                ]
            ).to_csv(
                data_dir / "representative_forecast_correlations.csv", index=False
            )

            with (
                patch.object(polish_jcgs_figures, "REAL_OUT", data_dir),
                patch.object(polish_jcgs_figures, "FIGURE_DIR", figure_dir),
            ):
                polish_jcgs_figures.make_representative_forecast_figure()

            svg = (
                figure_dir / "divvy_representative_forecast.svg"
            ).read_text(encoding="utf-8")

        self.assertNotIn("opacity:", svg)
        self.assertNotIn("opacity=", svg)


if __name__ == "__main__":
    unittest.main()
