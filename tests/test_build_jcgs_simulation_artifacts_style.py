from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from bwar.paper_jcgs import build_jcgs_simulation_artifacts


class DriftFigureStyleTests(unittest.TestCase):
    def test_local_bwar_uses_proposed_red_and_fixed_bwar_uses_blue(self) -> None:
        self.assertEqual(
            build_jcgs_simulation_artifacts._COLOR["local_bwar"], "#D62728"
        )
        self.assertEqual(
            build_jcgs_simulation_artifacts._COLOR["fixed_bwar"], "#1F77B4"
        )

    def test_drift_figure_export_is_opaque(self) -> None:
        reference_dir = (
            build_jcgs_simulation_artifacts.ROOT
            / "results"
            / "reference"
            / "rolling_drift"
        )
        raw = pd.read_csv(
            reference_dir / "local_reference_drift_raw.csv"
        )
        summary = pd.read_csv(
            reference_dir / "local_reference_drift_summary.csv"
        )
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "rolling_refit_reference_shift"
            build_jcgs_simulation_artifacts.make_reference_drift_figure(
                raw, summary, stem
            )
            svg = stem.with_suffix(".svg").read_text(encoding="utf-8")

        self.assertNotIn("opacity:", svg)
        self.assertNotIn("opacity=", svg)


if __name__ == "__main__":
    unittest.main()
