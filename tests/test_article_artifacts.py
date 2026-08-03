from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ArticleArtifactTests(unittest.TestCase):
    def test_tables_rebuild_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_artifacts.py"),
                    "--output-root",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            for name in (
                "geometry_robustness_rebuild.tex",
                "divvy_full_results_rebuild.tex",
                "pems_full_results.tex",
            ):
                self.assertEqual(
                    (output / "tables" / name).read_bytes(),
                    (ROOT / "artifacts" / "reference" / "tables" / name).read_bytes(),
                )

    def test_reference_panels_match_article_protocol(self) -> None:
        s1 = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "s1_geometry"
            / "strong_synthetic_transport_summary.csv"
        )
        self.assertEqual(set(s1["n_rep"]), {80})

        s2 = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "s2_reference_adaptation"
            / "summary.csv"
        )
        self.assertEqual(set(s2["n_replications"]), {100})
        self.assertEqual(set(s2["regime"]), {"stable", "continuing"})

        divvy = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "divvy"
            / "target_level_losses.csv"
        )
        self.assertEqual(set(divvy["horizon"]), {3, 4, 5})
        self.assertEqual(divvy["method"].nunique(), 7)
        targets = divvy.groupby(["method", "horizon"])["target_index"].nunique()
        self.assertEqual(set(targets), {183})


if __name__ == "__main__":
    unittest.main()
