from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
S2_METHODS = {
    "persistence",
    "euclidean",
    "cholesky",
    "log_euclidean",
    "fixed",
    "local",
}
PEMS_METHODS = {
    "Persistence",
    "Euclidean",
    "Cholesky",
    "Log-Euclidean",
    "BWAR",
}


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
        self.assertEqual(set(s2["method"]), S2_METHODS)
        s2_protocol = json.loads(
            (
                ROOT
                / "results"
                / "reference"
                / "s2_reference_adaptation"
                / "protocol.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(s2_protocol["methods"]), S2_METHODS)
        self.assertEqual(
            s2_protocol["local_ridge_policy"], "ridge selected for fixed BWAR"
        )
        s2_raw = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "s2_reference_adaptation"
            / "replication_results.csv.gz"
        )
        self.assertEqual(set(s2_raw["method"]), S2_METHODS)
        paired_ridges = s2_raw.loc[
            s2_raw["method"].isin(("fixed", "local")),
            ["replication", "regime", "target_delta", "horizon", "method", "selected_ridge"],
        ].pivot(
            index=["replication", "regime", "target_delta", "horizon"],
            columns="method",
            values="selected_ridge",
        )
        self.assertTrue(paired_ridges["fixed"].eq(paired_ridges["local"]).all())

        divvy = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "divvy"
            / "target_level_losses.csv"
        )
        self.assertEqual(set(divvy["horizon"]), {3, 4, 5})
        self.assertEqual(divvy["method"].nunique(), 7)
        protocol = json.loads(
            (
                ROOT
                / "results"
                / "reference"
                / "divvy"
                / "protocol.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(protocol["methods"]), set(divvy["method"]))
        self.assertEqual(
            protocol["selection_endpoint"],
            "training-standardized station-mean RMSE",
        )
        targets = divvy.groupby(["method", "horizon"])["target_index"].nunique()
        self.assertEqual(set(targets), {183})

        pems = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "pems_bay"
            / "test_method_summary.csv"
        )
        self.assertEqual(set(pems["horizon"]), {1, 3, 6})
        self.assertEqual(set(pems["method"]), PEMS_METHODS)

    def test_release_tree_excludes_superseded_workflows(self) -> None:
        release_roots = (
            ROOT / "configs",
            ROOT / "scripts",
            ROOT / "src",
            ROOT / "results" / "reference",
            ROOT / "artifacts" / "reference",
        )
        retired_tokens = {
            "beijing",
            "electricity",
            "weekly",
            "seasonal",
            "estimated_states",
            "theory_diagnostics",
            "rolling_drift",
            "local_shared",
        }
        for root in release_roots:
            for path in root.rglob("*"):
                normalized = str(path.relative_to(ROOT)).lower().replace("-", "_")
                self.assertFalse(
                    any(token in normalized for token in retired_tokens),
                    f"superseded release path retained: {normalized}",
                )


if __name__ == "__main__":
    unittest.main()
