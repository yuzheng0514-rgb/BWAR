#!/usr/bin/env python3
"""Verify archived results and regenerated article artifacts."""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def compare_text(expected: Path, actual: Path) -> None:
    left = expected.read_text(encoding="utf-8").splitlines()
    right = actual.read_text(encoding="utf-8").splitlines()
    if left != right:
        difference = "\n".join(
            difflib.unified_diff(
                left,
                right,
                fromfile=str(expected),
                tofile=str(actual),
                lineterm="",
            )
        )
        raise AssertionError(f"artifact mismatch:\n{difference}")


def main() -> None:
    reference = ROOT / "results" / "reference"
    submitted = ROOT / "artifacts" / "submitted"
    generated = ROOT / "artifacts" / "generated"

    fixed_raw = pd.read_csv(
        reference
        / "fixed_simulation"
        / "strong_synthetic_transport_raw.csv"
    )
    fixed_summary = pd.read_csv(
        reference
        / "fixed_simulation"
        / "strong_synthetic_transport_summary.csv"
    )
    require(len(fixed_raw) == 1250, "fixed simulation must contain 1,250 rows")
    require(len(fixed_summary) == 25, "fixed summary must contain 25 rows")
    require(fixed_raw["seed"].nunique() == 50, "fixed simulation needs 50 seeds")
    main_bwar = fixed_summary.loc[
        fixed_summary["design"].eq("Baseline")
        & fixed_summary["method"].eq("bwar_barycenter"),
        "w2_ratio_mean",
    ].iloc[0]
    require(np.isclose(main_bwar, 0.887, atol=5e-4), "unexpected BWAR main ratio")

    drift_raw = pd.read_csv(
        reference / "rolling_drift" / "local_reference_drift_raw.csv"
    )
    drift_summary = pd.read_csv(
        reference / "rolling_drift" / "local_reference_drift_summary.csv"
    )
    require(len(drift_raw) == 1500, "rolling-drift simulation needs 1,500 rows")
    require(
        drift_summary["n_rep"].eq(50).all(),
        "rolling-drift summary must use 50 replications",
    )
    gradual = drift_summary.loc[
        drift_summary["setting"].eq("Gradual joint shift")
        & drift_summary["method"].isin(["fixed_bwar", "local_bwar"])
    ].set_index("method")["w2_ratio_mean"]
    require(
        gradual["local_bwar"] < gradual["fixed_bwar"],
        "unexpected gradual-drift ordering",
    )

    panel = pd.read_csv(reference / "divvy" / "target_level_losses.csv")
    require(
        panel.loc[panel["horizon"].eq(3), "target_index"].nunique() == 183,
        "Divvy primary horizon must contain 183 targets",
    )
    inference = pd.read_csv(reference / "divvy" / "paired_inference.csv")
    local_fixed_w2 = inference.loc[
        inference["target_method"].eq("local_bwar")
        & inference["comparator"].eq("fixed_bwar")
        & inference["metric"].eq("w2"),
        "p_value",
    ].iloc[0]
    require(
        np.isclose(local_fixed_w2, 0.3320667933206679),
        "unexpected local-versus-fixed W2 inference",
    )

    for table in sorted((submitted / "tables").glob("*.tex")):
        compare_text(table, generated / "tables" / table.name)
    for figure in sorted((submitted / "figures").glob("*.pdf")):
        regenerated = generated / "figures" / figure.name
        require(regenerated.exists(), f"missing generated figure: {figure.name}")
        require(regenerated.stat().st_size > 10_000, f"empty figure: {figure.name}")

    prohibited = list(ROOT.glob("data/divvy/*.zip"))
    require(not prohibited, "raw Divvy archives must remain untracked")
    print("Reference results and article artifacts verified.")


if __name__ == "__main__":
    main()
