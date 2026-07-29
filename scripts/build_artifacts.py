#!/usr/bin/env python3
"""Build the article tables and figures from archived or regenerated results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bwar.paper_jcgs import build_jcgs_simulation_artifacts as simulation
from bwar.paper_jcgs import build_strong_synthetic_artifacts as fixed
from bwar.paper_jcgs import polish_jcgs_figures as figures


ROOT = Path(__file__).resolve().parents[1]
METHOD_ROWS = [
    ("Persistence", "persistence"),
    ("Raw VAR", "raw_var_window_ar"),
    ("Euclidean AR", "euclidean_gaussian_ar"),
    ("Cholesky AR", "cholesky_gaussian_ar"),
    ("Log-Euclidean AR", "log_euclidean_gaussian_ar"),
    ("Fixed BWAR", "fixed_bwar"),
    ("Local BWAR", "local_bwar"),
]


def _best_cell(value: float, best: float) -> str:
    formatted = f"{value:.3f}"
    return rf"\textbf{{{formatted}}}" if np.isclose(value, best) else formatted


def write_performance_table(summary: pd.DataFrame, path: Path) -> None:
    primary = summary.loc[summary["horizon"].eq(3)].set_index(
        ["method", "metric"]
    )
    best_raw = float(
        summary.loc[
            summary["horizon"].eq(3) & summary["metric"].eq("raw_rmse"),
            "mean",
        ].min()
    )
    best_w2 = float(
        summary.loc[
            summary["horizon"].eq(3) & summary["metric"].eq("w2"),
            "mean",
        ].min()
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Divvy performance at the primary horizon \(h=3\). "
            r"Entries average 183 held-out target windows across five "
            r"chronological rolling origins. Lower values are better. Local "
            r"BWAR selects its reference window length and ridge from the "
            r"validation block of each origin. The physical-mean RMSE is "
            r"standardized using fitting-block scales.}"
        ),
        r"\label{tab:divvy-performance}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Physical-mean RMSE & Gaussian \(W_2^2\) loss \\",
        r"\midrule",
    ]
    for label, method in METHOD_ROWS:
        raw = float(primary.loc[(method, "raw_rmse"), "mean"])
        w2 = float(primary.loc[(method, "w2"), "mean"])
        lines.append(
            f"{label} & {_best_cell(raw, best_raw)} & "
            f"{_best_cell(w2, best_w2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_horizon_table(summary: pd.DataFrame, path: Path) -> None:
    raw = summary.loc[summary["metric"].eq("raw_rmse")].set_index(
        ["method", "horizon"]
    )
    horizons = (3, 4)
    best = {
        horizon: float(
            summary.loc[
                summary["metric"].eq("raw_rmse")
                & summary["horizon"].eq(horizon),
                "mean",
            ].min()
        )
        for horizon in horizons
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Horizon sensitivity of the Divvy physical endpoint. "
            r"Entries are target-weighted standardized physical-mean RMSEs over the same "
            r"five chronological rolling origins. Lower values are better.}"
        ),
        r"\label{tab:divvy-horizon}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & \(h=3\) & \(h=4\) \\",
        r"\midrule",
    ]
    for label, method in METHOD_ROWS:
        cells = [
            _best_cell(float(raw.loc[(method, horizon), "mean"]), best[horizon])
            for horizon in horizons
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all empirical tables and figures from result CSVs."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "reference",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "artifacts" / "generated",
    )
    args = parser.parse_args()
    table_dir = args.artifact_root / "tables"
    figure_dir = args.artifact_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fixed_root = args.result_root / "fixed_simulation"
    drift_root = args.result_root / "rolling_drift"
    divvy_root = args.result_root / "divvy"

    fixed_summary = pd.read_csv(
        fixed_root / "strong_synthetic_transport_summary.csv"
    )
    fixed.write_main_table(
        fixed_summary,
        table_dir / "synthetic_transport_main.tex",
    )
    fixed.write_variation_table(
        fixed_summary,
        table_dir / "synthetic_transport_variation.tex",
    )

    drift_raw = pd.read_csv(drift_root / "local_reference_drift_raw.csv")
    drift_summary = pd.read_csv(
        drift_root / "local_reference_drift_summary.csv"
    )
    drift_strength_summary = pd.read_csv(
        drift_root / "drift_strength_summary.csv"
    )
    simulation.write_reference_drift_table(
        drift_summary,
        table_dir / "rolling_refit_reference_shift.tex",
        int(drift_summary["n_rep"].max()),
    )
    simulation.make_reference_drift_figure(
        drift_raw,
        drift_summary,
        drift_strength_summary,
        figure_dir / "rolling_refit_reference_shift",
    )

    bootstrap_summary = pd.read_csv(
        divvy_root / "method_level_bootstrap_summary.csv"
    )
    write_performance_table(
        bootstrap_summary,
        table_dir / "redone_realdata_application.tex",
    )
    write_horizon_table(
        bootstrap_summary,
        table_dir / "redone_realdata_horizon.tex",
    )
    figures.FIGURE_DIR = figure_dir
    figures.SYNTH_OUT = fixed_root
    figures.REAL_OUT = divvy_root
    figures.make_synthetic_figure()
    figures.make_realdata_figure()
    print(f"Generated article artifacts: {args.artifact_root}")


if __name__ == "__main__":
    main()
