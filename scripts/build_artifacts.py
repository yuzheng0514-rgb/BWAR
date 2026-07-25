#!/usr/bin/env python3
"""Build the article tables and figures from archived or regenerated results."""

from __future__ import annotations

import argparse
import shutil
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
    ("Seasonal", "seasonal_window_naive"),
    ("Euclidean AR", "euclidean_gaussian_ar"),
    ("Cholesky AR", "cholesky_gaussian_ar"),
    ("Log-Euclidean AR", "log_euclidean_gaussian_ar"),
    ("Fixed BWAR", "fixed_bwar"),
    ("Local BWAR", "local_bwar"),
]


def _format_p(value: float) -> str:
    return r"\(<0.001\)" if value < 0.001 else f"{value:.3f}"


def _format_contrast(
    row: pd.Series,
    *,
    digits: int,
    interval_digits: int | None = None,
) -> str:
    interval_digits = digits if interval_digits is None else interval_digits
    return (
        f"\\({float(row['mean_difference']):.{digits}f}\\;"
        f"[{float(row['ci_low']):.{interval_digits}f},"
        f"{float(row['ci_high']):.{interval_digits}f}]\\)"
    )


def write_inference_table(inference: pd.DataFrame, path: Path) -> None:
    primary = inference.loc[
        inference["target_method"].eq("local_bwar")
        & inference["horizon"].eq(3)
    ].copy()
    index = primary.set_index(["comparator", "metric"])
    rows = [
        ("Fixed BWAR", "fixed_bwar"),
        ("Cholesky AR", "cholesky_gaussian_ar"),
        ("Log-Euclidean AR", "log_euclidean_gaussian_ar"),
        ("Euclidean AR", "euclidean_gaussian_ar"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Paired local BWAR contrasts at \(h=3\). Entries report "
            r"local BWAR minus the comparator and a two-sided 95\% "
            r"origin-preserving moving-block bootstrap confidence interval "
            r"(CI). Negative values favor local BWAR\@. Intervals use block "
            r"length three and 10,000 replicates. Reported one-sided "
            r"\(p\)-values test for a negative expected difference and use Holm "
            r"adjustment over all six non-BWAR methods within each endpoint; "
            r"persistence, raw VAR, and seasonal rows are omitted from this "
            r"compact display. The prespecified local-versus-fixed comparison "
            r"is evaluated separately.}"
        ),
        r"\label{tab:divvy-inference}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Comparator & Raw RMSE difference (95\% CI) & \(p\) & "
            r"\(W_2^2\) difference (95\% CI) & \(p\) \\"
        ),
        r"\midrule",
    ]
    for label, comparator in rows:
        raw = index.loc[(comparator, "raw_rmse")]
        w2 = index.loc[(comparator, "w2")]
        separate = comparator == "fixed_bwar"
        raw_p = float(raw["p_value" if separate else "p_holm"])
        w2_p = float(w2["p_value" if separate else "p_holm"])
        raw_cell = _format_contrast(
            raw,
            digits=3,
            interval_digits=4 if separate else 3,
        )
        w2_cell = _format_contrast(w2, digits=3)
        lines.append(
            f"{label} & {raw_cell} & {_format_p(raw_p)} & "
            f"{w2_cell} & {_format_p(w2_p)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            r"validation block of each origin.}"
        ),
        r"\label{tab:divvy-performance}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Raw-window mean RMSE & Gaussian \(W_2^2\) loss \\",
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
    horizons = (3, 4, 6)
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
            r"Entries are target-weighted raw-window mean RMSEs over the same "
            r"five chronological rolling origins. Lower values are better.}"
        ),
        r"\label{tab:divvy-horizon}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & \(h=3\) & \(h=4\) & \(h=6\) \\",
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
    simulation.write_reference_drift_table(
        drift_summary,
        table_dir / "rolling_refit_reference_shift.tex",
        int(drift_summary["n_rep"].max()),
    )
    simulation.make_reference_drift_figure(
        drift_raw,
        drift_summary,
        figure_dir / "rolling_refit_reference_shift",
    )

    shutil.copy2(
        args.result_root / "frozen_drift" / "local_reference_shift_main.tex",
        table_dir / "local_reference_shift_main.tex",
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
    write_inference_table(
        pd.read_csv(divvy_root / "paired_inference.csv"),
        table_dir / "redone_realdata_inference.tex",
    )

    figures.FIGURE_DIR = figure_dir
    figures.SYNTH_OUT = fixed_root
    figures.REAL_OUT = divvy_root
    figures.make_synthetic_figure()
    figures.make_realdata_figure()
    print(f"Generated article artifacts: {args.artifact_root}")


if __name__ == "__main__":
    main()
