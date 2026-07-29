#!/usr/bin/env python3
"""Build the descriptive Divvy data-background figure used in the article."""

from __future__ import annotations

import argparse
import calendar
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bwar.paper_jcgs.divvy_data import divvy_matrix, load_divvy_hourly
from bwar.paper_jcgs.run_divvy_analysis import (
    DIMENSION,
    MAX_MATRICES,
    MONTHS,
    STEP,
    WINDOW,
)


BLUE = "#0F4D92"
BLUE_LIGHT = "#B9D3EA"
GRID = "#E8E8E8"
TEXT = "#272727"


def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-bwar")
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.9,
            "axes.labelsize": 6.9,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )
    return plt


def _panel_label(ax, label: str, *, x: float = -0.09, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _load_selected_counts() -> pd.DataFrame:
    hourly = load_divvy_hourly(MONTHS)
    matrix, station_ids = divvy_matrix(
        DIMENSION,
        months=MONTHS,
        window=WINDOW,
        step=STEP,
        max_matrices=MAX_MATRICES,
        return_columns=True,
    )
    selected = pd.DataFrame(matrix, index=hourly.index, columns=station_ids)
    if selected.shape != (len(hourly), DIMENSION):
        raise ValueError(
            f"unexpected selected Divvy matrix shape: {selected.shape}"
        )
    if not np.isfinite(selected.to_numpy(float)).all():
        raise ValueError("selected Divvy matrix contains nonfinite values")
    return selected


def _build_source_data(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    daily_total = selected.sum(axis=1).resample("D").sum().rename("daily_departures")
    daily = daily_total.to_frame()
    daily["rolling_7d_departures"] = daily_total.rolling(
        7, center=True, min_periods=1
    ).mean()
    daily = daily.reset_index(names="date")

    weekly_means = selected.resample("7D", origin="start_day").mean()
    station_center = weekly_means.mean(axis=0)
    station_scale = weekly_means.std(axis=0, ddof=0).replace(0.0, 1.0)
    weekly_z = (weekly_means - station_center) / station_scale
    weekly_rows: list[dict[str, object]] = []
    for station_rank, station_id in enumerate(selected.columns, start=1):
        for week_start in weekly_means.index:
            weekly_rows.append(
                {
                    "week_start": week_start,
                    "station_rank": station_rank,
                    "station_id": station_id,
                    "mean_hourly_departures": float(
                        weekly_means.loc[week_start, station_id]
                    ),
                    "within_station_z": float(weekly_z.loc[week_start, station_id]),
                }
            )
    weekly = pd.DataFrame(weekly_rows)

    monthly_correlations: list[np.ndarray] = []
    for month in range(1, 13):
        month_frame = selected.loc[selected.index.month == month]
        correlation = np.nan_to_num(
            month_frame.corr().to_numpy(float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.fill_diagonal(correlation, 1.0)
        monthly_correlations.append(correlation)

    denominator = np.sqrt(DIMENSION * (DIMENSION - 1))
    distance_matrix = np.zeros((12, 12), dtype=float)
    distance_rows: list[dict[str, object]] = []
    for month_a in range(12):
        for month_b in range(12):
            distance = float(
                np.linalg.norm(
                    monthly_correlations[month_a]
                    - monthly_correlations[month_b],
                    ord="fro",
                )
                / denominator
            )
            distance_matrix[month_a, month_b] = distance
            distance_rows.append(
                {
                    "month_1": month_a + 1,
                    "month_2": month_b + 1,
                    "month_1_label": calendar.month_abbr[month_a + 1],
                    "month_2_label": calendar.month_abbr[month_b + 1],
                    "normalized_frobenius_distance": distance,
                }
            )
    distances = pd.DataFrame(distance_rows)
    return daily, weekly, distances, distance_matrix


def _save_source_data(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    distances: pd.DataFrame,
    source_root: Path,
) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    daily.to_csv(source_root / "divvy_daily_total.csv", index=False)
    weekly.to_csv(source_root / "divvy_weekly_station_profiles.csv", index=False)
    distances.to_csv(
        source_root / "divvy_monthly_correlation_distance.csv", index=False
    )


def _build_figure(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    distance_matrix: np.ndarray,
    figure_stem: Path,
) -> None:
    plt = _setup_matplotlib()
    import matplotlib.dates as mdates

    weekly_dates = (
        weekly[["week_start", "station_rank"]]
        .drop_duplicates()
        .sort_values(["station_rank", "week_start"])["week_start"]
        .drop_duplicates()
    )
    weekly_matrix = (
        weekly.pivot(
            index="station_rank", columns="week_start", values="within_station_z"
        )
        .sort_index()
        .to_numpy(float)
    )

    fig = plt.figure(figsize=(7.05, 4.75))
    grid = fig.add_gridspec(
        2,
        5,
        height_ratios=[0.78, 1.65],
        width_ratios=[1.0, 1.0, 1.0, 0.86, 0.86],
        hspace=0.48,
        wspace=0.62,
    )
    ax_daily = fig.add_subplot(grid[0, :])
    ax_station = fig.add_subplot(grid[1, :3])
    ax_dependence = fig.add_subplot(grid[1, 3:])
    fig.subplots_adjust(
        left=0.075,
        right=0.88,
        top=0.95,
        bottom=0.22,
    )

    dates = pd.to_datetime(daily["date"])
    daily_values = daily["daily_departures"].to_numpy(float)
    rolling_values = daily["rolling_7d_departures"].to_numpy(float)
    ax_daily.fill_between(
        dates,
        0.0,
        daily_values,
        color=BLUE_LIGHT,
        alpha=0.42,
        linewidth=0.0,
        label="Daily total",
    )
    ax_daily.plot(
        dates,
        daily_values,
        color="#8AAFCF",
        lw=0.45,
        alpha=0.7,
    )
    ax_daily.plot(
        dates,
        rolling_values,
        color=BLUE,
        lw=1.35,
        label="7-day mean",
        zorder=4,
    )
    ax_daily.set_ylabel("Daily departures")
    ax_daily.set_title("System-level activity", loc="left", fontweight="bold")
    ax_daily.set_xlim(dates.min(), dates.max())
    ax_daily.set_ylim(bottom=0.0)
    ax_daily.xaxis.set_major_locator(mdates.MonthLocator())
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_daily.grid(axis="y", color=GRID, lw=0.38)
    ax_daily.set_axisbelow(True)
    ax_daily.legend(
        loc="upper left",
        ncol=2,
        handlelength=1.7,
        columnspacing=1.0,
    )
    _panel_label(ax_daily, "a", x=-0.055)

    station_image = ax_station.imshow(
        weekly_matrix,
        cmap="RdBu_r",
        vmin=-2.2,
        vmax=2.2,
        interpolation="nearest",
        aspect="auto",
    )
    ax_station.set_title(
        "Station-specific temporal profiles", loc="left", fontweight="bold"
    )
    ax_station.set_ylabel("Station rank")
    station_ticks = np.array([1, 5, 10, 15, 20, 25, 30])
    ax_station.set_yticks(station_ticks - 1)
    ax_station.set_yticklabels(station_ticks)
    month_starts = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    week_values = pd.to_datetime(weekly_dates).to_numpy()
    month_positions = [
        int(np.argmin(np.abs(week_values - np.datetime64(month_start))))
        for month_start in month_starts
    ]
    ax_station.set_xticks(month_positions)
    ax_station.set_xticklabels(
        [calendar.month_abbr[month] for month in range(1, 13)],
        rotation=0,
    )
    ax_station.set_xlabel("Week in 2024")
    ax_station.tick_params(length=0)
    for spine in ax_station.spines.values():
        spine.set_visible(False)
    # Use a figure-coordinate axes so the horizontal colorbar's QuadMesh,
    # outline, and ticks share one transform in vector PDF output.
    station_position = ax_station.get_position()
    station_cax = fig.add_axes(
        [
            station_position.x0 + 0.19 * station_position.width,
            station_position.y0 - 0.27 * station_position.height,
            0.62 * station_position.width,
            0.045 * station_position.height,
        ]
    )
    station_colorbar = fig.colorbar(
        station_image,
        cax=station_cax,
        orientation="horizontal",
        ticks=[-2.0, 0.0, 2.0],
    )
    station_colorbar.set_label("Within-station standardized demand", labelpad=1.5)
    station_colorbar.outline.set_linewidth(0.4)
    _panel_label(ax_station, "b", x=-0.09)

    dependence_image = ax_dependence.imshow(
        distance_matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=float(np.max(distance_matrix)),
        interpolation="nearest",
        aspect="equal",
    )
    month_labels = [calendar.month_abbr[month] for month in range(1, 13)]
    ax_dependence.set_title(
        "Monthly dependence change", loc="left", fontweight="bold"
    )
    ax_dependence.set_xticks(range(12))
    ax_dependence.set_xticklabels(month_labels, rotation=45, ha="right")
    ax_dependence.set_yticks(range(12))
    ax_dependence.set_yticklabels(month_labels)
    ax_dependence.tick_params(length=0, pad=1.0)
    for spine in ax_dependence.spines.values():
        spine.set_visible(False)
    dependence_colorbar = fig.colorbar(
        dependence_image,
        ax=ax_dependence,
        orientation="vertical",
        fraction=0.05,
        pad=0.04,
        shrink=0.78,
    )
    dependence_colorbar.set_label("Correlation distance", labelpad=2.0)
    dependence_colorbar.outline.set_linewidth(0.4)
    _panel_label(ax_dependence, "c", x=-0.12)

    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_stem.with_suffix(".pdf"))
    fig.savefig(figure_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(figure_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the descriptive Divvy data-background figure."
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=ROOT / "artifacts" / "generated" / "figures",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "artifacts" / "generated" / "source_data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = _load_selected_counts()
    daily, weekly, distances, distance_matrix = _build_source_data(selected)
    _save_source_data(daily, weekly, distances, args.source_root)
    stem = args.figure_root / "divvy_data_background"
    _build_figure(daily, weekly, distance_matrix, stem)
    print(f"Figure: {stem.with_suffix('.pdf')}")
    print(f"Source data: {args.source_root}")


if __name__ == "__main__":
    main()
