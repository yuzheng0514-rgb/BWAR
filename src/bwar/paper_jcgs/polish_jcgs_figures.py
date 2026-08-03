from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from bwar.paper_jcgs.divvy_target_level import method_level_summary


ROOT = Path(__file__).resolve().parents[3]
OVERLEAF = Path(
    os.environ.get("BWAR_OVERLEAF", ROOT / "artifacts" / "generated")
)
FIGURE_DIR = OVERLEAF / "figures"
SYNTH_OUT = Path(
    os.environ.get(
        "BWAR_SYNTH_OUT",
        ROOT / "results" / "reference" / "s1_geometry",
    )
)
REAL_OUT = Path(
    os.environ.get(
        "BWAR_REAL_OUT",
        ROOT / "results" / "reference" / "divvy",
    )
)

SYNTHETIC_METHODS = [
    "euclidean_gaussian_ar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "bwar_barycenter",
]
REAL_METHODS = [
    "persistence",
    "raw_var_window_ar",
    "euclidean_gaussian_ar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "fixed_bwar",
    "local_bwar",
]
REAL_HORIZON_METHODS = [
    "euclidean_gaussian_ar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "fixed_bwar",
    "local_bwar",
]
REAL_FOREST_COMPARATORS = [
    "fixed_bwar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "euclidean_gaussian_ar",
]
SYNTHETIC_MAIN_DISPLAY = "mean_se"

METHOD_LABEL = {
    "persistence": "Pers.",
    "raw_var_window_ar": "Raw VAR",
    "euclidean_gaussian_ar": "Euc.",
    "cholesky_gaussian_ar": "Chol.",
    "log_euclidean_gaussian_ar": "LogEuc.",
    "bwar_barycenter": "BWAR",
    "fixed_bwar": "Fixed BWAR",
    "local_bwar": "Local BWAR",
}

METHOD_COLOR = {
    "persistence": "#7F7F7F",
    "raw_var_window_ar": "#FF7F0E",
    "euclidean_gaussian_ar": "#111111",
    "cholesky_gaussian_ar": "#9467BD",
    "log_euclidean_gaussian_ar": "#1F77B4",
    "bwar_barycenter": "#D62728",
    "fixed_bwar": "#8C2D2D",
    "local_bwar": "#D62728",
}

METHOD_MARKER = {
    "persistence": "o",
    "raw_var_window_ar": "s",
    "euclidean_gaussian_ar": "o",
    "cholesky_gaussian_ar": "D",
    "log_euclidean_gaussian_ar": "v",
    "bwar_barycenter": "o",
    "fixed_bwar": "o",
    "local_bwar": "o",
}

METHOD_LINESTYLE = {
    "persistence": (0, (2, 2)),
    "raw_var_window_ar": (0, (4, 2)),
    "euclidean_gaussian_ar": "-",
    "cholesky_gaussian_ar": (0, (6, 2)),
    "log_euclidean_gaussian_ar": (0, (2, 1, 1, 1)),
    "bwar_barycenter": "-",
    "fixed_bwar": (0, (5, 2)),
    "local_bwar": "-",
}

METHOD_BOX_FACE = {
    "euclidean_gaussian_ar": "#E6E6E6",
    "cholesky_gaussian_ar": "#E9E2F2",
    "log_euclidean_gaussian_ar": "#DDEAF5",
    "bwar_barycenter": "#F6DEDE",
}

METHOD_SCATTER_COLOR = {
    "euclidean_gaussian_ar": "#111111",
    "cholesky_gaussian_ar": "#9467BD",
    "log_euclidean_gaussian_ar": "#1F77B4",
    "bwar_barycenter": "#D62728",
}


def setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-bwar")
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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
        }
    )
    return plt


def save_all(fig, path_stem: Path, *, dpi: int = 600) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = path_stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def clean_axis(ax, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#E8E8E8", lw=0.38)
    ax.set_axisbelow(True)
    ax.tick_params(pad=2.0)


def line_style(method: str) -> dict[str, object]:
    is_bwar = method in {"bwar_barycenter", "fixed_bwar", "local_bwar"}
    is_local = method == "local_bwar"
    ls = METHOD_LINESTYLE[method]
    return {
        "color": METHOD_COLOR[method],
        "marker": METHOD_MARKER[method],
        "markerfacecolor": "white" if method == "fixed_bwar" else METHOD_COLOR[method],
        "markeredgecolor": METHOD_COLOR[method],
        "markeredgewidth": 0.75,
        "lw": 1.65 if is_local else (1.15 if is_bwar else 0.85),
        "ms": 3.5 if is_local else (3.2 if is_bwar else 2.8),
        "capsize": 1.6,
        "elinewidth": 0.65,
        "alpha": 1.0,
        "linestyle": ls,
        "zorder": 5 if is_local else (4 if is_bwar else 3),
    }


def parameter_variation_style(method: str) -> dict[str, object]:
    """Style heterogeneous stress settings without implying an ordered path."""
    style = line_style(method)
    style["linestyle"] = "none"
    return style


def plot_reference_line(ax, value: float = 1.0, *, axis: str = "y") -> None:
    if axis == "y":
        ax.axhline(value, color="#8F8F8F", lw=0.6, ls=(0, (4, 2)), zorder=1)
    else:
        ax.axvline(value, color="#8F8F8F", lw=0.6, ls=(0, (4, 2)), zorder=1)


def padded_limits(values: pd.Series, errors: pd.Series | None = None, *, lower_floor: float | None = None) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if errors is not None:
        err = pd.to_numeric(errors, errors="coerce").fillna(0.0).to_numpy(float)
        lo = np.nanmin(vals - err)
        hi = np.nanmax(vals + err)
    else:
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    span = max(hi - lo, 1e-6)
    lo = lo - 0.08 * span
    hi = hi + 0.10 * span
    if lower_floor is not None:
        lo = min(lo, lower_floor) if lo > lower_floor else lo
    return float(lo), float(hi)


def make_synthetic_figure() -> None:
    plt = setup_matplotlib()
    summary = pd.read_csv(SYNTH_OUT / "strong_synthetic_transport_summary.csv")

    fig = plt.figure(figsize=(7.05, 4.65))
    fig.patch.set_edgecolor("#FFFFFF")
    fig.patch.set_linewidth(0.3)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.96, 1.10], hspace=0.54, wspace=0.32)
    ax_rep = fig.add_subplot(gs[0, 0])
    ax_cov = fig.add_subplot(gs[0, 1])
    ax_var = fig.add_subplot(gs[1, :])
    for ax in (ax_rep, ax_cov, ax_var):
        ax.patch.set_edgecolor("#FFFFFF")
        ax.patch.set_linewidth(0.3)

    baseline = summary.loc[
        summary["design"].eq("Baseline")
        & summary["method"].isin(SYNTHETIC_METHODS)
    ].copy()
    baseline["order"] = baseline["method"].map(
        {method: index for index, method in enumerate(SYNTHETIC_METHODS)}
    )
    baseline = baseline.sort_values("order")
    positions = np.arange(len(SYNTHETIC_METHODS))
    for position, (_, row) in zip(positions, baseline.iterrows(), strict=True):
        method = str(row["method"])
        ax_rep.errorbar(
            position,
            float(row["w2_ratio_mean"]),
            yerr=float(row["w2_ratio_se"]),
            fmt=METHOD_MARKER[method],
            color=METHOD_COLOR[method],
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor="white",
            markeredgewidth=0.35,
            ms=4.2 if method == "bwar_barycenter" else 3.7,
            elinewidth=0.8,
            capsize=2.0,
            zorder=4 if method == "bwar_barycenter" else 3,
        )
    plot_reference_line(ax_rep)
    ax_rep.set_xticks(positions)
    ax_rep.set_xticklabels([METHOD_LABEL[m] for m in SYNTHETIC_METHODS])
    ax_rep.set_ylabel(r"Full $W_2^2$ ratio")
    ax_rep.set_title("Main setting")
    ax_rep.set_ylim(0.86, 1.025)
    clean_axis(ax_rep)

    cov = baseline.copy()
    y = np.arange(len(cov))
    for yi, (_, row) in zip(y, cov.iterrows(), strict=False):
        method = str(row["method"])
        ax_cov.errorbar(
            float(row["cov_ratio_mean"]),
            yi,
            xerr=float(row["cov_ratio_se"]),
            fmt=METHOD_MARKER[method],
            color=METHOD_COLOR[method],
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor="white",
            markeredgewidth=0.35,
            ms=4.0 if method == "bwar_barycenter" else 3.5,
            lw=0,
            elinewidth=0.8,
            capsize=2.0,
            zorder=3,
        )
    plot_reference_line(ax_cov, axis="x")
    ax_cov.set_yticks(y)
    ax_cov.set_yticklabels([METHOD_LABEL[m] for m in cov["method"]])
    ax_cov.invert_yaxis()
    ax_cov.set_xlabel("Covariance loss ratio")
    ax_cov.set_title("Covariance component")
    ax_cov.set_xlim(0.86, 1.03)
    clean_axis(ax_cov, grid_axis="x")

    designs = ["Baseline", "Shorter series", "Higher dimension", "Weaker dynamics", "Larger variation"]
    x = np.arange(len(designs))
    offsets = np.linspace(-0.14, 0.14, len(SYNTHETIC_METHODS))
    for off, method in zip(offsets, SYNTHETIC_METHODS, strict=True):
        part = summary.loc[
            summary["method"].eq(method) & summary["design"].isin(designs)
        ].copy()
        part["design"] = pd.Categorical(part["design"], categories=designs, ordered=True)
        part = part.sort_values("design")
        ax_var.errorbar(
            x + off,
            part["w2_ratio_mean"].astype(float),
            yerr=part["w2_ratio_se"].astype(float),
            label=METHOD_LABEL[method],
            **parameter_variation_style(method),
        )
    plot_reference_line(ax_var)
    ax_var.set_xticks(x)
    ax_var.set_xticklabels(["Baseline", "Shorter\nseries", "Higher\ndimension", "Weaker\ndynamics", "Larger\nvariation"])
    ax_var.set_ylabel(r"Full $W_2^2$ ratio")
    ax_var.set_title("Parameter variation")
    ax_var.set_ylim(0.77, 1.035)
    clean_axis(ax_var)
    ax_var.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=4, handlelength=2.0, columnspacing=1.1)

    for label, ax in zip("abc", [ax_rep, ax_cov, ax_var], strict=True):
        panel_label(ax, label)
    fig.subplots_adjust(bottom=0.15, top=0.95)
    save_all(fig, FIGURE_DIR / "synthetic_transport_mechanism")
    plt.close(fig)


def make_realdata_figure() -> None:
    plt = setup_matplotlib()
    panel = pd.read_csv(REAL_OUT / "target_level_losses.csv")
    inference = pd.read_csv(REAL_OUT / "paired_inference.csv")
    summary_path = REAL_OUT / "method_level_bootstrap_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = method_level_summary(
            panel,
            methods=tuple(REAL_METHODS),
            block_length=3,
            replicates=10_000,
            seed=20_260_714,
        )
    horizons = sorted(int(h) for h in summary["horizon"].unique())
    main_h = 3

    fig = plt.figure(figsize=(7.05, 4.95))
    fig.patch.set_facecolor("#FFFFFF")
    fig.patch.set_edgecolor("#FFFFFF")
    fig.patch.set_linewidth(0.3)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.54, wspace=0.32)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_w2 = fig.add_subplot(gs[0, 1])
    ax_raw_diff = fig.add_subplot(gs[1, 0])
    ax_w2_diff = fig.add_subplot(gs[1, 1])
    for ax in (ax_raw, ax_w2, ax_raw_diff, ax_w2_diff):
        ax.patch.set_facecolor("#FFFFFF")
        ax.patch.set_edgecolor("#FFFFFF")
        ax.patch.set_linewidth(0.3)

    for ax, metric, ylabel, title in [
        (ax_raw, "raw_rmse", "Standardized mean RMSE", "Physical endpoint"),
        (ax_w2, "w2", r"Gaussian $W_2^2$ loss", "Distributional endpoint"),
    ]:
        metric_summary = summary.loc[summary["metric"].eq(metric)]
        for method in REAL_HORIZON_METHODS:
            part = metric_summary.loc[
                metric_summary["method"].eq(method)
            ].sort_values("horizon")
            values = part["mean"].to_numpy(dtype=float)
            lower = values - part["ci_low"].to_numpy(dtype=float)
            upper = part["ci_high"].to_numpy(dtype=float) - values
            ax.errorbar(
                part["horizon"].astype(int),
                values,
                yerr=np.vstack([lower, upper]),
                label=METHOD_LABEL[method],
                **line_style(method),
            )
        ax.set_xticks(horizons)
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        clean_axis(ax)
    displayed = summary.loc[summary["method"].isin(REAL_HORIZON_METHODS)]
    raw_displayed = displayed.loc[displayed["metric"].eq("raw_rmse")]
    w2_displayed = displayed.loc[displayed["metric"].eq("w2")]
    ax_raw.set_ylim(
        *padded_limits(
            pd.concat([raw_displayed["ci_low"], raw_displayed["ci_high"]]),
            lower_floor=0.0,
        )
    )
    ax_w2.set_ylim(
        *padded_limits(pd.concat([w2_displayed["ci_low"], w2_displayed["ci_high"]]))
    )

    for ax, metric, xlabel, title in [
        (
            ax_raw_diff,
            "raw_rmse",
            r"Difference in standardized mean RMSE",
            rf"Local BWAR contrasts at $h={main_h}$",
        ),
        (
            ax_w2_diff,
            "w2",
            r"Difference in Gaussian $W_2^2$ loss",
            rf"Local BWAR contrasts at $h={main_h}$",
        ),
    ]:
        contrasts = inference.loc[
            inference["target_method"].eq("local_bwar")
            & inference["horizon"].eq(main_h)
            & inference["metric"].eq(metric)
            & inference["comparator"].isin(REAL_FOREST_COMPARATORS)
        ].copy()
        contrasts["order"] = contrasts["comparator"].map(
            {method: index for index, method in enumerate(REAL_FOREST_COMPARATORS)}
        )
        contrasts = contrasts.sort_values("order")
        y = np.arange(len(contrasts))
        values = contrasts["mean_difference"].to_numpy(dtype=float)
        lower = values - contrasts["ci_low"].to_numpy(dtype=float)
        upper = contrasts["ci_high"].to_numpy(dtype=float) - values
        for yi, value, lo, hi, comparator in zip(
            y,
            values,
            lower,
            upper,
            contrasts["comparator"],
            strict=True,
        ):
            ax.errorbar(
                value,
                yi,
                xerr=np.asarray([[lo], [hi]]),
                fmt="o",
                color="#D62728",
                markerfacecolor="#D62728",
                markeredgecolor="#D62728",
                ms=3.5,
                elinewidth=0.9,
                capsize=2.0,
                zorder=3,
            )
        ax.axvline(0.0, color="#777777", lw=0.65, ls=(0, (4, 2)), zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABEL[m] for m in contrasts["comparator"]])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel + " (local $-$ comparator)")
        ax.set_title(title)
        x_low = min(float(contrasts["ci_low"].min()), 0.0)
        x_high = max(float(contrasts["ci_high"].max()), 0.0)
        span = max(x_high - x_low, 1e-6)
        ax.set_xlim(x_low - 0.08 * span, x_high + 0.08 * span)
        clean_axis(ax, grid_axis="x")

    for label, ax in zip(
        "abcd", [ax_raw, ax_w2, ax_raw_diff, ax_w2_diff], strict=True
    ):
        panel_label(ax, label)
    handles, labels = ax_raw.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.01),
        ncol=5,
        handlelength=1.8,
        columnspacing=1.05,
    )
    fig.subplots_adjust(bottom=0.15, top=0.95)
    save_all(fig, FIGURE_DIR / "redone_realdata_application")
    plt.close(fig)


def make_representative_forecast_figure() -> None:
    """Show the domain object for the prespecified first held-out Divvy target."""
    plt = setup_matplotlib()
    means = pd.read_csv(REAL_OUT / "representative_forecast_means.csv")
    correlations = pd.read_csv(
        REAL_OUT / "representative_forecast_correlations.csv"
    )
    methods = ["observed", "cholesky_gaussian_ar", "bwar_barycenter"]
    labels = {
        "observed": "Observed",
        "cholesky_gaussian_ar": "Chol.",
        "bwar_barycenter": "BWAR",
    }
    colors = {
        "observed": "#111111",
        "cholesky_gaussian_ar": METHOD_COLOR["cholesky_gaussian_ar"],
        "bwar_barycenter": METHOD_COLOR["bwar_barycenter"],
    }
    markers = {
        "observed": "o",
        "cholesky_gaussian_ar": METHOD_MARKER["cholesky_gaussian_ar"],
        "bwar_barycenter": METHOD_MARKER["bwar_barycenter"],
    }

    fig = plt.figure(figsize=(7.05, 4.45))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[0.82, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        hspace=0.43,
        wspace=0.16,
    )
    ax_mean = fig.add_subplot(gs[0, :3])
    heat_axes = [fig.add_subplot(gs[1, index]) for index in range(3)]
    cax = fig.add_subplot(gs[1, 3])

    for method in methods:
        part = means.loc[means["method"].eq(method)].sort_values("station_rank")
        ax_mean.plot(
            part["station_rank"].astype(int),
            part["mean_count"].astype(float),
            color=colors[method],
            marker=markers[method],
            ms=3.0 if method == "bwar_barycenter" else 2.5,
            lw=1.45 if method == "bwar_barycenter" else 0.9,
            linestyle="-" if method != "cholesky_gaussian_ar" else (0, (5, 2)),
            label=labels[method],
            zorder=4 if method == "bwar_barycenter" else 3,
        )
    station_count = int(means["station_rank"].max())
    ticks = np.unique(np.linspace(1, station_count, min(7, station_count), dtype=int))
    ax_mean.set_xticks(ticks)
    ax_mean.set_xlabel("Station rank (training-only activity order)")
    ax_mean.set_ylabel("Mean hourly trips")
    ax_mean.set_title("First target in the first held-out block")
    clean_axis(ax_mean)
    ax_mean.legend(loc="upper right", ncol=3, handlelength=2.1)
    panel_label(ax_mean, "a")

    image = None
    for index, (ax, method) in enumerate(zip(heat_axes, methods, strict=True)):
        part = correlations.loc[correlations["method"].eq(method)]
        matrix = part.pivot(index="row", columns="column", values="correlation")
        image = ax.imshow(
            matrix.to_numpy(float),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(labels[method])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#B8B8B8")
            spine.set_linewidth(0.45)
        panel_label(ax, chr(ord("b") + index))
    if image is None:
        raise ValueError("representative forecast contains no correlation matrices")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Station correlation", labelpad=3)
    colorbar.set_ticks([-1.0, 0.0, 1.0])
    fig.subplots_adjust(top=0.94, bottom=0.11, left=0.09, right=0.94)
    save_all(fig, FIGURE_DIR / "divvy_representative_forecast")
    plt.close(fig)


def main() -> None:
    make_synthetic_figure()
    make_realdata_figure()
    if (REAL_OUT / "representative_forecast_means.csv").exists():
        make_representative_forecast_figure()


if __name__ == "__main__":
    main()
