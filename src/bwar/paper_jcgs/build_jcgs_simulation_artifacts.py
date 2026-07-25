from __future__ import annotations

import argparse
from collections.abc import Callable
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.bwar_experiments import (  # noqa: E402
    bw_barycenter,
    mat_exp,
    mat_from_triu,
    mat_log,
    project_spd,
    triu_vec,
)
from bwar.paper_jcgs.build_strong_synthetic_artifacts import (  # noqa: E402
    DEFAULT_RIDGE_GRID,
    bwar_decode,
    bwar_encode,
    cholesky_decode,
    cholesky_encode,
    fit_var,
    gaussian_w2_squared,
    random_reference,
    recursive_predict,
    run as run_fixed_reference,
    safe_transport_from_vech,
    write_main_table,
    write_variation_table,
)
from bwar.paper_jcgs.local_reference_bwar import (  # noqa: E402
    build_local_bwar_geometry,
    local_bwar_decode,
)


FIXED_OUT_DIR = ROOT / "results" / "generated" / "fixed_simulation"
DRIFT_OUT_DIR = ROOT / "results" / "generated" / "rolling_drift"
DEFAULT_TARGET_DIR = ROOT / "artifacts" / "generated"

DRIFT_SETTINGS = (
    "No shift",
    "Mean shift",
    "Covariance shift",
    "Joint shift",
    "Gradual joint shift",
)
DRIFT_METHOD_ORDER = (
    "persistence",
    "euclidean",
    "cholesky",
    "log_euclidean",
    "fixed_bwar",
    "local_bwar",
)
METHOD_LABEL = {
    "persistence": "Persistence",
    "euclidean": "Euclidean AR",
    "cholesky": "Cholesky AR",
    "log_euclidean": "Log-Euclidean AR",
    "fixed_bwar": "Fixed BWAR",
    "local_bwar": "Local BWAR",
    "euclidean_gaussian_ar": "Euclidean AR",
    "cholesky_gaussian_ar": "Cholesky AR",
    "log_euclidean_gaussian_ar": "Log-Euclidean AR",
    "bwar_barycenter": "BWAR",
}

_FIXED_PLOT_METHODS = (
    "euclidean_gaussian_ar",
    "cholesky_gaussian_ar",
    "log_euclidean_gaussian_ar",
    "bwar_barycenter",
)
_COLOR = {
    "euclidean": "#111111",
    "cholesky": "#9467BD",
    "log_euclidean": "#17BECF",
    "fixed_bwar": "#1F77B4",
    "local_bwar": "#D62728",
    "euclidean_gaussian_ar": "#111111",
    "cholesky_gaussian_ar": "#9467BD",
    "log_euclidean_gaussian_ar": "#1F77B4",
    "bwar_barycenter": "#D62728",
}
_MARKER = {
    "euclidean": "o",
    "cholesky": "s",
    "log_euclidean": "^",
    "fixed_bwar": "D",
    "local_bwar": "o",
    "euclidean_gaussian_ar": "o",
    "cholesky_gaussian_ar": "s",
    "log_euclidean_gaussian_ar": "^",
    "bwar_barycenter": "o",
}
_LINESTYLE = {
    "euclidean": "-",
    "cholesky": (0, (5, 2)),
    "log_euclidean": (0, (1, 1.5)),
    "fixed_bwar": (0, (4, 1.5)),
    "local_bwar": "-",
    "euclidean_gaussian_ar": "-",
    "cholesky_gaussian_ar": (0, (5, 2)),
    "log_euclidean_gaussian_ar": (0, (1, 1.5)),
    "bwar_barycenter": "-",
}


def _standard_error(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else np.nan


def _split_boundaries(
    n: int,
    *,
    fit_fraction: float,
    validation_fraction: float,
) -> tuple[int, int]:
    if n < 30:
        raise ValueError("n must be at least 30")
    if not 0.0 < fit_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must lie strictly between zero and one")
    if fit_fraction + validation_fraction >= 0.9:
        raise ValueError("fit and validation fractions must leave a test block")
    fit_end = int(round(fit_fraction * n))
    validation_end = fit_end + int(round(validation_fraction * n))
    if fit_end < 10 or validation_end >= n - 5:
        raise ValueError("split fractions produce invalid chronological blocks")
    return fit_end, validation_end


def _shift_path(setting: str, n: int, fit_end: int) -> np.ndarray:
    if setting not in DRIFT_SETTINGS:
        raise ValueError(f"unknown reference-drift setting: {setting}")
    eta = np.zeros(n, dtype=float)
    if setting == "No shift":
        return eta
    if setting == "Gradual joint shift":
        eta[fit_end:] = np.linspace(0.0, 1.0, n - fit_end)
    else:
        eta[fit_end:] = 1.0
    return eta


def simulate_reference_drift_gaussians(
    *,
    setting: str,
    n: int,
    d: int,
    fit_end: int,
    phi: float,
    dispersion: float,
    seed: int,
    mean_shift_strength: float = 0.35,
    covariance_shift_strength: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Generate Gaussian states from transport dynamics around a moving reference."""

    if d < 2 or fit_end < 10 or fit_end >= n:
        raise ValueError("invalid drift simulation dimensions or fit boundary")
    if not 0.0 < phi < 1.0 or dispersion <= 0.0:
        raise ValueError("phi and dispersion must define a stable positive design")
    eta = _shift_path(setting, n, fit_end)
    mean_active = setting in {"Mean shift", "Joint shift", "Gradual joint shift"}
    covariance_active = setting in {
        "Covariance shift",
        "Joint shift",
        "Gradual joint shift",
    }

    rng = np.random.default_rng(38471 + int(seed))
    base_mean, base_covariance = random_reference(rng, d, condition=6.0)
    mean_direction = rng.normal(size=d)
    mean_direction /= max(float(np.linalg.norm(mean_direction)), 1e-12)
    raw_direction = rng.normal(size=(d, d))
    covariance_direction = 0.5 * (raw_direction + raw_direction.T)
    covariance_direction -= np.trace(covariance_direction) / d * np.eye(d)
    direction_scale = max(
        float(np.abs(np.linalg.eigvalsh(covariance_direction)).max()),
        1e-12,
    )
    covariance_direction /= direction_scale

    covariance_dimension = d * (d + 1) // 2
    coordinate_dimension = d + covariance_dimension
    slopes = np.zeros(coordinate_dimension, dtype=float)
    active_mean = rng.choice(d, size=max(2, int(np.ceil(0.4 * d))), replace=False)
    active_covariance = rng.choice(
        covariance_dimension,
        size=max(4, int(np.ceil(0.35 * covariance_dimension))),
        replace=False,
    )
    active = np.r_[active_mean, d + active_covariance]
    slopes[active] = np.clip(
        phi * rng.uniform(0.985, 1.015, size=len(active)),
        0.05,
        0.985,
    )
    stationary_scale = np.full(coordinate_dimension, 0.002, dtype=float)
    stationary_scale[active_mean] = 0.04
    stationary_scale[d + active_covariance] = dispersion
    innovation_scale = stationary_scale * np.sqrt(
        np.maximum(1.0 - slopes**2, 0.05)
    )
    coordinates = np.zeros((n, coordinate_dimension), dtype=float)
    coordinates[0] = rng.normal(scale=stationary_scale)
    for index in range(1, n):
        coordinates[index] = (
            slopes * coordinates[index - 1]
            + rng.normal(scale=innovation_scale)
        )

    means = np.empty((n, d), dtype=float)
    covariances = np.empty((n, d, d), dtype=float)
    reference_means = np.empty((n, d), dtype=float)
    reference_covariances = np.empty((n, d, d), dtype=float)
    for index in range(n):
        reference_mean = base_mean.copy()
        if mean_active:
            reference_mean += (
                mean_shift_strength * eta[index] * mean_direction
            )
        if covariance_active:
            deformation = mat_exp(
                covariance_shift_strength * eta[index] * covariance_direction
            )
            reference_covariance = project_spd(
                deformation @ base_covariance @ deformation,
                eps=1e-8,
            )
        else:
            reference_covariance = base_covariance
        transport = safe_transport_from_vech(
            coordinates[index, d:],
            d,
            min_transport_eig=0.12,
        )
        means[index] = reference_mean + coordinates[index, :d]
        covariances[index] = project_spd(
            transport @ reference_covariance @ transport,
            eps=1e-8,
        )
        reference_means[index] = reference_mean
        reference_covariances[index] = reference_covariance

    metadata = {
        "setting": setting,
        "mean_shift_strength": float(mean_shift_strength),
        "covariance_shift_strength": float(covariance_shift_strength),
        "coordinate_dimension": int(coordinate_dimension),
        "reference_condition": 6.0,
    }
    return means, covariances, reference_means, reference_covariances, metadata


def _fixed_specs(
    means: np.ndarray,
    covariances: np.ndarray,
    fit_end: int,
) -> dict[str, tuple[np.ndarray, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]]:
    dimension = means.shape[1]
    reference_mean = means[:fit_end].mean(axis=0)
    reference_covariance = bw_barycenter(covariances[:fit_end])
    specs: dict[
        str,
        tuple[
            Callable[[np.ndarray, np.ndarray], np.ndarray],
            Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        ],
    ] = {
        "euclidean": (
            lambda mean, covariance: np.r_[mean, triu_vec(covariance)],
            lambda coordinate: (
                np.asarray(coordinate[:dimension]),
                project_spd(
                    mat_from_triu(np.asarray(coordinate[dimension:]), dimension),
                    eps=1e-8,
                ),
            ),
        ),
        "cholesky": (
            cholesky_encode,
            lambda coordinate: cholesky_decode(coordinate, dimension),
        ),
        "log_euclidean": (
            lambda mean, covariance: np.r_[mean, triu_vec(mat_log(covariance))],
            lambda coordinate: (
                np.asarray(coordinate[:dimension]),
                mat_exp(
                    mat_from_triu(np.asarray(coordinate[dimension:]), dimension)
                ),
            ),
        ),
        "fixed_bwar": (
            lambda mean, covariance: bwar_encode(
                mean,
                covariance,
                reference_mean,
                reference_covariance,
            ),
            lambda coordinate: bwar_decode(
                coordinate,
                reference_mean,
                reference_covariance,
            ),
        ),
    }
    return {
        method: (
            np.vstack(
                [encoder(mean, covariance) for mean, covariance in zip(means, covariances)]
            ),
            decoder,
        )
        for method, (encoder, decoder) in specs.items()
    }


def _score_forecasts(
    means: np.ndarray,
    covariances: np.ndarray,
    sources: np.ndarray,
    forecast: Callable[[int], tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    losses: list[float] = []
    minimum_eigenvalues: list[float] = []
    for source in sources:
        predicted_mean, predicted_covariance = forecast(int(source))
        predicted_covariance = project_spd(predicted_covariance, eps=1e-8)
        loss, _, _ = gaussian_w2_squared(
            predicted_mean,
            predicted_covariance,
            means[source + 1],
            covariances[source + 1],
        )
        losses.append(float(loss))
        minimum_eigenvalues.append(
            float(np.linalg.eigvalsh(predicted_covariance).min())
        )
    return {
        "w2_mean": float(np.mean(losses)),
        "n_origins": int(len(losses)),
        "min_pred_eig": float(np.min(minimum_eigenvalues)),
    }


def _encoded_forecast(
    coordinates: np.ndarray,
    decoder: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    source: int,
    window_length: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = coordinates[source - window_length + 1 : source + 1]
    coefficients = fit_var(active, len(active), lam=ridge, model="diag")
    return decoder(recursive_predict(active[-1], coefficients, horizon=1))


def _local_forecast(
    geometry: object,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = geometry.coordinates
    coefficients = fit_var(coordinates, len(coordinates), lam=ridge, model="diag")
    prediction = recursive_predict(coordinates[-1], coefficients, horizon=1)
    return local_bwar_decode(
        prediction,
        geometry.reference.mean,
        geometry.reference.cov,
    )


def run_reference_drift_setting(
    *,
    setting: str,
    seed: int,
    n: int = 260,
    d: int = 5,
    phi: float = 0.70,
    dispersion: float = 0.16,
    window_length: int = 36,
    fit_fraction: float = 0.45,
    validation_fraction: float = 0.20,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    k_ref: int = 3,
    refresh_period: int = 12,
    residual_threshold: float = 1e-4,
    mean_shift_strength: float = 0.35,
    covariance_shift_strength: float = 0.80,
) -> pd.DataFrame:
    fit_end, validation_end = _split_boundaries(
        n,
        fit_fraction=fit_fraction,
        validation_fraction=validation_fraction,
    )
    if window_length < 3 or window_length > fit_end:
        raise ValueError("window_length must fit inside the fitting block")
    means, covariances, _, _, metadata = simulate_reference_drift_gaussians(
        setting=setting,
        n=n,
        d=d,
        fit_end=fit_end,
        phi=phi,
        dispersion=dispersion,
        seed=seed,
        mean_shift_strength=mean_shift_strength,
        covariance_shift_strength=covariance_shift_strength,
    )
    validation_sources = np.arange(
        max(window_length - 1, fit_end - 1),
        validation_end - 1,
        dtype=int,
    )
    test_sources = np.arange(
        max(window_length - 1, validation_end - 1),
        n - 1,
        dtype=int,
    )
    if len(validation_sources) == 0 or len(test_sources) == 0:
        raise ValueError("drift split must provide validation and test origins")
    all_sources = np.r_[validation_sources, test_sources]
    local_geometry = build_local_bwar_geometry(
        means,
        covariances,
        window_length=window_length,
        source_indices=all_sources,
        k_ref=k_ref,
        refresh_period=refresh_period,
        residual_threshold=residual_threshold,
    )
    fixed_specs = _fixed_specs(means, covariances, fit_end)

    persistence = _score_forecasts(
        means,
        covariances,
        test_sources,
        lambda source: (means[source], covariances[source]),
    )
    rows: list[dict[str, object]] = [
        {
            "setting": setting,
            "seed": int(seed),
            "n": int(n),
            "d": int(d),
            "phi": float(phi),
            "dispersion": float(dispersion),
            "window_length": int(window_length),
            "fit_end": int(fit_end),
            "validation_end": int(validation_end),
            "method": "persistence",
            "ridge": np.nan,
            "validation_w2_mean": np.nan,
            "w2_mean": float(persistence["w2_mean"]),
            "w2_ratio_mean": 1.0,
            "n_test_origins": int(persistence["n_origins"]),
            "min_pred_eig": float(persistence["min_pred_eig"]),
            **metadata,
        }
    ]

    for method in DRIFT_METHOD_ORDER[1:]:
        if method == "local_bwar":
            forecast_for_ridge = lambda ridge: (
                lambda source: _local_forecast(
                    local_geometry[source],
                    ridge=ridge,
                )
            )
        else:
            coordinates, decoder = fixed_specs[method]
            forecast_for_ridge = lambda ridge, coordinates=coordinates, decoder=decoder: (
                lambda source: _encoded_forecast(
                    coordinates,
                    decoder,
                    source=source,
                    window_length=window_length,
                    ridge=ridge,
                )
            )

        tuning = []
        for ridge in ridge_grid:
            score = _score_forecasts(
                means,
                covariances,
                validation_sources,
                forecast_for_ridge(float(ridge)),
            )
            tuning.append((float(score["w2_mean"]), float(ridge)))
        best_validation, selected_ridge = min(tuning, key=lambda item: (item[0], -item[1]))
        test_score = _score_forecasts(
            means,
            covariances,
            test_sources,
            forecast_for_ridge(selected_ridge),
        )
        rows.append(
            {
                "setting": setting,
                "seed": int(seed),
                "n": int(n),
                "d": int(d),
                "phi": float(phi),
                "dispersion": float(dispersion),
                "window_length": int(window_length),
                "fit_end": int(fit_end),
                "validation_end": int(validation_end),
                "method": method,
                "ridge": float(selected_ridge),
                "validation_w2_mean": float(best_validation),
                "w2_mean": float(test_score["w2_mean"]),
                "w2_ratio_mean": float(test_score["w2_mean"])
                / max(float(persistence["w2_mean"]), 1e-12),
                "n_test_origins": int(test_score["n_origins"]),
                "min_pred_eig": float(test_score["min_pred_eig"]),
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def run_reference_drift_experiment(reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reps < 1:
        raise ValueError("reps must be positive")
    frames = []
    for setting in DRIFT_SETTINGS:
        print(f"[drift] {setting}: reps={reps}", flush=True)
        for seed in range(reps):
            frames.append(run_reference_drift_setting(setting=setting, seed=seed))
    raw = pd.concat(frames, ignore_index=True)
    return raw, summarize_reference_drift(raw)


def summarize_reference_drift(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"setting", "seed", "method", "w2_ratio_mean", "min_pred_eig"}
    if not isinstance(raw, pd.DataFrame) or not required.issubset(raw.columns):
        raise ValueError("raw drift results use an invalid schema")
    return (
        raw.groupby(["setting", "method"], as_index=False, sort=False)
        .agg(
            n_rep=("seed", "nunique"),
            w2_ratio_mean=("w2_ratio_mean", "mean"),
            w2_ratio_se=("w2_ratio_mean", _standard_error),
            min_pred_eig=("min_pred_eig", "min"),
        )
    )


def _setup_matplotlib():
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
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )
    return plt


def _save_figure(fig, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        path_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def _make_backgrounds_opaque(fig, axes) -> None:
    """Avoid transparent PDF/SVG page and axes patches in journal exports."""
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("white")
    fig.patch.set_linewidth(0.3)
    for axis in axes:
        axis.patch.set_facecolor("white")
        axis.patch.set_edgecolor("white")
        axis.patch.set_linewidth(0.3)


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=8.2,
        ha="right",
        va="bottom",
    )


def _set_opaque_canvas(fig, axes: tuple[object, ...]) -> None:
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("white")
    fig.patch.set_linewidth(0.1)
    for axis in axes:
        axis.patch.set_facecolor("white")
        axis.patch.set_edgecolor("white")
        axis.patch.set_linewidth(0.1)


def make_fixed_reference_figure(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    path_stem: Path,
) -> None:
    plt = _setup_matplotlib()
    fig = plt.figure(figsize=(7.1, 4.75))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.95, 1.05), hspace=0.52, wspace=0.38)
    axis_replication = fig.add_subplot(grid[0, 0])
    axis_covariance = fig.add_subplot(grid[0, 1])
    axis_variation = fig.add_subplot(grid[1, :])
    _set_opaque_canvas(
        fig,
        (axis_replication, axis_covariance, axis_variation),
    )
    _make_backgrounds_opaque(
        fig,
        (axis_replication, axis_covariance, axis_variation),
    )

    baseline = raw.loc[
        raw["design"].eq("Baseline") & raw["method"].isin(_FIXED_PLOT_METHODS)
    ]
    positions = np.arange(len(_FIXED_PLOT_METHODS))
    values = [
        baseline.loc[baseline["method"].eq(method), "w2_ratio_to_persistence"].to_numpy(float)
        for method in _FIXED_PLOT_METHODS
    ]
    box = axis_replication.boxplot(
        values,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 0.9},
        whiskerprops={"color": "#404040", "linewidth": 0.6},
        capprops={"color": "#404040", "linewidth": 0.6},
        boxprops={"edgecolor": "#404040", "linewidth": 0.7},
    )
    for patch, method in zip(box["boxes"], _FIXED_PLOT_METHODS, strict=True):
        patch.set_facecolor("#D9E5ED" if method == "bwar_barycenter" else "#F0F0F0")
    rng = np.random.default_rng(20260713)
    for position, method, method_values in zip(
        positions,
        _FIXED_PLOT_METHODS,
        values,
        strict=True,
    ):
        jitter = rng.uniform(-0.11, 0.11, size=len(method_values))
        point_color = "#5B829D" if method == "bwar_barycenter" else "#B8B8B8"
        axis_replication.plot(
            position + jitter,
            method_values,
            linestyle="none",
            marker="o",
            markersize=2.3,
            markerfacecolor=point_color,
            markeredgecolor=point_color,
            markeredgewidth=0.3,
            zorder=2,
        )
    axis_replication.axhline(1.0, color="#8C8C8C", lw=0.65, ls=(0, (4, 2)))
    axis_replication.set_xticks(positions)
    axis_replication.set_xticklabels(
        [METHOD_LABEL[method].replace(" AR", "") for method in _FIXED_PLOT_METHODS]
    )
    axis_replication.set_ylabel(r"Full $W_2^2$ ratio")
    axis_replication.set_title("Main setting: 50 replications")
    axis_replication.set_ylim(0.84, 1.17)

    covariance = summary.loc[
        summary["design"].eq("Baseline")
        & summary["method"].isin(_FIXED_PLOT_METHODS)
    ].copy()
    covariance["order"] = covariance["method"].map(
        {method: index for index, method in enumerate(_FIXED_PLOT_METHODS)}
    )
    covariance = covariance.sort_values("order")
    y_positions = np.arange(len(covariance))
    for y_position, row in zip(y_positions, covariance.itertuples(index=False), strict=True):
        method = str(row.method)
        axis_covariance.errorbar(
            float(row.cov_ratio_mean),
            y_position,
            xerr=float(row.cov_ratio_se),
            fmt=_MARKER[method],
            color=_COLOR[method],
            ms=4.1 if method == "bwar_barycenter" else 3.5,
            elinewidth=0.75,
            capsize=1.8,
            markeredgecolor="white",
            markeredgewidth=0.3,
        )
    axis_covariance.axvline(1.0, color="#8C8C8C", lw=0.65, ls=(0, (4, 2)))
    axis_covariance.set_yticks(y_positions)
    axis_covariance.set_yticklabels(
        [METHOD_LABEL[method] for method in covariance["method"]]
    )
    axis_covariance.invert_yaxis()
    axis_covariance.set_xlabel("Covariance loss ratio")
    axis_covariance.set_title("Covariance component: mean (SE)")
    axis_covariance.set_xlim(0.86, 1.03)

    designs = (
        "Baseline",
        "Shorter series",
        "Higher dimension",
        "Weaker dynamics",
        "Larger variation",
    )
    y_positions = np.arange(len(designs))
    offsets = np.linspace(-0.18, 0.18, len(_FIXED_PLOT_METHODS))
    for offset, method in zip(offsets, _FIXED_PLOT_METHODS, strict=True):
        part = summary.loc[summary["method"].eq(method)].copy()
        part["design"] = pd.Categorical(part["design"], categories=designs, ordered=True)
        part = part.sort_values("design")
        axis_variation.errorbar(
            part["w2_ratio_mean"].to_numpy(float),
            y_positions + offset,
            xerr=part["w2_ratio_se"].to_numpy(float),
            color=_COLOR[method],
            marker=_MARKER[method],
            linestyle="none",
            ms=3.3 if method == "bwar_barycenter" else 2.8,
            elinewidth=0.65,
            capsize=1.6,
            label=METHOD_LABEL[method],
        )
    axis_variation.axvline(1.0, color="#8C8C8C", lw=0.65, ls=(0, (4, 2)))
    axis_variation.set_yticks(y_positions)
    axis_variation.set_yticklabels(
        ("Baseline", "Shorter series", "Higher dimension", "Weaker dynamics", "Larger variation")
    )
    axis_variation.invert_yaxis()
    axis_variation.set_xlabel(r"Full $W_2^2$ ratio")
    axis_variation.set_title("Prespecified parameter variations: mean (SE)")
    axis_variation.set_xlim(0.77, 1.04)
    axis_variation.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        handlelength=2.0,
        columnspacing=1.2,
    )

    for label, axis in zip("abc", (axis_replication, axis_covariance, axis_variation), strict=True):
        _panel_label(axis, label)
        axis.grid(
            axis="y" if axis is axis_replication else "x",
            color="#E7E7E7",
            lw=0.38,
        )
        axis.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.15, top=0.95)
    _save_figure(fig, path_stem)
    plt.close(fig)


def make_reference_drift_figure(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    path_stem: Path,
) -> None:
    plt = _setup_matplotlib()
    fig, (axis_summary, axis_difference) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.0),
        gridspec_kw={"width_ratios": (1.22, 1.0), "wspace": 0.38},
    )
    _set_opaque_canvas(fig, (axis_summary, axis_difference))
    _make_backgrounds_opaque(fig, (axis_summary, axis_difference))
    plotted_methods = DRIFT_METHOD_ORDER[1:]
    y_positions = np.arange(len(DRIFT_SETTINGS))
    offsets = np.linspace(-0.18, 0.18, len(plotted_methods))
    for offset, method in zip(offsets, plotted_methods, strict=True):
        part = summary.loc[summary["method"].eq(method)].copy()
        part["setting"] = pd.Categorical(
            part["setting"],
            categories=DRIFT_SETTINGS,
            ordered=True,
        )
        part = part.sort_values("setting")
        axis_summary.errorbar(
            part["w2_ratio_mean"].to_numpy(float),
            y_positions + offset,
            xerr=part["w2_ratio_se"].to_numpy(float),
            color=_COLOR[method],
            marker=_MARKER[method],
            linestyle="none",
            ms=3.5 if method == "local_bwar" else 2.9,
            elinewidth=0.65,
            capsize=1.6,
            label=METHOD_LABEL[method],
        )
    axis_summary.axvline(1.0, color="#8C8C8C", lw=0.65, ls=(0, (4, 2)))
    axis_summary.set_yticks(y_positions)
    axis_summary.set_yticklabels(("No shift", "Mean shift", "Covariance shift", "Joint shift", "Gradual joint shift"))
    axis_summary.invert_yaxis()
    axis_summary.set_xlabel(r"Full $W_2^2$ ratio")
    axis_summary.set_xlim(0.905, 1.005)
    axis_summary.set_title("Reference-shift stress test: mean (SE)")
    axis_summary.grid(axis="x", color="#E7E7E7", lw=0.38)

    paired = raw.pivot(index=["setting", "seed"], columns="method", values="w2_ratio_mean").reset_index()
    paired["local_minus_fixed"] = paired["local_bwar"] - paired["fixed_bwar"]
    difference_values = [
        paired.loc[paired["setting"].eq(setting), "local_minus_fixed"].to_numpy(float)
        for setting in DRIFT_SETTINGS
    ]
    boxes = axis_difference.boxplot(
        difference_values,
        positions=y_positions,
        widths=0.48,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 0.9},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.6},
        capprops={"color": "#4A4A4A", "linewidth": 0.6},
        boxprops={"edgecolor": "#D62728", "linewidth": 0.75},
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#F6DEDE")
    axis_difference.axvline(0.0, color="#8C8C8C", lw=0.65, ls=(0, (4, 2)))
    axis_difference.set_yticks(y_positions)
    axis_difference.set_yticklabels(("No shift", "Mean shift", "Covariance shift", "Joint shift", "Gradual joint shift"))
    axis_difference.invert_yaxis()
    axis_difference.set_xlabel("Local BWAR - Fixed BWAR")
    axis_difference.set_title("Paired replication contrast")
    axis_difference.grid(axis="x", color="#E7E7E7", lw=0.38)

    for label, axis in zip("ab", (axis_summary, axis_difference), strict=True):
        _panel_label(axis, label)
        axis.set_axisbelow(True)
    handles, labels = axis_summary.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.25, top=0.90)
    _save_figure(fig, path_stem)
    plt.close(fig)


def _format_mean_se(mean: float, standard_error: float, best: float) -> str:
    cell = f"{mean:.3f} ({standard_error:.3f})"
    return rf"\textbf{{{cell}}}" if np.isclose(mean, best, rtol=1e-10, atol=1e-12) else cell


def write_reference_drift_table(summary: pd.DataFrame, path: Path, reps: int) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Reference-shift performance under rolling coefficient "
            r"refitting. Entries are mean full Gaussian \(W_2^2\) loss ratios "
            rf"to persistence over {reps} independent replications, with "
            r"standard errors in parentheses. All autoregressive methods use "
            r"the same trailing window, diagonal AR(1) restriction, ridge "
            r"grid, and forecast origins. Lower values are better.}"
        ),
        r"\label{tab:rolling-refit-reference-shift}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Setting & Persistence & Euclidean AR & Cholesky AR & Log-Euclidean AR & Fixed BWAR & Local BWAR \\",
        r"\midrule",
    ]
    for setting in DRIFT_SETTINGS:
        part = summary.loc[summary["setting"].eq(setting)].set_index("method")
        best = float(part["w2_ratio_mean"].min())
        cells = [
            _format_mean_se(
                float(part.loc[method, "w2_ratio_mean"]),
                float(part.loc[method, "w2_ratio_se"]),
                best,
            )
            for method in DRIFT_METHOD_ORDER
        ]
        lines.append(f"{setting} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all(
    *,
    fixed_reps: int,
    drift_reps: int,
    target_dir: Path,
) -> None:
    table_dir = target_dir / "tables"
    figure_dir = target_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    FIXED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed_raw, fixed_summary = run_fixed_reference(fixed_reps, ar_model="diag")
    fixed_raw.to_csv(FIXED_OUT_DIR / "strong_synthetic_transport_raw.csv", index=False)
    fixed_summary.to_csv(FIXED_OUT_DIR / "strong_synthetic_transport_summary.csv", index=False)
    write_main_table(fixed_summary, table_dir / "synthetic_transport_main.tex")
    write_variation_table(fixed_summary, table_dir / "synthetic_transport_variation.tex")
    make_fixed_reference_figure(
        fixed_raw,
        fixed_summary,
        figure_dir / "synthetic_transport_mechanism",
    )

    drift_raw, drift_summary = run_reference_drift_experiment(drift_reps)
    drift_raw.to_csv(DRIFT_OUT_DIR / "local_reference_drift_raw.csv", index=False)
    drift_summary.to_csv(DRIFT_OUT_DIR / "local_reference_drift_summary.csv", index=False)
    write_reference_drift_table(
        drift_summary,
        table_dir / "rolling_refit_reference_shift.tex",
        drift_reps,
    )
    make_reference_drift_figure(
        drift_raw,
        drift_summary,
        figure_dir / "rolling_refit_reference_shift",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the complete JCGS-style BWAR simulation package.",
    )
    parser.add_argument("--fixed-reps", type=int, default=50)
    parser.add_argument("--drift-reps", type=int, default=50)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    args = parser.parse_args()
    build_all(
        fixed_reps=args.fixed_reps,
        drift_reps=args.drift_reps,
        target_dir=args.target_dir,
    )


if __name__ == "__main__":
    main()
