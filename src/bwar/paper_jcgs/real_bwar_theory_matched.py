from __future__ import annotations

import argparse
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
import time
import zipfile
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.bwar_experiments import (  # noqa: E402
    bw2_cov,
    bw_barycenter,
    fetch_yahoo_adjclose,
    mat_exp,
    mat_from_triu,
    mat_log,
    ot_map,
    project_spd,
    triu_vec,
)
from bwar.bwar_online_real import download_uci_har, read_har_split  # noqa: E402


DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "paper_jcgs" / "outputs" / "real_bwar_theory_matched"
DEFAULT_RIDGE_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PRIMARY_METRIC_DEFAULT = "domain"
REFERENCE_LIBRARY_MODE = "full"
DEFAULT_DOMAIN_METRIC_PROFILE = {
    "metric": "window_mean_rmse",
    "label": "standardized window mean RMSE",
    "kind": "mean_rmse",
}
DOMAIN_METRIC_PROFILES: dict[str, dict[str, object]] = {
    "mhealth_full": {
        "metric": "motion_sensor_field_rmse",
        "label": "standardized motion-sensor level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "hapt": {
        "metric": "motion_sensor_field_rmse",
        "label": "standardized motion-sensor level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "pamap2": {
        "metric": "motion_sensor_field_rmse",
        "label": "standardized motion-sensor level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "intel_lab": {
        "metric": "environment_field_rmse",
        "label": "standardized environmental sensor-field level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "hai21": {
        "metric": "industrial_process_field_rmse",
        "label": "standardized industrial process-variable level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "sml2010": {
        "metric": "building_sensor_field_rmse",
        "label": "standardized building sensor-field level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "household_power": {
        "metric": "load_mean_rmse",
        "label": "standardized active-load level/volatility RMSE",
        "kind": "mean_sd_rmse",
        "indices": (0, 3, 4, 5, 6),
    },
    "household_power_strong": {
        "metric": "load_mean_rmse",
        "label": "standardized active-load level/volatility RMSE",
        "kind": "mean_sd_rmse",
        "indices": (0, 3, 4, 5, 6),
    },
    "capital_bikeshare": {
        "metric": "station_demand_level_rmse",
        "label": "standardized log station-demand level RMSE",
        "kind": "mean_rmse",
    },
    "baywheels": {
        "metric": "station_demand_level_rmse",
        "label": "standardized log station-demand level RMSE",
        "kind": "mean_rmse",
    },
    "bluebikes": {
        "metric": "station_demand_level_rmse",
        "label": "standardized log station-demand level RMSE",
        "kind": "mean_rmse",
    },
    "divvy": {
        "metric": "station_demand_level_rmse",
        "label": "standardized log station-demand level RMSE",
        "kind": "mean_rmse",
    },
    "citibike": {
        "metric": "station_demand_level_rmse",
        "label": "standardized log station-demand level RMSE",
        "kind": "mean_rmse",
    },
    "melbourne_pedestrian": {
        "metric": "pedestrian_demand_rmse",
        "label": "standardized log pedestrian-count level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "nyc_taxi": {
        "metric": "taxi_demand_level_rmse",
        "label": "standardized log taxi pickup-demand level RMSE",
        "kind": "mean_rmse",
    },
    "mta_subway": {
        "metric": "station_flow_rmse",
        "label": "standardized log station-flow level/Bures co-fluctuation RMSE",
        "kind": "mean_bures_rmse",
    },
    "solar_energy_hd": {
        "metric": "solar_generation_rmse",
        "label": "standardized solar-generation level/volatility RMSE",
        "kind": "mean_sd_rmse",
    },
    "electricity_hd": {
        "metric": "electricity_demand_rmse",
        "label": "standardized electricity-demand level/volatility RMSE",
        "kind": "mean_sd_rmse",
    },
    "beijing_air_hd": {
        "metric": "pollution_concentration_rmse",
        "label": "standardized log pollutant-concentration level/volatility RMSE",
        "kind": "mean_sd_rmse",
    },
    "finance_etf_hd": {
        "metric": "market_volatility_mae",
        "label": "average return-variance MAE",
        "kind": "variance_mae",
    },
    "gas_drift8": {
        "metric": "sensor_response_rmse",
        "label": "standardized gas-sensor response level/volatility RMSE",
        "kind": "mean_sd_rmse",
    },
    "hydraulic_systems": {
        "metric": "condition_sensor_rmse",
        "label": "standardized condition-sensor level/volatility RMSE",
        "kind": "mean_sd_rmse",
    },
    "wpp_population_age": {
        "metric": "population_age_structure_rmse",
        "label": "population age-share structure level/co-movement RMSE",
        "kind": "mean_cov_rmse",
    },
    "wpp_population_recent_age": {
        "metric": "population_recent_45_90_rmse",
        "label": "population 45--90 age-share level/co-movement RMSE",
        "kind": "mean_cov_rmse",
    },
}
FINANCE_ETF_HD_SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "USO",
    "VNQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
)
BEIJING_AIR_STATIONS = (
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
)
BEIJING_AIR_FEATURES = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")
INTEL_LAB_MEASURES = ("temp", "humidity", "light", "voltage")
HAI21_TRAIN1_PATH = DATA / "hai_21_03" / "train1.csv.gz"
MELBOURNE_PEDESTRIAN_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
    "?lang=en&timezone=Australia%2FMelbourne&use_labels=false&delimiter=%2C"
)
HYDRAULIC_SENSOR_FILES = (
    "CE.txt",
    "CP.txt",
    "EPS1.txt",
    "FS1.txt",
    "FS2.txt",
    "PS1.txt",
    "PS2.txt",
    "PS3.txt",
    "PS4.txt",
    "PS5.txt",
    "PS6.txt",
    "SE.txt",
    "TS1.txt",
    "TS2.txt",
    "TS3.txt",
    "TS4.txt",
    "VS1.txt",
)
WPP_POPULATION_AGE5_PERCENT_PATH = (
    "assets/Excel Files/1_Indicator (Standard)/CSV_FILES/"
    "WPP2024_PopulationByAge5GroupSex_Percentage_Medium.csv.gz"
)
WPP_POPULATION_AGE5_PERCENT_URL = "https://population.un.org/wpp/" + quote(
    WPP_POPULATION_AGE5_PERCENT_PATH,
    safe="/()_-.%",
)
WPP_POPULATION_AGE5_PERCENT_CACHE = DATA / "wpp_population" / "WPP2024_PopulationByAge5GroupSex_Percentage_Medium.csv.gz"
WPP_G20_ISO3 = (
    "ARG",
    "AUS",
    "BRA",
    "CAN",
    "CHN",
    "DEU",
    "FRA",
    "GBR",
    "IDN",
    "IND",
    "ITA",
    "JPN",
    "KOR",
    "MEX",
    "RUS",
    "SAU",
    "TUR",
    "USA",
    "ZAF",
)
WPP_AGEING_ISO3 = (
    "AUT",
    "BEL",
    "CHE",
    "CHN",
    "DEU",
    "ESP",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "HKG",
    "ITA",
    "JPN",
    "KOR",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "SGP",
    "SWE",
    "TWN",
)
PAMAP2_FEATURE_COLUMNS = tuple(
    j
    for base in (3, 20, 37)
    for j in range(base + 1, base + 13)
)


def domain_metric_profile(dataset: str | None) -> dict[str, object]:
    key = str(dataset or "").lower()
    if key in DOMAIN_METRIC_PROFILES:
        return dict(DOMAIN_METRIC_PROFILES[key])
    if "bikeshare" in key or "bike" in key:
        return dict(DOMAIN_METRIC_PROFILES["capital_bikeshare"])
    if "taxi" in key:
        return dict(DOMAIN_METRIC_PROFILES["nyc_taxi"])
    if "mta" in key or "subway" in key:
        return dict(DOMAIN_METRIC_PROFILES["mta_subway"])
    if "pedestrian" in key:
        return dict(DOMAIN_METRIC_PROFILES["melbourne_pedestrian"])
    if "mhealth" in key or "hapt" in key or "pamap" in key:
        return dict(DOMAIN_METRIC_PROFILES["mhealth_full"])
    if "household" in key or "power" in key:
        return dict(DOMAIN_METRIC_PROFILES["household_power_strong"])
    if "hai" in key or "industrial" in key:
        return dict(DOMAIN_METRIC_PROFILES["hai21"])
    if "intel" in key or "sensor" in key:
        return dict(DOMAIN_METRIC_PROFILES["intel_lab"])
    if "finance" in key or "etf" in key:
        return dict(DOMAIN_METRIC_PROFILES["finance_etf_hd"])
    return dict(DEFAULT_DOMAIN_METRIC_PROFILE)


def _domain_indices(profile: dict[str, object], dimension: int) -> np.ndarray:
    raw = profile.get("indices")
    if raw is None:
        return np.arange(dimension, dtype=int)
    idx = np.asarray(raw, dtype=int)
    idx = idx[(0 <= idx) & (idx < dimension)]
    if len(idx) == 0:
        return np.arange(dimension, dtype=int)
    return idx


def domain_loss_from_moments(
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    profile: dict[str, object] | None = None,
) -> float:
    profile = dict(DEFAULT_DOMAIN_METRIC_PROFILE if profile is None else profile)
    pred_mean = np.asarray(pred_mean, dtype=float)
    target_mean = np.asarray(target_mean, dtype=float)
    pred_cov = np.asarray(pred_cov, dtype=float)
    target_cov = np.asarray(target_cov, dtype=float)
    idx = _domain_indices(profile, len(pred_mean))
    kind = str(profile.get("kind", "mean_rmse"))

    if kind == "energy_mae":
        pred_energy = (float(pred_mean[idx] @ pred_mean[idx]) + float(np.trace(pred_cov[np.ix_(idx, idx)]))) / len(idx)
        target_energy = (
            float(target_mean[idx] @ target_mean[idx]) + float(np.trace(target_cov[np.ix_(idx, idx)]))
        ) / len(idx)
        return float(abs(pred_energy - target_energy))

    if kind == "variance_mae":
        pred_var = float(np.mean(np.diag(pred_cov)[idx]))
        target_var = float(np.mean(np.diag(target_cov)[idx]))
        return float(abs(pred_var - target_var))

    diff = pred_mean[idx] - target_mean[idx]
    if kind == "mean_cov_rmse":
        mean_rmse = float(np.sqrt(np.mean(diff**2)))
        pred_sub = pred_cov[np.ix_(idx, idx)]
        target_sub = target_cov[np.ix_(idx, idx)]
        cov_rmse = float(np.sqrt(np.mean((pred_sub - target_sub) ** 2)))
        cov_weight = float(profile.get("cov_weight", 1.0))
        return float(np.sqrt(mean_rmse**2 + cov_weight * cov_rmse**2))
    if kind == "mean_bures_rmse":
        mean_rmse = float(np.sqrt(np.mean(diff**2)))
        pred_sub = project_spd(pred_cov[np.ix_(idx, idx)], eps=1e-8)
        target_sub = project_spd(target_cov[np.ix_(idx, idx)], eps=1e-8)
        cov_rmse = float(np.sqrt(max(bw2_cov(pred_sub, target_sub), 0.0) / len(idx)))
        cov_weight = float(profile.get("cov_weight", 1.0))
        return float(np.sqrt(mean_rmse**2 + cov_weight * cov_rmse**2))
    if kind == "mean_sd_rmse":
        mean_rmse = float(np.sqrt(np.mean(diff**2)))
        pred_sd = np.sqrt(np.clip(np.diag(pred_cov)[idx], 0.0, None))
        target_sd = np.sqrt(np.clip(np.diag(target_cov)[idx], 0.0, None))
        sd_rmse = float(np.sqrt(np.mean((pred_sd - target_sd) ** 2)))
        sd_weight = float(profile.get("sd_weight", 1.0))
        return float(np.sqrt(mean_rmse**2 + sd_weight * sd_rmse**2))
    if kind == "mean_mae":
        return float(np.mean(np.abs(diff)))
    if kind == "physical_mean_rmse":
        center = np.asarray(profile.get("center"), dtype=float)
        scale = np.asarray(profile.get("scale"), dtype=float)
        if center.shape[0] != len(pred_mean) or scale.shape[0] != len(pred_mean):
            raise ValueError("physical_mean_rmse requires center and scale vectors matching the Gaussian dimension")
        pred_phys = pred_mean[idx] * scale[idx] + center[idx]
        target_phys = target_mean[idx] * scale[idx] + center[idx]
        return float(np.sqrt(np.mean((pred_phys - target_phys) ** 2)))
    return float(np.sqrt(np.mean(diff**2)))


def metric_mean_key(metric: str, prefix: str) -> str:
    if metric in {"domain", "domain_loss"}:
        return f"{prefix}_domain_loss_mean"
    if metric == "w2":
        return f"{prefix}_w2_mean"
    if metric == "log_score":
        return f"{prefix}_log_score_mean"
    raise ValueError(f"unknown metric: {metric}")


def choose_primary_loss_column(raw: pd.DataFrame, preferred: str | None = None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["test_domain_loss_mean", "test_log_score_mean", "test_w2_mean"])
    for col in candidates:
        if col in raw.columns and raw[col].notna().any():
            return col
    raise ValueError("raw results contain none of test_domain_loss_mean, test_log_score_mean, or test_w2_mean")


def relative_loss_reduction(target_loss: pd.Series, baseline_loss: pd.Series) -> pd.Series:
    denom = baseline_loss.astype(float).abs().clip(lower=1e-12)
    return (baseline_loss.astype(float) - target_loss.astype(float)) / denom


def gaussian_w2_squared(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    return float(np.sum((np.asarray(mean_a) - np.asarray(mean_b)) ** 2) + bw2_cov(cov_a, cov_b))


def gaussian_log_score_from_moments(
    pred_mean: np.ndarray,
    pred_cov: np.ndarray,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    *,
    eps: float = 1e-8,
) -> float:
    """Average Gaussian negative log likelihood of a target window from its moments."""
    pred_mean = np.asarray(pred_mean, dtype=float)
    target_mean = np.asarray(target_mean, dtype=float)
    pred_cov = project_spd(np.asarray(pred_cov, dtype=float), eps=eps)
    target_cov = project_spd(np.asarray(target_cov, dtype=float), eps=eps)
    sign, logdet = np.linalg.slogdet(pred_cov)
    if sign <= 0:
        pred_cov = project_spd(pred_cov, eps=max(eps, 1e-6))
        sign, logdet = np.linalg.slogdet(pred_cov)
    diff = target_mean - pred_mean
    solved_target = np.linalg.solve(pred_cov, target_cov)
    solved_diff = np.linalg.solve(pred_cov, diff)
    d = pred_cov.shape[0]
    return float(
        0.5
        * (
            d * np.log(2.0 * np.pi)
            + logdet
            + np.trace(solved_target)
            + float(diff @ solved_diff)
        )
    )


def safe_transport_from_vech(z: np.ndarray, d: int, min_eig: float = 0.05) -> np.ndarray:
    H = mat_from_triu(z, d)
    vals = np.linalg.eigvalsh(np.eye(d) + H)
    if vals.min() < min_eig:
        h_min = float(np.linalg.eigvalsh(H).min())
        if h_min < 0.0:
            scale = (1.0 - min_eig) / max(abs(h_min), 1e-12)
            H = scale * H
    return project_spd(np.eye(d) + H, eps=1e-8)


def bwar_gaussian_encode(
    mean: np.ndarray,
    cov: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov: np.ndarray,
) -> np.ndarray:
    d = ref_cov.shape[0]
    A = ot_map(ref_cov, cov)
    return np.r_[np.asarray(mean) - np.asarray(ref_mean), triu_vec(A - np.eye(d))]


def bwar_gaussian_decode(
    z: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d = ref_cov.shape[0]
    pred_mean = np.asarray(ref_mean) + np.asarray(z[:d])
    A = safe_transport_from_vech(np.asarray(z[d:]), d)
    pred_cov = project_spd(A @ ref_cov @ A, eps=1e-8)
    return pred_mean, pred_cov


def cholesky_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = project_spd(cov, eps=1e-8)
    L = np.linalg.cholesky(cov)
    idx = np.tril_indices(cov.shape[0])
    z = L[idx].copy()
    diag_positions = [np.where((idx[0] == j) & (idx[1] == j))[0][0] for j in range(cov.shape[0])]
    z[diag_positions] = np.log(np.clip(z[diag_positions], 1e-12, None))
    return np.r_[mean, z]


def cholesky_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(z[:d])
    lower = np.asarray(z[d:])
    L = np.zeros((d, d), dtype=float)
    idx = np.tril_indices(d)
    L[idx] = lower
    for j in range(d):
        pos = np.where((idx[0] == j) & (idx[1] == j))[0][0]
        L[j, j] = np.exp(np.clip(L[j, j], -30, 30))
    return mean, project_spd(L @ L.T, eps=1e-8)


def euclidean_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.r_[mean, triu_vec(cov)]


def euclidean_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(z[:d]), project_spd(mat_from_triu(np.asarray(z[d:]), d), eps=1e-8)


def log_euclidean_encode(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.r_[mean, triu_vec(mat_log(cov))]


def log_euclidean_decode(z: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(z[:d]), project_spd(mat_exp(mat_from_triu(np.asarray(z[d:]), d)), eps=1e-8)


def fit_var(Z: np.ndarray, end: int, *, lam: float, model: str) -> np.ndarray:
    X = Z[: end - 1]
    Y = Z[1:end]
    if len(X) < 2:
        raise ValueError("not enough observations to fit VAR")

    if model == "diag":
        W = np.zeros((Z.shape[1] + 1, Z.shape[1]))
        for j in range(Z.shape[1]):
            Xj = np.column_stack([np.ones(len(X)), X[:, j]])
            penalty = lam * np.eye(2)
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(Xj.T @ Xj + penalty, Xj.T @ Y[:, j])
            W[0, j] = coef[0]
            W[j + 1, j] = coef[1]
        return W

    if model == "full":
        Xa = np.column_stack([np.ones(len(X)), X])
        penalty = lam * np.eye(Xa.shape[1])
        penalty[0, 0] = 0.0
        return np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ Y)

    raise ValueError(f"unknown ar_model: {model}")


def recursive_predict_z(z0: np.ndarray, W: np.ndarray, horizon: int) -> np.ndarray:
    z = np.asarray(z0, dtype=float)
    for _ in range(horizon):
        z = np.r_[1.0, z] @ W
    return z


def split_indices(n: int) -> tuple[int, int, int]:
    fit_end = max(20, int(0.45 * n))
    val_end = max(fit_end + 8, int(0.65 * n))
    val_end = min(val_end, n - 8)
    return fit_end, val_end, n


def score_recursive_forecasts(
    means: np.ndarray,
    covs: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    decode,
    *,
    start_t: int,
    stop_t: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    losses = []
    log_scores = []
    domain_losses = []
    min_eigs = []
    for t in range(start_t, stop_t):
        if t + horizon >= len(covs):
            continue
        pred_mean, pred_cov = decode(recursive_predict_z(Z[t], W, horizon))
        pred_cov = project_spd(pred_cov, eps=1e-8)
        losses.append(gaussian_w2_squared(pred_mean, pred_cov, means[t + horizon], covs[t + horizon]))
        log_scores.append(
            gaussian_log_score_from_moments(pred_mean, pred_cov, means[t + horizon], covs[t + horizon])
        )
        domain_losses.append(
            domain_loss_from_moments(pred_mean, pred_cov, means[t + horizon], covs[t + horizon], profile)
        )
        min_eigs.append(float(np.linalg.eigvalsh(pred_cov).min()))
    arr = np.asarray(losses, dtype=float)
    log_arr = np.asarray(log_scores, dtype=float)
    domain_arr = np.asarray(domain_losses, dtype=float)
    if len(arr) == 0:
        return {
            "w2_mean": np.nan,
            "w2_median": np.nan,
            "w2_q90": np.nan,
            "log_score_mean": np.nan,
            "log_score_median": np.nan,
            "log_score_q90": np.nan,
            "domain_loss_mean": np.nan,
            "domain_loss_median": np.nan,
            "domain_loss_q90": np.nan,
            "n_pairs": 0,
            "min_pred_eig": np.nan,
        }
    return {
        "w2_mean": float(arr.mean()),
        "w2_median": float(np.median(arr)),
        "w2_q90": float(np.quantile(arr, 0.9)),
        "log_score_mean": float(log_arr.mean()),
        "log_score_median": float(np.median(log_arr)),
        "log_score_q90": float(np.quantile(log_arr, 0.9)),
        "domain_loss_mean": float(domain_arr.mean()),
        "domain_loss_median": float(np.median(domain_arr)),
        "domain_loss_q90": float(np.quantile(domain_arr, 0.9)),
        "n_pairs": int(len(arr)),
        "min_pred_eig": float(np.min(min_eigs)),
    }


def evaluate_encoded_ar(
    means: np.ndarray,
    covs: np.ndarray,
    encode,
    decode,
    *,
    fit_end: int,
    val_end: int,
    horizon: int,
    ar_model: str,
    selection_metric: str = PRIMARY_METRIC_DEFAULT,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    Z = np.vstack([encode(m, C) for m, C in zip(means, covs)])
    best_lam = ridge_grid[0]
    best_val = np.inf
    best_val_metrics: dict[str, float | int] | None = None
    val_start = max(0, fit_end - horizon)
    val_stop = max(val_start, val_end - horizon)

    for lam in ridge_grid:
        W = fit_var(Z, fit_end, lam=lam, model=ar_model)
        val_metrics = score_recursive_forecasts(
            means,
            covs,
            Z,
            W,
            decode,
            start_t=val_start,
            stop_t=val_stop,
            horizon=horizon,
            domain_profile=profile,
        )
        val_score = float(val_metrics[metric_mean_key(selection_metric, "val").removeprefix("val_")])
        if np.isfinite(val_score) and val_score < best_val:
            best_val = val_score
            best_lam = lam
            best_val_metrics = val_metrics

    if best_val_metrics is None:
        raise ValueError("no finite validation score")

    final_W = fit_var(Z, val_end, lam=best_lam, model=ar_model)
    test_start = max(0, val_end - horizon)
    test_metrics = score_recursive_forecasts(
        means,
        covs,
        Z,
        final_W,
        decode,
        start_t=test_start,
        stop_t=len(covs) - horizon,
        horizon=horizon,
        domain_profile=profile,
    )
    return {
        "ridge": float(best_lam),
        "val_w2_mean": float(best_val_metrics["w2_mean"]),
        "val_w2_median": float(best_val_metrics["w2_median"]),
        "val_log_score_mean": float(best_val_metrics["log_score_mean"]),
        "val_log_score_median": float(best_val_metrics["log_score_median"]),
        "val_domain_loss_mean": float(best_val_metrics["domain_loss_mean"]),
        "val_domain_loss_median": float(best_val_metrics["domain_loss_median"]),
        "test_w2_mean": float(test_metrics["w2_mean"]),
        "test_w2_median": float(test_metrics["w2_median"]),
        "test_w2_q90": float(test_metrics["w2_q90"]),
        "test_log_score_mean": float(test_metrics["log_score_mean"]),
        "test_log_score_median": float(test_metrics["log_score_median"]),
        "test_log_score_q90": float(test_metrics["log_score_q90"]),
        "test_domain_loss_mean": float(test_metrics["domain_loss_mean"]),
        "test_domain_loss_median": float(test_metrics["domain_loss_median"]),
        "test_domain_loss_q90": float(test_metrics["domain_loss_q90"]),
        "n_test_pairs": int(test_metrics["n_pairs"]),
        "min_pred_eig": float(test_metrics["min_pred_eig"]),
    }


def persistence_metrics(
    means: np.ndarray,
    covs: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile

    def score(start_t: int, stop_t: int) -> dict[str, float | int]:
        losses = []
        log_scores = []
        domain_losses = []
        for t in range(start_t, stop_t):
            if t + horizon < len(covs):
                losses.append(gaussian_w2_squared(means[t], covs[t], means[t + horizon], covs[t + horizon]))
                log_scores.append(
                    gaussian_log_score_from_moments(means[t], covs[t], means[t + horizon], covs[t + horizon])
                )
                domain_losses.append(
                    domain_loss_from_moments(means[t], covs[t], means[t + horizon], covs[t + horizon], profile)
                )
        arr = np.asarray(losses, dtype=float)
        log_arr = np.asarray(log_scores, dtype=float)
        domain_arr = np.asarray(domain_losses, dtype=float)
        return {
            "w2_mean": float(arr.mean()) if len(arr) else np.nan,
            "w2_median": float(np.median(arr)) if len(arr) else np.nan,
            "w2_q90": float(np.quantile(arr, 0.9)) if len(arr) else np.nan,
            "log_score_mean": float(log_arr.mean()) if len(log_arr) else np.nan,
            "log_score_median": float(np.median(log_arr)) if len(log_arr) else np.nan,
            "log_score_q90": float(np.quantile(log_arr, 0.9)) if len(log_arr) else np.nan,
            "domain_loss_mean": float(domain_arr.mean()) if len(domain_arr) else np.nan,
            "domain_loss_median": float(np.median(domain_arr)) if len(domain_arr) else np.nan,
            "domain_loss_q90": float(np.quantile(domain_arr, 0.9)) if len(domain_arr) else np.nan,
            "n_pairs": int(len(arr)),
        }

    val = score(max(0, fit_end - horizon), max(0, val_end - horizon))
    test = score(max(0, val_end - horizon), len(covs) - horizon)
    return {
        "ridge": np.nan,
        "val_w2_mean": float(val["w2_mean"]),
        "val_w2_median": float(val["w2_median"]),
        "val_log_score_mean": float(val["log_score_mean"]),
        "val_log_score_median": float(val["log_score_median"]),
        "val_domain_loss_mean": float(val["domain_loss_mean"]),
        "val_domain_loss_median": float(val["domain_loss_median"]),
        "test_w2_mean": float(test["w2_mean"]),
        "test_w2_median": float(test["w2_median"]),
        "test_w2_q90": float(test["w2_q90"]),
        "test_log_score_mean": float(test["log_score_mean"]),
        "test_log_score_median": float(test["log_score_median"]),
        "test_log_score_q90": float(test["log_score_q90"]),
        "test_domain_loss_mean": float(test["domain_loss_mean"]),
        "test_domain_loss_median": float(test["domain_loss_median"]),
        "test_domain_loss_q90": float(test["domain_loss_q90"]),
        "n_test_pairs": int(test["n_pairs"]),
        "min_pred_eig": float(min(np.linalg.eigvalsh(C).min() for C in covs[max(0, val_end - horizon) : len(covs) - horizon])),
    }


def bw_geodesic(S0: np.ndarray, S1: np.ndarray, alpha: float) -> np.ndarray:
    A = ot_map(S0, S1)
    M = (1.0 - alpha) * np.eye(S0.shape[0]) + alpha * A
    return project_spd(M @ S0 @ M, eps=1e-8)


def set_reference_library_mode(mode: str) -> None:
    if mode not in {"full", "no_barycenter", "fast"}:
        raise ValueError(f"unknown reference library mode: {mode}")
    global REFERENCE_LIBRARY_MODE
    REFERENCE_LIBRARY_MODE = mode


def candidate_gaussian_references(
    means_fit: np.ndarray,
    covs_fit: np.ndarray,
    *,
    max_sample_refs: int = 4,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    train_covs = np.asarray([project_spd(C, eps=1e-8) for C in covs_fit])
    train_means = np.asarray(means_fit, dtype=float)
    d = train_covs.shape[1]
    mean_ref = train_means.mean(axis=0)
    pooled = project_spd(np.mean(train_covs, axis=0), eps=1e-8)
    diag_pooled = project_spd(np.diag(np.diag(pooled)), eps=1e-8)
    scaled_identity = project_spd(np.eye(d) * np.trace(pooled) / d, eps=1e-8)
    refs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("pooled_cov", mean_ref, pooled),
        ("log_euclidean_mean", mean_ref, mat_exp(np.mean([mat_log(C) for C in train_covs], axis=0))),
        ("diag_pooled", mean_ref, diag_pooled),
        ("scaled_identity", mean_ref, scaled_identity),
    ]
    if REFERENCE_LIBRARY_MODE == "fast":
        refs.append(("geodesic_identity_a0.25", mean_ref, bw_geodesic(pooled, scaled_identity, 0.25)))
        return refs
    if REFERENCE_LIBRARY_MODE == "full":
        try:
            refs.append(("bw_barycenter", mean_ref, bw_barycenter(train_covs[: min(len(train_covs), 250)])))
        except Exception:
            pass

    idx = np.linspace(0, len(train_covs) - 1, min(max_sample_refs, len(train_covs)), dtype=int)
    for j, i in enumerate(idx):
        refs.append((f"sample_{j}", train_means[i], train_covs[i]))
    for alpha in (0.25, 0.5, 0.75):
        refs.append((f"geodesic_diag_a{alpha}", mean_ref, bw_geodesic(pooled, diag_pooled, alpha)))
        refs.append((f"geodesic_identity_a{alpha}", mean_ref, bw_geodesic(pooled, scaled_identity, alpha)))
    return refs


def evaluate_bwar_reference(
    means: np.ndarray,
    covs: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    horizon: int,
    ar_model: str,
    selection_metric: str = PRIMARY_METRIC_DEFAULT,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    return evaluate_encoded_ar(
        means,
        covs,
        lambda m, C: bwar_gaussian_encode(m, C, ref_mean, ref_cov),
        lambda z: bwar_gaussian_decode(z, ref_mean, ref_cov),
        fit_end=fit_end,
        val_end=val_end,
        horizon=horizon,
        ar_model=ar_model,
        selection_metric=selection_metric,
        domain_profile=domain_profile,
    )


def optimize_diagonal_yule_walker_reference(
    means: np.ndarray,
    covs: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    horizon: int,
    ar_model: str,
    selection_metric: str = PRIMARY_METRIC_DEFAULT,
    domain_profile: dict[str, object] | None = None,
    max_iter: int = 4,
    initial_step: float = 0.5,
    min_step: float = 0.05,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
) -> dict[str, object]:
    """Optimize a diagonal SPD reference using the selected validation metric."""
    means = np.asarray(means, dtype=float)
    covs = np.asarray([project_spd(C, eps=1e-8) for C in covs], dtype=float)
    if fit_end < 3 or val_end <= fit_end:
        raise ValueError("fit_end and val_end do not define a usable validation split")
    train_means = means[:fit_end]
    train_covs = covs[:fit_end]
    ref_mean = train_means.mean(axis=0)
    pooled = project_spd(np.mean(train_covs, axis=0), eps=1e-8)
    theta = np.log(np.clip(np.diag(pooled), 1e-8, None))

    def evaluate(theta_value: np.ndarray) -> tuple[float, dict[str, float | int], np.ndarray]:
        ref_cov = project_spd(np.diag(np.exp(theta_value)), eps=1e-8)
        metrics = evaluate_bwar_reference(
            means,
            covs,
            ref_mean,
            ref_cov,
            fit_end=fit_end,
            val_end=val_end,
            horizon=horizon,
            ar_model=ar_model,
            selection_metric=selection_metric,
            domain_profile=domain_profile,
        )
        score_key = metric_mean_key(selection_metric, "val")
        return float(metrics[score_key]), metrics, ref_cov

    initial_score, initial_metrics, initial_cov = evaluate(theta)
    best_score = initial_score
    best_metrics = initial_metrics
    best_cov = initial_cov
    best_theta = theta.copy()
    step = float(initial_step)
    score_key = metric_mean_key(selection_metric, "val")
    history = [{"step": 0, score_key: float(best_score)}]

    for iteration in range(1, max_iter + 1):
        improved = False
        for j in range(len(best_theta)):
            for direction in (-1.0, 1.0):
                candidate = best_theta.copy()
                candidate[j] += direction * step
                score, metrics, ref_cov = evaluate(candidate)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_metrics = metrics
                    best_cov = ref_cov
                    best_theta = candidate
                    improved = True
        history.append({"step": iteration, score_key: float(best_score), "search_step": float(step)})
        if not improved:
            step *= 0.5
            if step < min_step:
                break

    return {
        "reference_name": "optimized_diag_yw",
        "ref_mean": ref_mean,
        "ref_cov": best_cov,
        "log_diag": best_theta,
        "initial_ref_cov": initial_cov,
        "selection_metric": selection_metric,
        "validation_score_key": score_key,
        "initial_val_score": float(initial_score),
        "best_val_score": float(best_score),
        "initial_val_log_score_mean": float(initial_metrics["val_log_score_mean"]),
        "best_val_log_score_mean": float(best_metrics["val_log_score_mean"]),
        "best_metrics": best_metrics,
        "history": history,
    }


def run_single_series(
    *,
    job: str,
    dataset: str,
    means: np.ndarray,
    covs: np.ndarray,
    meta: dict,
    horizons: list[int],
    ar_model: str = "diag",
    selection_metric: str = PRIMARY_METRIC_DEFAULT,
    min_n: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = np.asarray(means, dtype=float)
    covs = np.asarray([project_spd(C, eps=1e-8) for C in covs], dtype=float)
    if len(means) != len(covs):
        raise ValueError("means and covs must have the same length")
    if len(covs) < min_n:
        return pd.DataFrame(), pd.DataFrame()
    fit_end, val_end, test_end = split_indices(len(covs))
    d = covs.shape[1]
    profile = domain_metric_profile(dataset)
    rows: list[dict[str, object]] = []
    ref_rows: list[dict[str, object]] = []

    def add(horizon: int, method: str, metrics: dict[str, object], selected_reference: str = "") -> None:
        rows.append(
            {
                "job": job,
                "dataset": dataset,
                "method": method,
                "selected_reference": selected_reference,
                "horizon": int(horizon),
                "n_matrices": int(len(covs)),
                "dimension": int(d),
                "fit_end": int(fit_end),
                "val_end": int(val_end),
                "test_end": int(test_end),
                "ar_model": ar_model,
                "selection_metric": selection_metric,
                "domain_metric": str(profile["metric"]),
                "domain_metric_label": str(profile["label"]),
                **meta,
                **metrics,
            }
        )

    refs = candidate_gaussian_references(means[:fit_end], covs[:fit_end])
    ref_by_name = {name: (m, C) for name, m, C in refs}

    for horizon in horizons:
        if len(covs) <= val_end + horizon + 2:
            continue
        add(
            horizon,
            "persistence",
            persistence_metrics(
                means,
                covs,
                fit_end=fit_end,
                val_end=val_end,
                horizon=horizon,
                domain_profile=profile,
            ),
        )
        add(
            horizon,
            "euclidean_gaussian_ar",
            evaluate_encoded_ar(
                means,
                covs,
                euclidean_encode,
                lambda z: euclidean_decode(z, d),
                fit_end=fit_end,
                val_end=val_end,
                horizon=horizon,
                ar_model=ar_model,
                selection_metric=selection_metric,
                domain_profile=profile,
            ),
        )
        add(
            horizon,
            "cholesky_gaussian_ar",
            evaluate_encoded_ar(
                means,
                covs,
                cholesky_encode,
                lambda z: cholesky_decode(z, d),
                fit_end=fit_end,
                val_end=val_end,
                horizon=horizon,
                ar_model=ar_model,
                selection_metric=selection_metric,
                domain_profile=profile,
            ),
        )
        add(
            horizon,
            "log_euclidean_gaussian_ar",
            evaluate_encoded_ar(
                means,
                covs,
                log_euclidean_encode,
                lambda z: log_euclidean_decode(z, d),
                fit_end=fit_end,
                val_end=val_end,
                horizon=horizon,
                ar_model=ar_model,
                selection_metric=selection_metric,
                domain_profile=profile,
            ),
        )

        for fixed_name, method in [("pooled_cov", "bwar_pooled_ref"), ("bw_barycenter", "bwar_barycenter")]:
            if fixed_name in ref_by_name:
                ref_mean, ref_cov = ref_by_name[fixed_name]
                try:
                    add(
                        horizon,
                        method,
                        evaluate_bwar_reference(
                            means,
                            covs,
                            ref_mean,
                            ref_cov,
                            fit_end=fit_end,
                            val_end=val_end,
                            horizon=horizon,
                            ar_model=ar_model,
                            selection_metric=selection_metric,
                            domain_profile=profile,
                        ),
                        selected_reference=fixed_name,
                    )
                except Exception as exc:
                    add(
                        horizon,
                        method,
                        {
                            "test_w2_mean": np.nan,
                            "test_log_score_mean": np.nan,
                            "test_domain_loss_mean": np.nan,
                            "error": repr(exc),
                        },
                        selected_reference=fixed_name,
                    )

        candidate_metrics = []
        for ref_name, ref_mean, ref_cov in refs:
            try:
                metrics = evaluate_bwar_reference(
                    means,
                    covs,
                    ref_mean,
                    ref_cov,
                    fit_end=fit_end,
                    val_end=val_end,
                    horizon=horizon,
                    ar_model=ar_model,
                    selection_metric=selection_metric,
                    domain_profile=profile,
                )
                candidate_metrics.append((ref_name, metrics))
                ref_rows.append(
                    {
                        "job": job,
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "reference": ref_name,
                        "split_used": "train",
                        "val_w2_mean": metrics["val_w2_mean"],
                        "test_w2_mean": metrics["test_w2_mean"],
                        "val_log_score_mean": metrics["val_log_score_mean"],
                        "test_log_score_mean": metrics["test_log_score_mean"],
                        "val_domain_loss_mean": metrics["val_domain_loss_mean"],
                        "test_domain_loss_mean": metrics["test_domain_loss_mean"],
                        "domain_metric": str(profile["metric"]),
                        "domain_metric_label": str(profile["label"]),
                        "ridge": metrics["ridge"],
                        **meta,
                    }
                )
            except Exception as exc:
                ref_rows.append(
                    {
                        "job": job,
                        "dataset": dataset,
                        "horizon": int(horizon),
                        "reference": ref_name,
                        "split_used": "train",
                        "val_w2_mean": np.nan,
                        "test_w2_mean": np.nan,
                        "val_log_score_mean": np.nan,
                        "test_log_score_mean": np.nan,
                        "val_domain_loss_mean": np.nan,
                        "test_domain_loss_mean": np.nan,
                        "domain_metric": str(profile["metric"]),
                        "domain_metric_label": str(profile["label"]),
                        "ridge": np.nan,
                        "error": repr(exc),
                        **meta,
                    }
                )
        if candidate_metrics:
            best_name, best_metrics = min(
                candidate_metrics,
                key=lambda item: float(item[1][metric_mean_key(selection_metric, "val")]),
            )
            add(horizon, "bwar_selected_ref", best_metrics, selected_reference=best_name)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["test_loss_ratio_to_persistence"] = np.nan
        out["test_log_score_ratio_to_persistence"] = np.nan
        for horizon, part in out.groupby("horizon"):
            persistence = part.loc[part["method"] == "persistence", "test_w2_mean"]
            if not persistence.empty:
                denom = max(float(persistence.iloc[0]), 1e-12)
                out.loc[part.index, "test_loss_ratio_to_persistence"] = part["test_w2_mean"].astype(float) / denom
            if "test_log_score_mean" in part.columns:
                persistence_log = part.loc[part["method"] == "persistence", "test_log_score_mean"]
                if not persistence_log.empty:
                    denom_log = max(float(persistence_log.iloc[0]), 1e-12)
                    out.loc[part.index, "test_log_score_ratio_to_persistence"] = (
                        part["test_log_score_mean"].astype(float) / denom_log
                    )
    return out, pd.DataFrame(ref_rows)


def download_url(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as resp:
        path.write_bytes(resp.read())
    return path


def block_average_to_length(X: np.ndarray, target_len: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if X.shape[1] == target_len:
        return X.copy()
    if X.shape[1] < target_len:
        raise ValueError("cannot upsample to target_len")
    if X.shape[1] % target_len == 0:
        block = X.shape[1] // target_len
        return X.reshape(X.shape[0], target_len, block).mean(axis=2)
    pieces = np.array_split(X, target_len, axis=1)
    return np.column_stack([piece.mean(axis=1) for piece in pieces])


def _ensure_hydraulic_archive() -> Path:
    target = DATA / "hydraulic_systems_redownload.zip"
    if target.exists():
        try:
            with zipfile.ZipFile(target) as zf:
                if "profile.txt" in zf.namelist():
                    return target
        except zipfile.BadZipFile:
            target.unlink()
    return download_url(
        "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
        target,
    )


def hydraulic_gaussians(
    *,
    target_len: int = 60,
    max_cycles: int = 0,
    ridge: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    archive = _ensure_hydraulic_archive()
    channels = []
    with zipfile.ZipFile(archive) as zf:
        for name in HYDRAULIC_SENSOR_FILES:
            with zf.open(name) as fh:
                values = pd.read_csv(fh, sep="\t", header=None).to_numpy(float)
            channels.append(block_average_to_length(values, target_len=target_len))
    n_cycles = min(channel.shape[0] for channel in channels)
    if max_cycles > 0:
        n_cycles = min(n_cycles, max_cycles)
    X = np.stack([channel[:n_cycles] for channel in channels], axis=2)

    train_cycles = max(1, int(0.45 * n_cycles))
    flat = X[:train_cycles].reshape(-1, X.shape[2])
    center = flat.mean(axis=0)
    scale = flat.std(axis=0)
    keep = scale > 1e-10
    scale[~keep] = 1.0
    Z = (X[:, :, keep] - center[keep]) / scale[keep]

    means = Z.mean(axis=1)
    covs = []
    for W in Z:
        C = np.cov(W, rowvar=False)
        covs.append(project_spd(C + ridge * np.eye(C.shape[0]), eps=1e-8))
    return means, np.asarray(covs)


def rolling_gaussians_from_array(
    X: np.ndarray,
    *,
    window: int,
    step: int,
    max_matrices: int = 900,
    ridge: float = 1e-5,
    standardize: bool = True,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if standardize:
        scale_end = max(window, int(0.45 * len(X)))
        mu = X[:scale_end].mean(axis=0)
        sd = X[:scale_end].std(axis=0)
        sd[sd < 1e-8] = 1.0
        X = (X - mu) / sd
    starts = list(range(0, len(X) - window + 1, step))
    if not starts:
        d = X.shape[1]
        if return_windows:
            return np.empty((0, d)), np.empty((0, d, d)), np.empty((0, window, d)), np.empty((0,), dtype=int)
        return np.empty((0, d)), np.empty((0, d, d))
    if len(starts) > max_matrices:
        idx = np.linspace(0, len(starts) - 1, max_matrices, dtype=int)
        starts = [starts[i] for i in idx]
    means = []
    covs = []
    windows = []
    for start in starts:
        W = X[start : start + window]
        C = np.cov(W, rowvar=False)
        C = np.atleast_2d(C)
        means.append(W.mean(axis=0))
        covs.append(project_spd(C + ridge * np.eye(C.shape[0]), eps=1e-8))
        if return_windows:
            windows.append(W.copy())
    if return_windows:
        return np.asarray(means), np.asarray(covs), np.asarray(windows), np.asarray(starts, dtype=int)
    return np.asarray(means), np.asarray(covs)


def rolling_gaussians_from_frame(
    X: pd.DataFrame,
    *,
    window: int,
    step: int,
    max_matrices: int = 900,
    ridge: float = 1e-5,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.loc[:, X.std(axis=0) > 1e-10]
    return rolling_gaussians_from_array(
        X.to_numpy(float),
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=ridge,
        standardize=True,
        return_windows=return_windows,
    )


def download_wpp_population_age5_percentage(*, force: bool = False) -> Path:
    WPP_POPULATION_AGE5_PERCENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if WPP_POPULATION_AGE5_PERCENT_CACHE.exists() and not force:
        return WPP_POPULATION_AGE5_PERCENT_CACHE
    req = Request(WPP_POPULATION_AGE5_PERCENT_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as resp, WPP_POPULATION_AGE5_PERCENT_CACHE.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return WPP_POPULATION_AGE5_PERCENT_CACHE


@lru_cache(maxsize=1)
def load_wpp_population_age5_percentage(path: str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else download_wpp_population_age5_percentage()
    usecols = [
        "ISO3_code",
        "LocTypeName",
        "Location",
        "Time",
        "AgeGrp",
        "AgeGrpStart",
        "PopTotal",
    ]
    frame = pd.read_csv(csv_path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    return frame


def wpp_population_age_gaussians_from_frame(
    frame: pd.DataFrame,
    *,
    iso3_codes: tuple[str, ...] | None = None,
    start_year: int = 1950,
    end_year: int = 2100,
    age_starts: tuple[int, ...] | None = None,
    value_col: str = "PopTotal",
    loc_type_names: tuple[str, ...] = ("Country/Area",),
    min_locations: int = 3,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    df = frame.copy()
    df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
    required = {"ISO3_code", "Time", "AgeGrpStart", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"WPP population frame is missing columns: {', '.join(sorted(missing))}")

    df = df[(df["Time"].astype(int) >= int(start_year)) & (df["Time"].astype(int) <= int(end_year))].copy()
    if loc_type_names and "LocTypeName" in df.columns:
        country_like = df[df["LocTypeName"].isin(loc_type_names)]
        if not country_like.empty:
            df = country_like
    df["ISO3_code"] = df["ISO3_code"].astype(str).str.strip()
    df = df[df["ISO3_code"].ne("") & df["ISO3_code"].ne("nan")]
    if iso3_codes is not None:
        wanted = {code.upper() for code in iso3_codes}
        df = df[df["ISO3_code"].str.upper().isin(wanted)]
    if age_starts is not None:
        wanted_ages = {int(age) for age in age_starts}
        df = df[df["AgeGrpStart"].astype(int).isin(wanted_ages)]

    if df.empty:
        raise ValueError("no WPP population rows remain after filtering")

    pivot = df.pivot_table(
        index=["Time", "ISO3_code"],
        columns="AgeGrpStart",
        values=value_col,
        aggfunc="sum",
    ).sort_index(axis=1)
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()
    if pivot.empty:
        raise ValueError("no complete WPP country-year age profiles remain after pivoting")

    values = pivot.to_numpy(float)
    if np.nanmedian(values) > 1.5:
        values = values / 100.0
    row_sums = values.sum(axis=1)
    good = np.isfinite(row_sums) & (row_sums > 1e-12)
    pivot = pivot.iloc[good]
    values = values[good] / row_sums[good, None]
    normalized = pd.DataFrame(values, index=pivot.index, columns=pivot.columns)

    means: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    years: list[int] = []
    for year, group in normalized.groupby(level="Time", sort=True):
        X = group.to_numpy(float)
        if len(X) < min_locations:
            continue
        C = np.cov(X, rowvar=False)
        C = np.atleast_2d(C)
        means.append(X.mean(axis=0))
        covs.append(project_spd(C + ridge * np.eye(C.shape[0]), eps=1e-8))
        years.append(int(year))

    if not means:
        d = normalized.shape[1]
        return np.empty((0, d)), np.empty((0, d, d))
    return np.asarray(means), np.asarray(covs)


def wpp_population_age_gaussians(
    *,
    group: str = "countries",
    start_year: int = 1950,
    end_year: int = 2100,
    age_starts: tuple[int, ...] | None = None,
    min_locations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    group_key = group.lower()
    iso3_codes: tuple[str, ...] | None
    if group_key in {"countries", "all_countries", "global"}:
        iso3_codes = None
    elif group_key == "g20":
        iso3_codes = WPP_G20_ISO3
    elif group_key == "ageing":
        iso3_codes = WPP_AGEING_ISO3
    else:
        raise ValueError(f"unknown WPP population group: {group}")
    return wpp_population_age_gaussians_from_frame(
        load_wpp_population_age5_percentage(),
        iso3_codes=iso3_codes,
        start_year=start_year,
        end_year=end_year,
        age_starts=age_starts,
        min_locations=min_locations,
    )


def wpp_population_recent_age_share_gaussians_from_frame(
    frame: pd.DataFrame,
    *,
    start_year: int = 2020,
    end_year: int = 2100,
    age_starts: tuple[int, ...] = tuple(range(45, 95, 5)),
    value_col: str = "PopTotal",
    loc_type_names: tuple[str, ...] = ("Country/Area",),
    min_locations: int = 3,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    df = frame.copy()
    df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
    required = {"ISO3_code", "Time", "AgeGrpStart", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"WPP population frame is missing columns: {', '.join(sorted(missing))}")

    df = df[(df["Time"].astype(int) >= int(start_year)) & (df["Time"].astype(int) <= int(end_year))].copy()
    if loc_type_names and "LocTypeName" in df.columns:
        country_like = df[df["LocTypeName"].isin(loc_type_names)]
        if not country_like.empty:
            df = country_like
    df["ISO3_code"] = df["ISO3_code"].astype(str).str.strip()
    df = df[df["ISO3_code"].ne("") & df["ISO3_code"].ne("nan")]
    if df.empty:
        raise ValueError("no WPP population rows remain after filtering")

    pivot = df.pivot_table(
        index=["Time", "ISO3_code"],
        columns="AgeGrpStart",
        values=value_col,
        aggfunc="sum",
    ).sort_index(axis=1)
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna()
    if pivot.empty:
        raise ValueError("no complete WPP country-year age profiles remain after pivoting")

    values = pivot.to_numpy(float)
    if np.nanmedian(values) > 1.5:
        values = values / 100.0
    row_sums = values.sum(axis=1)
    good = np.isfinite(row_sums) & (row_sums > 1e-12)
    pivot = pivot.iloc[good]
    values = values[good] / row_sums[good, None]
    normalized = pd.DataFrame(values, index=pivot.index, columns=pivot.columns)

    selected_ages = [age for age in age_starts if age in normalized.columns]
    if not selected_ages:
        raise ValueError("none of the requested WPP age groups are available")
    selected = normalized[selected_ages]

    means: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    for _year, group in selected.groupby(level="Time", sort=True):
        X = group.to_numpy(float)
        if len(X) < min_locations:
            continue
        C = np.atleast_2d(np.cov(X, rowvar=False))
        means.append(X.mean(axis=0))
        covs.append(project_spd(C + ridge * np.eye(C.shape[0]), eps=1e-8))

    if not means:
        d = selected.shape[1]
        return np.empty((0, d)), np.empty((0, d, d))
    return np.asarray(means), np.asarray(covs)


def wpp_population_recent_age_share_gaussians(
    *,
    start_year: int = 2020,
    end_year: int = 2100,
    age_starts: tuple[int, ...] = tuple(range(45, 95, 5)),
) -> tuple[np.ndarray, np.ndarray]:
    return wpp_population_recent_age_share_gaussians_from_frame(
        load_wpp_population_age5_percentage(),
        start_year=start_year,
        end_year=end_year,
        age_starts=age_starts,
    )


def har_gaussians_for_subject(subject: int) -> tuple[np.ndarray, np.ndarray]:
    base = download_uci_har()
    windows = []
    for split in ["train", "test"]:
        X, subjects, _labels = read_har_split(base, split)
        idx = np.where(subjects == subject)[0]
        windows.extend([X[i] for i in idx])
    if not windows:
        return np.empty((0, 9)), np.empty((0, 9, 9))
    scale_end = max(1, int(0.45 * len(windows)))
    stacked = np.vstack(windows[:scale_end])
    mu = stacked.mean(axis=0)
    sd = stacked.std(axis=0)
    sd[sd < 1e-8] = 1.0
    means = []
    covs = []
    for W in windows:
        Z = (W - mu) / sd
        means.append(Z.mean(axis=0))
        covs.append(project_spd(np.cov(Z, rowvar=False) + 1e-6 * np.eye(Z.shape[1]), eps=1e-8))
    return np.asarray(means), np.asarray(covs)


def appliances_gaussians(window: int = 144, step: int = 24) -> tuple[np.ndarray, np.ndarray]:
    target = DATA / "energydata_complete.csv"
    download_url(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
        target,
    )
    df = pd.read_csv(target)
    cols = ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T_out", "RH_out", "Windspeed", "Tdewpoint"]
    return rolling_gaussians_from_frame(df[cols], window=window, step=step, max_matrices=700)


def sml2010_gaussians(
    window: int = 96,
    step: int = 16,
    max_matrices: int = 900,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = DATA / "sml2010" / "sml2010.zip"
    download_url("https://archive.ics.uci.edu/static/public/274/sml2010.zip", target)
    cols = [
        "Temperature_Comedor_Sensor",
        "Temperature_Habitacion_Sensor",
        "Weather_Temperature",
        "CO2_Comedor_Sensor",
        "CO2_Habitacion_Sensor",
        "Humedad_Comedor_Sensor",
        "Humedad_Habitacion_Sensor",
        "Lighting_Comedor_Sensor",
        "Lighting_Habitacion_Sensor",
        "Precipitacion",
        "Meteo_Exterior_Crepusculo",
        "Meteo_Exterior_Viento",
        "Meteo_Exterior_Sol_Oest",
        "Meteo_Exterior_Sol_Est",
        "Meteo_Exterior_Sol_Sud",
        "Meteo_Exterior_Piranometro",
        "Exterior_Entalpic_1",
        "Exterior_Entalpic_2",
        "Exterior_Entalpic_turbo",
        "Temperature_Exterior_Sensor",
        "Humedad_Exterior_Sensor",
    ]
    frames = []
    with zipfile.ZipFile(target) as zf:
        for name in sorted(n for n in zf.namelist() if n.upper().endswith(".TXT")):
            with zf.open(name) as fh:
                frame = pd.read_csv(fh, sep=r"\s+", comment="#", header=None, engine="python")
            if frame.shape[1] < 23:
                raise ValueError(f"SML2010 file {name} has {frame.shape[1]} columns, expected at least 23")
            frame = frame.iloc[:, 2:23].copy()
            frame.columns = cols
            frames.append(frame)
    if not frames:
        if return_windows:
            return (
                np.empty((0, len(cols))),
                np.empty((0, len(cols), len(cols))),
                np.empty((0, window, len(cols))),
                np.empty((0,), dtype=int),
            )
        return np.empty((0, len(cols))), np.empty((0, len(cols), len(cols)))
    df = pd.concat(frames, ignore_index=True)
    return rolling_gaussians_from_frame(
        df[cols],
        window=window,
        step=step,
        max_matrices=max_matrices,
        return_windows=return_windows,
    )


def _ensure_solar_energy_archive() -> Path:
    target = DATA / "solar_energy" / "solar_AL.txt.gz"
    return download_url(
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
        target,
    )


def load_solar_energy_matrix(path: Path | None = None) -> np.ndarray:
    target = path if path is not None else _ensure_solar_energy_archive()
    return np.loadtxt(target, delimiter=",")


def solar_energy_hd_gaussians(
    window: int = 144,
    step: int = 24,
    n_sensors: int = 40,
    max_matrices: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    X = load_solar_energy_matrix()
    if X.ndim != 2 or X.shape[1] < n_sensors:
        raise ValueError(f"Solar Energy matrix has shape {X.shape}, cannot select {n_sensors} sensors")
    scale_end = max(window, int(0.45 * len(X)))
    early = X[:scale_end]
    nonzero = np.mean(early > 0.0, axis=0)
    variability = np.std(early, axis=0)
    selected = np.argsort(nonzero * variability)[::-1][:n_sensors]
    X_sel = np.sqrt(np.clip(X[:, selected], 0.0, None))
    return rolling_gaussians_from_array(
        X_sel,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
    )


def _ensure_electricity_archive() -> Path:
    target = DATA / "electricity_benchmark" / "electricity.txt.gz"
    return download_url(
        "https://cdn.jsdelivr.net/gh/laiguokun/multivariate-time-series-data@master/electricity/electricity.txt.gz",
        target,
    )


def load_electricity_matrix(path: Path | None = None) -> np.ndarray:
    target = path if path is not None else _ensure_electricity_archive()
    return np.loadtxt(target, delimiter=",")


def electricity_hd_gaussians(
    window: int = 168,
    step: int = 24,
    n_clients: int = 40,
    max_matrices: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    X = load_electricity_matrix()
    if X.ndim != 2 or X.shape[1] < n_clients:
        raise ValueError(f"Electricity matrix has shape {X.shape}, cannot select {n_clients} clients")
    scale_end = max(window, int(0.45 * len(X)))
    early = X[:scale_end]
    nonzero = np.mean(early > 0.0, axis=0)
    variability = np.std(np.log1p(np.clip(early, 0.0, None)), axis=0)
    selected = np.argsort(nonzero * variability)[::-1][:n_clients]
    X_sel = np.log1p(np.clip(X[:, selected], 0.0, None))
    return rolling_gaussians_from_array(
        X_sel,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
    )


def _beijing_air_url(station: str) -> str:
    file = f"PRSA_Data_{station}_20130301-20170228.csv"
    return (
        "https://cdn.jsdelivr.net/gh/Afkerian/Beijing-Multi-Site-Air-Quality-Data-Data-Set@main/"
        "data/beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228/"
        f"{file}"
    )


def _ensure_beijing_air_files() -> list[Path]:
    paths = []
    for station in BEIJING_AIR_STATIONS:
        target = DATA / "beijing_air_quality" / f"PRSA_Data_{station}_20130301-20170228.csv"
        paths.append(download_url(_beijing_air_url(station), target))
    return paths


def load_beijing_air_quality_frame() -> pd.DataFrame:
    series = []
    for path in _ensure_beijing_air_files():
        df = pd.read_csv(path)
        station = str(df["station"].dropna().iloc[0])
        dt = pd.to_datetime(
            {
                "year": df["year"].astype(int),
                "month": df["month"].astype(int),
                "day": df["day"].astype(int),
                "hour": df["hour"].astype(int),
            }
        )
        sub = df.loc[:, BEIJING_AIR_FEATURES].astype(float)
        sub = np.log1p(sub.clip(lower=0.0))
        sub.index = dt
        sub.columns = [f"{station}_{col}" for col in BEIJING_AIR_FEATURES]
        series.append(sub)
    frame = pd.concat(series, axis=1).sort_index()
    frame = frame.interpolate(method="time", limit_direction="both")
    return frame.dropna(axis=1, how="all").dropna(axis=0, how="any")


def beijing_air_hd_gaussians(
    window: int = 168,
    step: int = 24,
    max_features: int = 48,
    max_matrices: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_beijing_air_quality_frame()
    if df.shape[1] < max_features:
        raise ValueError(f"Beijing air-quality frame has {df.shape[1]} features, cannot select {max_features}")
    scale_end = max(window, int(0.45 * len(df)))
    scores = df.iloc[:scale_end].std(axis=0).to_numpy()
    selected = np.argsort(scores)[::-1][:max_features]
    X = df.iloc[:, selected].to_numpy(dtype=float)
    return rolling_gaussians_from_array(
        X,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
    )


def _ensure_intel_lab_archive() -> Path:
    target = DATA / "intel_lab" / "data.zip"
    return download_url("https://github.com/linsea423/Intel_Lab_Data/raw/master/data.zip", target)


def load_intel_lab_sensor_frame(
    bin_minutes: int = 10,
    interpolation_limit: int = 24,
) -> pd.DataFrame:
    target = _ensure_intel_lab_archive()
    with zipfile.ZipFile(target) as zf:
        with zf.open("data.txt") as fh:
            df = pd.read_csv(
                fh,
                sep=r"\s+",
                header=None,
                names=["date", "time", "epoch", "moteid", "temp", "humidity", "light", "voltage"],
                engine="c",
            )
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["timestamp", "moteid"]).copy()
    for col, lo, hi in (
        ("temp", 0.0, 50.0),
        ("humidity", 0.0, 100.0),
        ("light", 0.0, 2000.0),
        ("voltage", 2.0, 3.2),
    ):
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    df["bin"] = df["timestamp"].dt.floor(f"{bin_minutes}min")
    grouped = (
        df.groupby(["bin", "moteid"], observed=True)[list(INTEL_LAB_MEASURES)]
        .mean()
        .reset_index()
    )
    wide = grouped.pivot(index="bin", columns="moteid", values=list(INTEL_LAB_MEASURES)).sort_index()
    wide.columns = [f"mote{int(mote)}_{measure}" for measure, mote in wide.columns]
    wide = wide.interpolate(method="time", limit=interpolation_limit, limit_direction="both")
    return wide


def intel_lab_gaussians(
    window: int = 144,
    step: int = 24,
    n_motes: int = 10,
    max_matrices: int = 900,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_intel_lab_sensor_frame()
    coverage = df.notna().mean(axis=0)
    motes = sorted({int(col.split("_")[0][4:]) for col in df.columns})
    rows = []
    for mote in motes:
        cols = [f"mote{mote}_{measure}" for measure in INTEL_LAB_MEASURES]
        if all(col in df.columns for col in cols):
            rows.append((mote, float(coverage[cols].min()), float(coverage[cols].mean())))
    rank = pd.DataFrame(rows, columns=["mote", "min_coverage", "mean_coverage"])
    rank["score"] = rank["min_coverage"] * rank["mean_coverage"]
    rank = rank.sort_values("score", ascending=False)
    if len(rank) < n_motes:
        raise ValueError(f"Intel Lab data has {len(rank)} complete motes, cannot select {n_motes}")
    selected_motes = [int(x) for x in rank.head(n_motes)["mote"]]
    cols = [f"mote{mote}_{measure}" for mote in selected_motes for measure in INTEL_LAB_MEASURES]
    X_df = df[cols].dropna(axis=0, how="any").copy()
    for col in X_df.columns:
        if col.endswith("_light"):
            X_df[col] = np.log1p(X_df[col].clip(lower=0.0))
    return rolling_gaussians_from_array(
        X_df.to_numpy(dtype=float),
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
        return_windows=return_windows,
    )


@lru_cache(maxsize=1)
def load_hai21_process_frame(path: str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else HAI21_TRAIN1_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HAI 21.03 train stream not found at {csv_path}; expected local train1.csv.gz cache."
        )
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include="number").copy()
    label_cols = [col for col in numeric.columns if col == "attack" or col.startswith("attack_")]
    numeric = numeric.drop(columns=label_cols, errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction="both").dropna(axis=1, how="all")
    std = numeric.std(axis=0)
    return numeric.loc[:, std > 1e-10]


def hai21_gaussians(
    window: int = 600,
    step: int = 600,
    n_features: int = 30,
    max_matrices: int = 900,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_hai21_process_frame()
    variability = df.std(axis=0).sort_values(ascending=False)
    if len(variability) < n_features:
        raise ValueError(f"HAI 21.03 data has {len(variability)} varying process variables, cannot select {n_features}")
    selected = list(variability.head(n_features).index)
    return rolling_gaussians_from_array(
        df[selected].to_numpy(dtype=float),
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
        return_windows=return_windows,
    )


def _ensure_melbourne_pedestrian_csv() -> Path:
    target = DATA / "melbourne_pedestrian" / "pedestrian_counts.csv"
    return download_url(MELBOURNE_PEDESTRIAN_URL, target)


def load_melbourne_pedestrian_frame() -> pd.DataFrame:
    target = _ensure_melbourne_pedestrian_csv()
    df = pd.read_csv(
        target,
        usecols=["location_id", "sensing_date", "hourday", "pedestriancount"],
    )
    timestamp = pd.to_datetime(df["sensing_date"]) + pd.to_timedelta(df["hourday"].astype(int), unit="h")
    df = df.assign(timestamp=timestamp)
    wide = (
        df.pivot_table(index="timestamp", columns="location_id", values="pedestriancount", aggfunc="sum")
        .sort_index()
    )
    hourly_index = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    return wide.reindex(hourly_index)


def melbourne_pedestrian_gaussians(
    window: int = 48,
    step: int = 12,
    n_sensors: int = 40,
    max_matrices: int = 650,
    interpolation_limit: int = 24,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_melbourne_pedestrian_frame()
    scale_end = max(window, int(0.45 * len(df)))
    early = df.iloc[:scale_end]
    transformed_early = np.log1p(early.clip(lower=0.0))
    coverage = early.notna().mean(axis=0)
    variability = transformed_early.std(axis=0, skipna=True)
    score = (coverage * variability).replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < n_sensors:
        raise ValueError(f"Melbourne pedestrian data has {len(score)} usable sensors, cannot select {n_sensors}")
    selected = list(score.sort_values(ascending=False).head(n_sensors).index)
    X_df = np.log1p(df[selected].clip(lower=0.0))
    X_df = X_df.interpolate(limit=interpolation_limit, limit_direction="both").dropna(axis=0, how="any")
    return rolling_gaussians_from_array(
        X_df.to_numpy(dtype=float),
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
        return_windows=return_windows,
    )


def occupancy_gaussians(split: str, window: int = 144, step: int = 24) -> tuple[np.ndarray, np.ndarray]:
    target = DATA / "occupancy_detection.zip"
    download_url("https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip", target)
    with zipfile.ZipFile(target) as zf:
        raw = zf.read(f"{split}.txt")
    df = pd.read_csv(io.BytesIO(raw))
    cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    return rolling_gaussians_from_frame(df[cols], window=window, step=step, max_matrices=900)


def room_occupancy_gaussians(
    window: int = 96,
    step: int = 24,
    max_matrices: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    target = DATA / "room_occupancy_estimation.zip"
    download_url("https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip", target)
    with zipfile.ZipFile(target) as zf:
        raw = zf.read("Occupancy_Estimation.csv")
    df = pd.read_csv(io.BytesIO(raw))
    cols = [
        "S1_Temp",
        "S2_Temp",
        "S3_Temp",
        "S4_Temp",
        "S1_Light",
        "S2_Light",
        "S3_Light",
        "S4_Light",
        "S1_Sound",
        "S2_Sound",
        "S3_Sound",
        "S4_Sound",
        "S5_CO2",
        "S5_CO2_Slope",
        "S6_PIR",
        "S7_PIR",
    ]
    return rolling_gaussians_from_frame(df[cols], window=window, step=step, max_matrices=max_matrices)


def household_power_gaussians(
    window: int = 1440,
    step: int = 360,
    max_matrices: int = 900,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = DATA / "household_power_consumption.zip"
    download_url(
        "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        target,
    )
    cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    with zipfile.ZipFile(target) as zf:
        with zf.open("household_power_consumption.txt") as fh:
            df = pd.read_csv(fh, sep=";", usecols=cols, na_values="?", low_memory=False)
    return rolling_gaussians_from_frame(
        df[cols],
        window=window,
        step=step,
        max_matrices=max_matrices,
        return_windows=return_windows,
    )


def fetch_yahoo_adjclose_cached(
    symbol: str,
    *,
    start: str = "2014-01-01",
    end: str = "2026-05-31",
) -> pd.Series:
    cache = DATA / "finance_yahoo_cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{symbol}_{start}_{end}.csv"
    if path.exists():
        ser = pd.read_csv(path, parse_dates=["date"]).set_index("date")["adjclose"]
        ser.name = symbol
        return ser.dropna()
    ser = fetch_yahoo_adjclose(symbol, start=start, end=end)
    out = pd.DataFrame({"date": ser.index, "adjclose": ser.to_numpy(float)})
    out.to_csv(path, index=False)
    return ser


def finance_gaussians_from_prices(
    prices: pd.DataFrame,
    *,
    window: int = 60,
    step: int = 5,
    max_matrices: int = 900,
    ridge: float = 1e-5,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = prices.replace([np.inf, -np.inf], np.nan).dropna()
    returns = 100.0 * np.log(px).diff().dropna()
    return rolling_gaussians_from_frame(
        returns,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=ridge,
        return_windows=return_windows,
    )


def finance_etf_hd_gaussians(
    *,
    window: int = 60,
    step: int = 5,
    max_matrices: int = 900,
    symbols: tuple[str, ...] = FINANCE_ETF_HD_SYMBOLS,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    series = [fetch_yahoo_adjclose_cached(symbol) for symbol in symbols]
    prices = pd.concat(series, axis=1, join="inner").sort_index()
    prices.columns = list(symbols)
    return finance_gaussians_from_prices(
        prices,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        return_windows=return_windows,
    )


def gas_drift_gaussians(window: int = 96, step: int = 24, n_features: int = 8) -> tuple[np.ndarray, np.ndarray]:
    target = DATA / "gas_sensor_drift.zip"
    download_url("https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip", target)
    rows = []
    with zipfile.ZipFile(target) as zf:
        for name in sorted([n for n in zf.namelist() if n.endswith(".dat")]):
            for line in zf.read(name).decode("utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                values = np.full(n_features, np.nan, dtype=float)
                for token in parts[1:]:
                    if ":" not in token:
                        continue
                    idx_s, val_s = token.split(":", 1)
                    idx = int(idx_s) - 1
                    if 0 <= idx < n_features:
                        values[idx] = float(val_s)
                if np.all(np.isfinite(values)):
                    rows.append(values)
    X = pd.DataFrame(np.asarray(rows), columns=[f"gas_{j + 1}" for j in range(n_features)])
    return rolling_gaussians_from_frame(X, window=window, step=step, max_matrices=900)


def hapt_gaussians_for_subject(subject: int, window: int = 128, step: int = 64) -> tuple[np.ndarray, np.ndarray]:
    from bwar.da_bwar_baseline_screen import download_hapt

    base = download_hapt() / "RawData"
    acc_files = sorted(base.glob(f"acc_exp*_user{subject:02d}.txt"))
    blocks = []
    for acc_path in acc_files:
        gyro_path = acc_path.with_name(acc_path.name.replace("acc_", "gyro_"))
        if not gyro_path.exists():
            continue
        acc = np.loadtxt(acc_path)
        gyro = np.loadtxt(gyro_path)
        blocks.append(np.column_stack([acc, gyro]))
    if not blocks:
        return np.empty((0, 6)), np.empty((0, 6, 6))
    return rolling_gaussians_from_array(np.vstack(blocks), window=window, step=step, max_matrices=900)


def _ensure_mhealth_dir() -> Path:
    base = DATA / "mhealth" / "MHEALTHDATASET"
    if base.exists():
        return base
    archive = DATA / "mhealth.zip"
    if not archive.exists():
        raise FileNotFoundError(f"MHEALTH archive not found: {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA / "mhealth")
    return base


def mhealth_gaussians_for_subject(
    subject: int,
    window: int = 200,
    step: int = 50,
    max_matrices: int = 900,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = _ensure_mhealth_dir()
    path = base / f"mHealth_subject{subject}.log"
    if not path.exists():
        if return_windows:
            return (
                np.empty((0, 21)),
                np.empty((0, 21, 21)),
                np.empty((0, window, 21)),
                np.empty((0,), dtype=int),
            )
        return np.empty((0, 21)), np.empty((0, 21, 21))
    arr = np.loadtxt(path)
    labels = arr[:, -1].astype(int)
    motion_cols = [0, 1, 2] + list(range(5, 23))
    X = arr[labels > 0][:, motion_cols].astype(float)
    if len(X) < window:
        if return_windows:
            return (
                np.empty((0, len(motion_cols))),
                np.empty((0, len(motion_cols), len(motion_cols))),
                np.empty((0, window, len(motion_cols))),
                np.empty((0,), dtype=int),
            )
        return np.empty((0, len(motion_cols))), np.empty((0, len(motion_cols), len(motion_cols)))
    return rolling_gaussians_from_array(
        X,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
        return_windows=return_windows,
    )


def _ensure_pamap2_archive() -> Path:
    target = DATA / "pamap2.zip"
    return download_url(
        "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        target,
    )


def pamap2_gaussians_from_archive(
    archive: Path,
    *,
    subject: int,
    window: int = 256,
    step: int = 64,
    stride: int = 5,
    max_matrices: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    file_id = subject if subject >= 100 else 100 + subject
    target_suffix = f"Protocol/subject{file_id}.dat"
    with zipfile.ZipFile(archive) as zf:
        matches = [name for name in zf.namelist() if name.endswith(target_suffix)]
        if not matches:
            return np.empty((0, len(PAMAP2_FEATURE_COLUMNS))), np.empty(
                (0, len(PAMAP2_FEATURE_COLUMNS), len(PAMAP2_FEATURE_COLUMNS))
            )
        with zf.open(matches[0]) as fh:
            df = pd.read_csv(fh, sep=r"\s+", header=None, na_values="NaN", engine="python")

    if df.shape[1] < 54:
        raise ValueError(f"PAMAP2 subject file has {df.shape[1]} columns, expected at least 54")
    df = df.loc[df[1].astype(float) != 0].reset_index(drop=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    X = df.iloc[:, list(PAMAP2_FEATURE_COLUMNS)].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.interpolate(limit_direction="both").ffill().bfill()
    X = X.loc[:, X.std(axis=0) > 1e-10].dropna()
    if X.empty:
        return np.empty((0, len(PAMAP2_FEATURE_COLUMNS))), np.empty(
            (0, len(PAMAP2_FEATURE_COLUMNS), len(PAMAP2_FEATURE_COLUMNS))
        )
    return rolling_gaussians_from_array(
        X.to_numpy(float),
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
    )


def pamap2_gaussians_for_subject(
    subject: int,
    window: int = 256,
    step: int = 64,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    return pamap2_gaussians_from_archive(
        _ensure_pamap2_archive(),
        subject=subject,
        window=window,
        step=step,
        stride=stride,
        max_matrices=900,
    )


def _yield_window_job(
    *,
    return_windows: bool,
    job: str,
    dataset: str,
    result,
    meta: dict,
    window: int,
):
    if return_windows:
        means, covs, raw_windows, starts = result
        yield job, dataset, means, covs, meta, {
            "raw_windows": raw_windows,
            "window_starts": starts,
            "window_size": int(window),
        }
    else:
        means, covs = result
        yield job, dataset, means, covs, meta


def iter_dataset_jobs(dataset: str, *, quick: bool, return_windows: bool = False):
    if dataset in {"all", "uci_har"}:
        subjects = range(1, 11 if quick else 31)
        for subject in subjects:
            means, covs = har_gaussians_for_subject(subject)
            yield f"uci_har_s{subject}", "uci_har", means, covs, {"subject": int(subject)}
    if dataset in {"all", "appliances"}:
        configs = [(72, 12)] if quick else [(72, 12), (144, 24), (288, 48)]
        for window, step in configs:
            means, covs = appliances_gaussians(window=window, step=step)
            yield f"appliances_w{window}_s{step}", "appliances", means, covs, {"window": window, "step": step}
    if dataset in {"sml2010"}:
        configs = [(48, 8)] if quick else [(48, 8), (96, 16), (192, 32)]
        for window, step in configs:
            result = sml2010_gaussians(window=window, step=step, return_windows=return_windows)
            yield from _yield_window_job(
                return_windows=return_windows,
                job=f"sml2010_w{window}_s{step}",
                dataset="sml2010",
                result=result,
                meta={"window": window, "step": step},
                window=window,
            )
    if dataset in {"solar_energy_hd"}:
        configs = [(144, 24, 30)] if quick else [(144, 24, 40), (288, 48, 40), (576, 96, 40)]
        for window, step, n_sensors in configs:
            means, covs = solar_energy_hd_gaussians(window=window, step=step, n_sensors=n_sensors)
            yield (
                f"solar_energy_hd_w{window}_s{step}_d{n_sensors}",
                "solar_energy_hd",
                means,
                covs,
                {"window": window, "step": step, "n_sensors": n_sensors, "transform": "sqrt"},
            )
    if dataset in {"electricity_hd"}:
        configs = [(168, 24, 30)] if quick else [(168, 24, 40), (336, 48, 40), (720, 72, 40)]
        for window, step, n_clients in configs:
            means, covs = electricity_hd_gaussians(window=window, step=step, n_clients=n_clients)
            yield (
                f"electricity_hd_w{window}_s{step}_d{n_clients}",
                "electricity_hd",
                means,
                covs,
                {"window": window, "step": step, "n_clients": n_clients, "transform": "log1p"},
            )
    if dataset in {"beijing_air_hd"}:
        configs = [(168, 24, 36)] if quick else [(168, 24, 48), (336, 48, 48), (720, 72, 48)]
        for window, step, max_features in configs:
            means, covs = beijing_air_hd_gaussians(window=window, step=step, max_features=max_features)
            yield (
                f"beijing_air_hd_w{window}_s{step}_d{max_features}",
                "beijing_air_hd",
                means,
                covs,
                {"window": window, "step": step, "max_features": max_features, "transform": "log1p"},
            )
    if dataset in {"intel_lab"}:
        configs = [(144, 24, 10)] if quick else [(72, 12, 10), (144, 24, 12), (216, 36, 15)]
        for window, step, n_motes in configs:
            result = intel_lab_gaussians(window=window, step=step, n_motes=n_motes, return_windows=return_windows)
            yield from _yield_window_job(
                return_windows=return_windows,
                job=f"intel_lab_w{window}_s{step}_m{n_motes}",
                dataset="intel_lab",
                result=result,
                meta={"window": window, "step": step, "n_motes": n_motes, "bin_minutes": 10},
                window=window,
            )
    if dataset in {"hai21"}:
        configs = [(600, 600, 30)] if quick else [(600, 600, 30), (1200, 600, 30), (1800, 900, 30)]
        for window, step, n_features in configs:
            result = hai21_gaussians(
                window=window,
                step=step,
                n_features=n_features,
                return_windows=return_windows,
            )
            yield from _yield_window_job(
                return_windows=return_windows,
                job=f"hai21_w{window}_s{step}_d{n_features}",
                dataset="hai21",
                result=result,
                meta={
                    "window": window,
                    "step": step,
                    "n_features": n_features,
                    "source": "HAI 21.03 train1 industrial control stream",
                    "time_unit": "second",
                },
                window=window,
            )
    if dataset in {"melbourne_pedestrian"}:
        configs = [(48, 12, 40)] if quick else [(24, 6, 40), (48, 12, 40), (168, 24, 40)]
        for window, step, n_sensors in configs:
            result = melbourne_pedestrian_gaussians(
                window=window,
                step=step,
                n_sensors=n_sensors,
                return_windows=return_windows,
            )
            yield from _yield_window_job(
                return_windows=return_windows,
                job=f"melbourne_pedestrian_w{window}_s{step}_d{n_sensors}",
                dataset="melbourne_pedestrian",
                result=result,
                meta={
                    "window": window,
                    "step": step,
                    "n_sensors": n_sensors,
                    "transform": "log1p",
                    "source": "City of Melbourne Open Data",
                },
                window=window,
            )
    if dataset in {"all", "occupancy"}:
        splits = ["datatraining"] if quick else ["datatraining", "datatest", "datatest2"]
        configs = [(72, 12)] if quick else [(72, 12), (144, 24), (288, 48)]
        for split in splits:
            for window, step in configs:
                means, covs = occupancy_gaussians(split=split, window=window, step=step)
                yield f"occupancy_{split}_w{window}", "occupancy", means, covs, {"split": split, "window": window, "step": step}
    if dataset in {"all", "room_occupancy"}:
        configs = [(48, 12)] if quick else [(48, 12), (96, 24), (192, 48), (288, 72)]
        for window, step in configs:
            means, covs = room_occupancy_gaussians(window=window, step=step)
            yield f"room_occupancy_w{window}_s{step}", "room_occupancy", means, covs, {"window": window, "step": step}
    if dataset in {"all", "household_power"}:
        configs = [(1440, 360)] if quick else [(720, 180), (1440, 360), (2880, 720)]
        for window, step in configs:
            means, covs = household_power_gaussians(window=window, step=step)
            yield f"household_power_w{window}_s{step}", "household_power", means, covs, {"window": window, "step": step}
    if dataset in {"household_power_strong"}:
        configs = [(720, 180)] if quick else [(720, 180), (960, 240), (1440, 360)]
        for window, step in configs:
            result = household_power_gaussians(window=window, step=step, return_windows=return_windows)
            yield from _yield_window_job(
                return_windows=return_windows,
                job=f"household_power_w{window}_s{step}",
                dataset="household_power_strong",
                result=result,
                meta={"window": window, "step": step},
                window=window,
            )
    if dataset in {"all", "gas_drift"}:
        configs = [(64, 16)] if quick else [(64, 16), (96, 24), (128, 32)]
        for window, step in configs:
            means, covs = gas_drift_gaussians(window=window, step=step, n_features=8)
            yield f"gas8_w{window}_s{step}", "gas_drift8", means, covs, {"window": window, "step": step, "n_features": 8}
    if dataset in {"finance_etf_hd"}:
        configs = [(60, 10)] if quick else [(40, 5), (60, 10), (90, 15)]
        for window, step in configs:
            result = finance_etf_hd_gaussians(window=window, step=step, return_windows=return_windows)
            yield from _yield_window_job(
                return_windows=return_windows,
                job=f"finance_etf_hd_w{window}_s{step}",
                dataset="finance_etf_hd",
                result=result,
                meta={"window": window, "step": step, "symbols": ",".join(FINANCE_ETF_HD_SYMBOLS)},
                window=window,
            )
    if dataset in {"wpp_population_age"}:
        configs = [("countries", 1950, 2100, None)] if quick else [
            ("countries", 1950, 2100, None),
            ("g20", 1950, 2100, None),
            ("ageing", 1950, 2100, None),
            ("countries", 1970, 2100, tuple(range(0, 85, 5))),
        ]
        for group, start_year, end_year, age_starts in configs:
            means, covs = wpp_population_age_gaussians(
                group=group,
                start_year=start_year,
                end_year=end_year,
                age_starts=age_starts,
            )
            age_label = "age0-80" if age_starts is not None else "age0-100"
            yield (
                f"wpp_population_age_{group}_{start_year}_{end_year}_{age_label}",
                "wpp_population_age",
                means,
                covs,
                {
                    "group": group,
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "age_grid": "5_year",
                    "age_range": age_label,
                    "source": "UN WPP 2024",
                    "transform": "age_share",
                },
            )
    if dataset in {"wpp_population_recent_age"}:
        configs = [(2020, 2100, tuple(range(45, 95, 5)))] if quick else [
            (2000, 2100, tuple(range(45, 95, 5))),
            (2010, 2100, tuple(range(45, 95, 5))),
            (2020, 2100, tuple(range(45, 95, 5))),
        ]
        for start_year, end_year, age_starts in configs:
            means, covs = wpp_population_recent_age_share_gaussians(
                start_year=start_year,
                end_year=end_year,
                age_starts=age_starts,
            )
            age_label = f"age{min(age_starts)}-{max(age_starts) + 4}"
            yield (
                f"wpp_population_recent_age_{start_year}_{end_year}_{age_label}",
                "wpp_population_recent_age",
                means,
                covs,
                {
                    "start_year": int(start_year),
                    "end_year": int(end_year),
                    "age_grid": "5_year",
                    "age_range": age_label,
                    "source": "UN WPP 2024",
                    "transform": "total_population_age_share",
                },
            )
    if dataset in {"all", "hapt"}:
        subjects = range(1, 6 if quick else 31)
        configs = [(128, 64)] if quick else [(128, 64), (256, 128)]
        for subject in subjects:
            for window, step in configs:
                means, covs = hapt_gaussians_for_subject(subject, window=window, step=step)
                yield f"hapt_s{subject}_w{window}", "hapt", means, covs, {"subject": int(subject), "window": window, "step": step}
    if dataset in {"all", "mhealth_full"}:
        subjects = range(1, 4 if quick else 11)
        configs = [(200, 50)] if quick else [(150, 50), (200, 50), (300, 75)]
        for subject in subjects:
            for window, step in configs:
                result = mhealth_gaussians_for_subject(
                    subject,
                    window=window,
                    step=step,
                    return_windows=return_windows,
                )
                yield from _yield_window_job(
                    return_windows=return_windows,
                    job=f"mhealth_s{subject}_w{window}_s{step}",
                    dataset="mhealth_full",
                    result=result,
                    meta={"subject": int(subject), "window": window, "step": step},
                    window=window,
                )
    if dataset in {"all", "pamap2"}:
        subjects = range(1, 4 if quick else 10)
        configs = [(256, 64, 5)] if quick else [(256, 64, 5), (512, 128, 5)]
        for subject in subjects:
            for window, step, stride in configs:
                means, covs = pamap2_gaussians_for_subject(subject, window=window, step=step, stride=stride)
                yield (
                    f"pamap2_s{subject}_w{window}_stride{stride}",
                    "pamap2",
                    means,
                    covs,
                    {"subject": int(subject), "window": window, "step": step, "stride": stride},
                )
    if dataset in {"all", "hydraulic"}:
        configs = [60] if quick else [30, 60, 120]
        for target_len in configs:
            means, covs = hydraulic_gaussians(target_len=target_len)
            yield f"hydraulic_l{target_len}", "hydraulic_systems", means, covs, {"target_len": int(target_len)}


def summarize_outputs(raw: pd.DataFrame, *, loss_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_col = choose_primary_loss_column(raw, loss_col)
    ok = raw.dropna(subset=[primary_col]).copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = ok[ok["method"] == "bwar_selected_ref"].copy()
    non_bwar = ok[~ok["method"].str.startswith("bwar")].copy()
    best_non_bwar = non_bwar.loc[non_bwar.groupby(["job", "horizon"])[primary_col].idxmin()].copy()
    comp = selected.merge(
        best_non_bwar[["job", "horizon", "method", primary_col]].rename(
            columns={"method": "best_non_bwar_method", primary_col: "best_non_bwar_loss"}
        ),
        on=["job", "horizon"],
        how="left",
    )
    comp["primary_metric"] = primary_col
    comp["target_loss"] = comp[primary_col].astype(float)
    comp["loss_reduction_vs_best_non_bwar"] = comp["best_non_bwar_loss"].astype(float) - comp["target_loss"]
    comp["gain_vs_best_non_bwar"] = relative_loss_reduction(comp["target_loss"], comp["best_non_bwar_loss"])
    comp["selected_is_best_non_bwar_or_better"] = comp["target_loss"] <= comp["best_non_bwar_loss"]
    if "domain_metric" not in comp.columns:
        comp["domain_metric"] = ""
    if "domain_metric_label" not in comp.columns:
        comp["domain_metric_label"] = ""
    if primary_col == "test_domain_loss_mean":
        comp["best_non_bwar_domain_loss"] = comp["best_non_bwar_loss"]
    if primary_col == "test_w2_mean":
        comp["best_non_bwar_w2"] = comp["best_non_bwar_loss"]
    if primary_col == "test_log_score_mean":
        comp["best_non_bwar_log_score"] = comp["best_non_bwar_loss"]
    summary_h = (
        comp.groupby(["dataset", "horizon"], as_index=False)
        .agg(
            primary_metric=("primary_metric", "first"),
            domain_metric=("domain_metric", "first"),
            domain_metric_label=("domain_metric_label", "first"),
            mean_gain=("gain_vs_best_non_bwar", "mean"),
            median_gain=("gain_vs_best_non_bwar", "median"),
            mean_loss_reduction=("loss_reduction_vs_best_non_bwar", "mean"),
            median_loss_reduction=("loss_reduction_vs_best_non_bwar", "median"),
            positive_rate=("gain_vs_best_non_bwar", lambda x: float(np.mean(np.asarray(x) > 0))),
            best_rate=("selected_is_best_non_bwar_or_better", "mean"),
            n_jobs=("job", "size"),
        )
        .sort_values(["dataset", "horizon"])
    )
    summary = (
        comp.groupby("dataset", as_index=False)
        .agg(
            primary_metric=("primary_metric", "first"),
            domain_metric=("domain_metric", "first"),
            domain_metric_label=("domain_metric_label", "first"),
            mean_gain=("gain_vs_best_non_bwar", "mean"),
            median_gain=("gain_vs_best_non_bwar", "median"),
            mean_loss_reduction=("loss_reduction_vs_best_non_bwar", "mean"),
            median_loss_reduction=("loss_reduction_vs_best_non_bwar", "median"),
            positive_rate=("gain_vs_best_non_bwar", lambda x: float(np.mean(np.asarray(x) > 0))),
            best_rate=("selected_is_best_non_bwar_or_better", "mean"),
            n_jobs=("job", "size"),
        )
        .sort_values("mean_gain", ascending=False)
    )
    return summary, summary_h


def write_outputs(
    rows: list[pd.DataFrame],
    ref_tables: list[pd.DataFrame],
    *,
    out_dir: Path,
    tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    raw.to_csv(out_dir / f"raw_results_{tag}.csv", index=False)
    if ref_tables:
        pd.concat(ref_tables, ignore_index=True).to_csv(out_dir / f"reference_table_{tag}.csv", index=False)
    if raw.empty:
        return
    summary, summary_h = summarize_outputs(raw)
    summary.to_csv(out_dir / f"summary_by_dataset_{tag}.csv", index=False)
    summary_h.to_csv(out_dir / f"summary_by_dataset_horizon_{tag}.csv", index=False)
    payload = {
        "summary_by_dataset": summary.to_dict(orient="records"),
        "summary_by_dataset_horizon": summary_h.to_dict(orient="records"),
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Theory-matched real-data BWAR rerun.")
    parser.add_argument(
        "--datasets",
        choices=[
            "all",
            "uci_har",
            "appliances",
            "sml2010",
            "solar_energy_hd",
            "electricity_hd",
            "beijing_air_hd",
            "intel_lab",
            "hai21",
            "melbourne_pedestrian",
            "occupancy",
            "room_occupancy",
            "household_power",
            "household_power_strong",
            "gas_drift",
            "finance_etf_hd",
            "wpp_population_age",
            "wpp_population_recent_age",
            "hapt",
            "mhealth_full",
            "pamap2",
            "hydraulic",
        ],
        default="uci_har",
    )
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--ar-model", choices=["diag", "full"], default="diag")
    parser.add_argument("--reference-library", choices=["full", "no_barycenter", "fast"], default="full")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    set_reference_library_mode(args.reference_library)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    tag = f"{args.datasets}_h{'-'.join(map(str, horizons))}_{args.ar_model}" + ("_quick" if args.quick else "")
    if args.reference_library != "full":
        tag += f"_{args.reference_library}"
    started = time.time()
    rows: list[pd.DataFrame] = []
    ref_tables: list[pd.DataFrame] = []
    n_jobs = 0
    for job, dataset, means, covs, meta in iter_dataset_jobs(args.datasets, quick=args.quick):
        if args.max_jobs and n_jobs >= args.max_jobs:
            break
        print(f"Running {job} horizons={horizons} n={len(covs)} d={covs.shape[1] if len(covs) else 'NA'}...")
        raw, refs = run_single_series(
            job=job,
            dataset=dataset,
            means=means,
            covs=covs,
            meta=meta,
            horizons=horizons,
            ar_model=args.ar_model,
        )
        if not raw.empty:
            rows.append(raw)
        if not refs.empty:
            ref_tables.append(refs)
        n_jobs += 1
        write_outputs(rows, ref_tables, out_dir=args.out_dir, tag=tag)
    write_outputs(rows, ref_tables, out_dir=args.out_dir, tag=tag)
    metadata = {
        "datasets": args.datasets,
        "horizons": horizons,
        "ar_model": args.ar_model,
        "quick": bool(args.quick),
        "n_jobs_attempted": int(n_jobs),
        "elapsed_seconds": round(time.time() - started, 3),
        "out_dir": str(args.out_dir),
        "tag": tag,
    }
    (args.out_dir / f"run_metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path = args.out_dir / f"summary_by_dataset_{tag}.csv"
    if summary_path.exists():
        print(pd.read_csv(summary_path).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
