from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.real_bwar_theory_matched import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_RIDGE_GRID,
    bwar_gaussian_decode,
    bwar_gaussian_encode,
    candidate_gaussian_references,
    cholesky_decode,
    cholesky_encode,
    domain_loss_from_moments,
    domain_metric_profile,
    euclidean_decode,
    euclidean_encode,
    fit_var,
    gaussian_log_score_from_moments,
    gaussian_w2_squared,
    iter_dataset_jobs,
    log_euclidean_decode,
    log_euclidean_encode,
    metric_mean_key,
    persistence_metrics,
    project_spd,
    recursive_predict_z,
    score_recursive_forecasts,
    set_reference_library_mode,
)


ROLLING_OUT = DEFAULT_OUT / "rolling_origin"
PRIMARY_METRIC_DEFAULT = "domain"


def _metric_key(metric: str, prefix: str) -> str:
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


def make_rolling_origin_splits(
    n: int,
    *,
    initial_fit_frac: float = 0.35,
    validation_frac: float = 0.15,
    test_block_frac: float = 0.12,
    min_fit: int = 60,
    min_validation: int = 24,
    min_test_block: int = 24,
    max_origins: int = 0,
) -> list[tuple[int, int, int]]:
    fit_end = max(min_fit, int(initial_fit_frac * n))
    validation_size = max(min_validation, int(validation_frac * n))
    block_size = max(min_test_block, int(test_block_frac * n))
    val_end = fit_end + validation_size
    splits: list[tuple[int, int, int]] = []
    while val_end + 2 < n:
        test_end = min(n, val_end + block_size)
        if test_end - val_end >= 2:
            splits.append((fit_end, val_end, test_end))
        if max_origins and len(splits) >= max_origins:
            break
        fit_end += block_size
        val_end += block_size
    return splits


def select_encoded_model(
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
) -> dict[str, object]:
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
    return {
        "Z": Z,
        "W": final_W,
        "ridge": float(best_lam),
        "val_w2_mean": float(best_val_metrics["w2_mean"]),
        "val_w2_median": float(best_val_metrics["w2_median"]),
        "val_log_score_mean": float(best_val_metrics["log_score_mean"]),
        "val_log_score_median": float(best_val_metrics["log_score_median"]),
        "val_domain_loss_mean": float(best_val_metrics["domain_loss_mean"]),
        "val_domain_loss_median": float(best_val_metrics["domain_loss_median"]),
        "decode": decode,
    }


def score_future_block(
    means: np.ndarray,
    covs: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    decode,
    *,
    block_start: int,
    block_end: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    start_t = max(0, block_start - horizon)
    stop_t = max(start_t, block_end - horizon)
    metrics = score_recursive_forecasts(
        means,
        covs,
        Z,
        W,
        decode,
        start_t=start_t,
        stop_t=stop_t,
        horizon=horizon,
        domain_profile=profile,
    )
    return {
        "test_w2_mean": float(metrics["w2_mean"]),
        "test_w2_median": float(metrics["w2_median"]),
        "test_w2_q90": float(metrics["w2_q90"]),
        "test_log_score_mean": float(metrics["log_score_mean"]),
        "test_log_score_median": float(metrics["log_score_median"]),
        "test_log_score_q90": float(metrics["log_score_q90"]),
        "test_domain_loss_mean": float(metrics["domain_loss_mean"]),
        "test_domain_loss_median": float(metrics["domain_loss_median"]),
        "test_domain_loss_q90": float(metrics["domain_loss_q90"]),
        "n_test_pairs": int(metrics["n_pairs"]),
        "min_pred_eig": float(metrics["min_pred_eig"]),
    }


def score_persistence_block(
    means: np.ndarray,
    covs: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    block_start: int,
    block_end: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    val = persistence_metrics(means, covs, fit_end=fit_end, val_end=val_end, horizon=horizon, domain_profile=profile)
    losses = []
    log_scores = []
    domain_losses = []
    for t in range(max(0, block_start - horizon), max(0, block_end - horizon)):
        if t + horizon < len(covs):
            losses.append(gaussian_w2_squared(means[t], covs[t], means[t + horizon], covs[t + horizon]))
            log_scores.append(
                gaussian_log_score_from_moments(means[t], covs[t], means[t + horizon], covs[t + horizon])
            )
            domain_losses.append(domain_loss_from_moments(means[t], covs[t], means[t + horizon], covs[t + horizon], profile))
    arr = np.asarray(losses, dtype=float)
    log_arr = np.asarray(log_scores, dtype=float)
    domain_arr = np.asarray(domain_losses, dtype=float)
    return {
        "ridge": np.nan,
        "val_w2_mean": float(val["val_w2_mean"]),
        "val_w2_median": float(val["val_w2_median"]),
        "val_log_score_mean": float(val["val_log_score_mean"]),
        "val_log_score_median": float(val["val_log_score_median"]),
        "val_domain_loss_mean": float(val["val_domain_loss_mean"]),
        "val_domain_loss_median": float(val["val_domain_loss_median"]),
        "test_w2_mean": float(arr.mean()) if len(arr) else np.nan,
        "test_w2_median": float(np.median(arr)) if len(arr) else np.nan,
        "test_w2_q90": float(np.quantile(arr, 0.9)) if len(arr) else np.nan,
        "test_log_score_mean": float(log_arr.mean()) if len(log_arr) else np.nan,
        "test_log_score_median": float(np.median(log_arr)) if len(log_arr) else np.nan,
        "test_log_score_q90": float(np.quantile(log_arr, 0.9)) if len(log_arr) else np.nan,
        "test_domain_loss_mean": float(domain_arr.mean()) if len(domain_arr) else np.nan,
        "test_domain_loss_median": float(np.median(domain_arr)) if len(domain_arr) else np.nan,
        "test_domain_loss_q90": float(np.quantile(domain_arr, 0.9)) if len(domain_arr) else np.nan,
        "n_test_pairs": int(len(arr)),
        "min_pred_eig": float(
            min(np.linalg.eigvalsh(C).min() for C in covs[max(0, block_start - horizon) : max(0, block_end - horizon)])
        )
        if block_end > block_start
        else np.nan,
    }


def _fit_raw_var(raw_series: np.ndarray, end_index: int, *, lam: float, model: str = "full") -> np.ndarray:
    end_index = int(min(max(end_index, 3), len(raw_series)))
    Z = np.asarray(raw_series[:end_index], dtype=float)
    X = Z[:-1]
    Y = Z[1:]
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
    Xa = np.column_stack([np.ones(len(X)), X])
    penalty = lam * np.eye(Xa.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ Y)


def reconstruct_raw_series_from_windows(raw_windows: np.ndarray, window_starts: np.ndarray) -> np.ndarray:
    raw_windows = np.asarray(raw_windows, dtype=float)
    window_starts = np.asarray(window_starts, dtype=int)
    if raw_windows.ndim != 3:
        raise ValueError("raw_windows must have shape (n_windows, window, dimension)")
    if len(raw_windows) != len(window_starts):
        raise ValueError("raw_windows and window_starts must have the same length")
    n, window_size, d = raw_windows.shape
    if n == 0:
        return np.empty((0, d))
    total_len = int(window_starts.max() + window_size)
    accum = np.zeros((total_len, d), dtype=float)
    counts = np.zeros(total_len, dtype=float)
    for W, start in zip(raw_windows, window_starts):
        start = int(start)
        stop = start + window_size
        accum[start:stop] += W
        counts[start:stop] += 1.0
    valid = counts > 0.0
    if not np.all(valid):
        last = np.zeros(d, dtype=float)
        for i in range(total_len):
            if valid[i]:
                last = accum[i] / counts[i]
            else:
                accum[i] = last
                counts[i] = 1.0
    return accum / counts[:, None]


def _predict_raw_target_window(
    raw_series: np.ndarray,
    window_starts: np.ndarray,
    *,
    source_index: int,
    horizon: int,
    window_size: int,
    W: np.ndarray,
) -> np.ndarray:
    target_index = source_index + horizon
    source_end = int(window_starts[source_index] + window_size)
    target_start = int(window_starts[target_index])
    target_end = int(target_start + window_size)
    pieces: list[np.ndarray] = []
    known_end = min(source_end, target_end)
    if target_start < known_end:
        pieces.append(np.asarray(raw_series[target_start:known_end], dtype=float))
    state = np.asarray(raw_series[source_end - 1], dtype=float)
    for pos in range(source_end, target_end):
        state = np.r_[1.0, state] @ W
        if pos >= target_start:
            pieces.append(state.reshape(1, -1))
    if not pieces:
        return np.empty((0, raw_series.shape[1]))
    return np.vstack(pieces)


def _window_moments(window: np.ndarray, *, ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(window, dtype=float)
    mean = W.mean(axis=0)
    cov = np.cov(W, rowvar=False)
    cov = np.atleast_2d(cov)
    return mean, project_spd(cov + ridge * np.eye(cov.shape[0]), eps=1e-8)


def _score_raw_var_with_weights(
    means: np.ndarray,
    covs: np.ndarray,
    raw_series: np.ndarray,
    window_starts: np.ndarray,
    *,
    window_size: int,
    W: np.ndarray,
    start_t: int,
    stop_t: int,
    horizon: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    losses = []
    log_scores = []
    domain_losses = []
    for t in range(start_t, stop_t):
        if t + horizon >= len(covs):
            continue
        pred_window = _predict_raw_target_window(
            raw_series,
            window_starts,
            source_index=t,
            horizon=horizon,
            window_size=window_size,
            W=W,
        )
        if len(pred_window) < 2:
            continue
        pred_mean, pred_cov = _window_moments(pred_window)
        losses.append(gaussian_w2_squared(pred_mean, pred_cov, means[t + horizon], covs[t + horizon]))
        log_scores.append(
            gaussian_log_score_from_moments(pred_mean, pred_cov, means[t + horizon], covs[t + horizon])
        )
        domain_losses.append(
            domain_loss_from_moments(pred_mean, pred_cov, means[t + horizon], covs[t + horizon], profile)
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


def score_raw_var_window_baseline(
    means: np.ndarray,
    covs: np.ndarray,
    raw_series: np.ndarray,
    window_starts: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    block_start: int,
    block_end: int,
    horizon: int,
    window_size: int,
    selection_metric: str = PRIMARY_METRIC_DEFAULT,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    raw_series = np.asarray(raw_series, dtype=float)
    window_starts = np.asarray(window_starts, dtype=int)
    fit_raw_end = int(window_starts[fit_end - 1] + window_size)
    val_raw_end = int(window_starts[val_end - 1] + window_size)
    val_start = max(0, fit_end - horizon)
    val_stop = max(val_start, val_end - horizon)
    best_lam = ridge_grid[0]
    best_val = np.inf
    best_val_metrics: dict[str, float | int] | None = None
    for lam in ridge_grid:
        W = _fit_raw_var(raw_series, fit_raw_end, lam=lam, model="full")
        val_metrics = _score_raw_var_with_weights(
            means,
            covs,
            raw_series,
            window_starts,
            window_size=window_size,
            W=W,
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
        raise ValueError("no finite validation score for raw VAR baseline")
    W_final = _fit_raw_var(raw_series, val_raw_end, lam=best_lam, model="full")
    test_metrics = _score_raw_var_with_weights(
        means,
        covs,
        raw_series,
        window_starts,
        window_size=window_size,
        W=W_final,
        start_t=max(0, block_start - horizon),
        stop_t=max(0, block_end - horizon),
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
        "min_pred_eig": np.nan,
    }


def _seasonal_source_index(target_index: int, forecast_origin: int, period_windows: int) -> int | None:
    """Return the latest observable same-phase window for a seasonal naive forecast."""
    if period_windows <= 0:
        return None
    source_index = target_index - period_windows
    while source_index > forecast_origin:
        source_index -= period_windows
    if source_index < 0:
        return None
    return source_index


def _score_seasonal_window_block(
    means: np.ndarray,
    covs: np.ndarray,
    raw_windows: np.ndarray,
    *,
    start_t: int,
    stop_t: int,
    horizon: int,
    seasonal_period_windows: int,
    domain_profile: dict[str, object],
) -> dict[str, float | int]:
    losses: list[float] = []
    log_scores: list[float] = []
    domain_losses: list[float] = []
    min_pred_eigs: list[float] = []
    for t in range(start_t, stop_t):
        target_index = t + horizon
        if target_index >= len(means):
            continue
        source_index = _seasonal_source_index(target_index, t, seasonal_period_windows)
        if source_index is None or source_index >= len(raw_windows):
            continue
        pred_mean, pred_cov = _window_moments(np.asarray(raw_windows[source_index], dtype=float))
        target_mean = means[target_index]
        target_cov = covs[target_index]
        losses.append(gaussian_w2_squared(pred_mean, pred_cov, target_mean, target_cov))
        log_scores.append(gaussian_log_score_from_moments(pred_mean, pred_cov, target_mean, target_cov))
        domain_losses.append(domain_loss_from_moments(pred_mean, pred_cov, target_mean, target_cov, domain_profile))
        min_pred_eigs.append(float(np.min(np.linalg.eigvalsh(project_spd(pred_cov, eps=1e-10)))))
    arr = np.asarray(losses, dtype=float)
    log_arr = np.asarray(log_scores, dtype=float)
    domain_arr = np.asarray(domain_losses, dtype=float)
    eig_arr = np.asarray(min_pred_eigs, dtype=float)
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
        "min_pred_eig": float(eig_arr.min()) if len(eig_arr) else np.nan,
    }


def score_seasonal_window_baseline(
    means: np.ndarray,
    covs: np.ndarray,
    raw_windows: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    block_start: int,
    block_end: int,
    horizon: int,
    seasonal_period_windows: int,
    domain_profile: dict[str, object] | None = None,
) -> dict[str, float | int]:
    profile = domain_metric_profile(None) if domain_profile is None else domain_profile
    raw_windows = np.asarray(raw_windows, dtype=float)
    val_metrics = _score_seasonal_window_block(
        means,
        covs,
        raw_windows,
        start_t=max(0, fit_end - horizon),
        stop_t=max(0, val_end - horizon),
        horizon=horizon,
        seasonal_period_windows=seasonal_period_windows,
        domain_profile=profile,
    )
    test_metrics = _score_seasonal_window_block(
        means,
        covs,
        raw_windows,
        start_t=max(0, block_start - horizon),
        stop_t=max(0, block_end - horizon),
        horizon=horizon,
        seasonal_period_windows=seasonal_period_windows,
        domain_profile=profile,
    )
    return {
        "ridge": np.nan,
        "val_w2_mean": float(val_metrics["w2_mean"]),
        "val_w2_median": float(val_metrics["w2_median"]),
        "val_log_score_mean": float(val_metrics["log_score_mean"]),
        "val_log_score_median": float(val_metrics["log_score_median"]),
        "val_domain_loss_mean": float(val_metrics["domain_loss_mean"]),
        "val_domain_loss_median": float(val_metrics["domain_loss_median"]),
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


def run_rolling_origin_series(
    *,
    job: str,
    dataset: str,
    means: np.ndarray,
    covs: np.ndarray,
    meta: dict,
    horizons: list[int],
    raw_series: np.ndarray | None = None,
    raw_windows: np.ndarray | None = None,
    window_starts: np.ndarray | None = None,
    window_size: int | None = None,
    ar_model: str = "diag",
    selection_metric: str = PRIMARY_METRIC_DEFAULT,
    domain_profile_override: dict[str, object] | None = None,
    max_origins: int = 3,
    min_n: int = 120,
    splits_override: list[tuple[int, int, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = np.asarray(means, dtype=float)
    covs = np.asarray([project_spd(C, eps=1e-8) for C in covs], dtype=float)
    if len(covs) < min_n:
        return pd.DataFrame(), pd.DataFrame()
    d = covs.shape[1]
    profile = dict(domain_metric_profile(dataset) if domain_profile_override is None else domain_profile_override)
    rows: list[dict[str, object]] = []
    ref_rows: list[dict[str, object]] = []
    splits = (
        make_rolling_origin_splits(len(covs), max_origins=max_origins)
        if splits_override is None
        else list(splits_override)
    )
    if any(not (2 <= fit_end < val_end < test_end <= len(covs)) for fit_end, val_end, test_end in splits):
        raise ValueError("rolling-origin splits must satisfy 2 <= fit_end < val_end < test_end <= n")
    if raw_series is None and raw_windows is not None and window_starts is not None:
        raw_series = reconstruct_raw_series_from_windows(raw_windows, window_starts)
        if window_size is None:
            window_size = int(np.asarray(raw_windows).shape[1])
    has_raw_baseline = raw_series is not None and window_starts is not None and window_size is not None
    seasonal_period_windows = int(meta.get("seasonal_period_windows", 0) or 0)
    has_seasonal_baseline = raw_windows is not None and seasonal_period_windows > 0

    def add(
        *,
        origin: int,
        fit_end: int,
        val_end: int,
        test_end: int,
        horizon: int,
        method: str,
        metrics: dict[str, object],
        selected_reference: str = "",
    ) -> None:
        rows.append(
            {
                "job": job,
                "dataset": dataset,
                "origin": int(origin),
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

    for origin, (fit_end, val_end, test_end) in enumerate(splits):
        refs = candidate_gaussian_references(means[:fit_end], covs[:fit_end])
        for horizon in horizons:
            if test_end <= val_end + horizon:
                continue
            add(
                origin=origin,
                fit_end=fit_end,
                val_end=val_end,
                test_end=test_end,
                horizon=horizon,
                method="persistence",
                metrics=score_persistence_block(
                    means,
                    covs,
                    fit_end=fit_end,
                    val_end=val_end,
                    block_start=val_end,
                    block_end=test_end,
                    horizon=horizon,
                    domain_profile=profile,
                ),
            )
            if has_raw_baseline:
                try:
                    add(
                        origin=origin,
                        fit_end=fit_end,
                        val_end=val_end,
                        test_end=test_end,
                        horizon=horizon,
                        method="raw_var_window_ar",
                        metrics=score_raw_var_window_baseline(
                            means,
                            covs,
                            np.asarray(raw_series, dtype=float),
                            np.asarray(window_starts, dtype=int),
                            fit_end=fit_end,
                            val_end=val_end,
                            block_start=val_end,
                            block_end=test_end,
                            horizon=horizon,
                            window_size=int(window_size),
                            selection_metric=selection_metric,
                            domain_profile=profile,
                        ),
                    )
                except Exception as exc:
                    add(
                        origin=origin,
                        fit_end=fit_end,
                        val_end=val_end,
                        test_end=test_end,
                        horizon=horizon,
                        method="raw_var_window_ar",
                        metrics={
                            "test_w2_mean": np.nan,
                            "test_log_score_mean": np.nan,
                            "test_domain_loss_mean": np.nan,
                            "error": repr(exc),
                        },
                    )
            if has_seasonal_baseline:
                try:
                    add(
                        origin=origin,
                        fit_end=fit_end,
                        val_end=val_end,
                        test_end=test_end,
                        horizon=horizon,
                        method="seasonal_window_naive",
                        metrics=score_seasonal_window_baseline(
                            means,
                            covs,
                            np.asarray(raw_windows, dtype=float),
                            fit_end=fit_end,
                            val_end=val_end,
                            block_start=val_end,
                            block_end=test_end,
                            horizon=horizon,
                            seasonal_period_windows=seasonal_period_windows,
                            domain_profile=profile,
                        ),
                    )
                except Exception as exc:
                    add(
                        origin=origin,
                        fit_end=fit_end,
                        val_end=val_end,
                        test_end=test_end,
                        horizon=horizon,
                        method="seasonal_window_naive",
                        metrics={
                            "test_w2_mean": np.nan,
                            "test_log_score_mean": np.nan,
                            "test_domain_loss_mean": np.nan,
                            "error": repr(exc),
                        },
                    )

            encoders = [
                ("euclidean_gaussian_ar", euclidean_encode, lambda z, dim=d: euclidean_decode(z, dim)),
                ("cholesky_gaussian_ar", cholesky_encode, lambda z, dim=d: cholesky_decode(z, dim)),
                ("log_euclidean_gaussian_ar", log_euclidean_encode, lambda z, dim=d: log_euclidean_decode(z, dim)),
            ]
            for method, encode, decode in encoders:
                selected = select_encoded_model(
                    means,
                    covs,
                    encode,
                    decode,
                    fit_end=fit_end,
                    val_end=val_end,
                    horizon=horizon,
                    ar_model=ar_model,
                    selection_metric=selection_metric,
                    domain_profile=profile,
                )
                metrics = {
                    "ridge": selected["ridge"],
                    "val_w2_mean": selected["val_w2_mean"],
                    "val_w2_median": selected["val_w2_median"],
                    "val_log_score_mean": selected["val_log_score_mean"],
                    "val_log_score_median": selected["val_log_score_median"],
                    "val_domain_loss_mean": selected["val_domain_loss_mean"],
                    "val_domain_loss_median": selected["val_domain_loss_median"],
                    **score_future_block(
                        means,
                        covs,
                        selected["Z"],
                        selected["W"],
                        selected["decode"],
                        block_start=val_end,
                        block_end=test_end,
                        horizon=horizon,
                        domain_profile=profile,
                    ),
                }
                add(
                    origin=origin,
                    fit_end=fit_end,
                    val_end=val_end,
                    test_end=test_end,
                    horizon=horizon,
                    method=method,
                    metrics=metrics,
                )

            candidate_metrics = []
            for ref_name, ref_mean, ref_cov in refs:
                try:
                    selected = select_encoded_model(
                        means,
                        covs,
                        lambda m, C, rm=ref_mean, rc=ref_cov: bwar_gaussian_encode(m, C, rm, rc),
                        lambda z, rm=ref_mean, rc=ref_cov: bwar_gaussian_decode(z, rm, rc),
                        fit_end=fit_end,
                        val_end=val_end,
                        horizon=horizon,
                        ar_model=ar_model,
                        selection_metric=selection_metric,
                        domain_profile=profile,
                    )
                    block_metrics = score_future_block(
                        means,
                        covs,
                        selected["Z"],
                        selected["W"],
                        selected["decode"],
                        block_start=val_end,
                        block_end=test_end,
                        horizon=horizon,
                        domain_profile=profile,
                    )
                    metrics = {
                        "ridge": selected["ridge"],
                        "val_w2_mean": selected["val_w2_mean"],
                        "val_w2_median": selected["val_w2_median"],
                        "val_log_score_mean": selected["val_log_score_mean"],
                        "val_log_score_median": selected["val_log_score_median"],
                        "val_domain_loss_mean": selected["val_domain_loss_mean"],
                        "val_domain_loss_median": selected["val_domain_loss_median"],
                        **block_metrics,
                    }
                    candidate_metrics.append((ref_name, metrics))
                    ref_rows.append(
                        {
                            "job": job,
                            "dataset": dataset,
                            "origin": int(origin),
                            "horizon": int(horizon),
                            "reference": ref_name,
                            "fit_end": int(fit_end),
                            "val_end": int(val_end),
                            "test_end": int(test_end),
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
                            "origin": int(origin),
                            "horizon": int(horizon),
                            "reference": ref_name,
                            "fit_end": int(fit_end),
                            "val_end": int(val_end),
                            "test_end": int(test_end),
                            "val_w2_mean": np.nan,
                            "test_w2_mean": np.nan,
                            "ridge": np.nan,
                            "val_domain_loss_mean": np.nan,
                            "test_domain_loss_mean": np.nan,
                            "domain_metric": str(profile["metric"]),
                            "domain_metric_label": str(profile["label"]),
                            "error": repr(exc),
                            **meta,
                        }
                    )
            if candidate_metrics:
                best_name, best_metrics = min(
                    candidate_metrics,
                    key=lambda item: float(item[1][metric_mean_key(selection_metric, "val")]),
                )
                add(
                    origin=origin,
                    fit_end=fit_end,
                    val_end=val_end,
                    test_end=test_end,
                    horizon=horizon,
                    method="bwar_selected_ref",
                    metrics=best_metrics,
                    selected_reference=best_name,
                )

    return pd.DataFrame(rows), pd.DataFrame(ref_rows)


def summarize_rolling(raw: pd.DataFrame, *, loss_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_col = choose_primary_loss_column(raw, loss_col)
    ok = raw.dropna(subset=[primary_col]).copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected = ok[ok["method"] == "bwar_selected_ref"].copy()
    non_bwar = ok[~ok["method"].str.startswith("bwar")].copy()
    best_non = non_bwar.loc[non_bwar.groupby(["job", "origin", "horizon"])[primary_col].idxmin()].copy()
    comp = selected.merge(
        best_non[["job", "origin", "horizon", "method", primary_col]].rename(
            columns={"method": "best_non_bwar_method", primary_col: "best_non_bwar_loss"}
        ),
        on=["job", "origin", "horizon"],
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
    return comp, summary, summary_h


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
    comp, summary, summary_h = summarize_rolling(raw)
    comp.to_csv(out_dir / f"comparison_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"summary_by_dataset_{tag}.csv", index=False)
    summary_h.to_csv(out_dir / f"summary_by_dataset_horizon_{tag}.csv", index=False)
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(
            {
                "summary_by_dataset": summary.to_dict(orient="records"),
                "summary_by_dataset_horizon": summary_h.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin BWAR backtest for sequential deployment checks.")
    parser.add_argument(
        "--datasets",
        choices=[
            "uci_har",
            "appliances",
            "sml2010",
            "solar_energy_hd",
            "electricity_hd",
            "beijing_air_hd",
            "intel_lab",
            "hai21",
            "melbourne_pedestrian",
            "room_occupancy",
            "household_power",
            "household_power_strong",
            "finance_etf_hd",
            "wpp_population_age",
            "wpp_population_recent_age",
            "hapt",
            "mhealth_full",
        ],
        default="household_power_strong",
    )
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--ar-model", choices=["diag", "full"], default="diag")
    parser.add_argument("--reference-library", choices=["full", "no_barycenter", "fast"], default="full")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--skip-jobs", type=int, default=0)
    parser.add_argument("--max-origins", type=int, default=3)
    parser.add_argument("--no-raw-baseline", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROLLING_OUT)
    args = parser.parse_args()
    set_reference_library_mode(args.reference_library)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    tag = (
        f"{args.datasets}_h{'-'.join(map(str, horizons))}_{args.ar_model}"
        f"_orig{args.max_origins}"
        + (f"_skip{args.skip_jobs}" if args.skip_jobs else "")
        + (f"_max{args.max_jobs}" if args.max_jobs else "")
        + ("_quick" if args.quick else "")
    )
    if args.reference_library != "full":
        tag += f"_{args.reference_library}"
    started = time.time()
    rows: list[pd.DataFrame] = []
    ref_tables: list[pd.DataFrame] = []
    n_jobs = 0
    use_raw_baseline = not bool(args.no_raw_baseline)
    for stream_index, payload in enumerate(
        iter_dataset_jobs(args.datasets, quick=args.quick, return_windows=use_raw_baseline)
    ):
        raw_info = None
        if len(payload) == 6:
            job, dataset, means, covs, meta, raw_info = payload
        else:
            job, dataset, means, covs, meta = payload
        if stream_index < args.skip_jobs:
            continue
        if args.max_jobs and n_jobs >= args.max_jobs:
            break
        print(
            f"Rolling-origin {job} horizons={horizons} n={len(covs)} "
            f"d={covs.shape[1] if len(covs) else 'NA'}...",
            flush=True,
        )
        raw, refs = run_rolling_origin_series(
            job=job,
            dataset=dataset,
            means=means,
            covs=covs,
            meta=meta,
            horizons=horizons,
            raw_windows=raw_info["raw_windows"] if raw_info else None,
            window_starts=raw_info["window_starts"] if raw_info else None,
            window_size=raw_info["window_size"] if raw_info else None,
            ar_model=args.ar_model,
            max_origins=args.max_origins,
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
        "max_origins": int(args.max_origins),
        "raw_baseline": bool(use_raw_baseline),
        "skip_jobs": int(args.skip_jobs),
        "n_jobs_attempted": int(n_jobs),
        "elapsed_seconds": round(time.time() - started, 3),
        "out_dir": str(args.out_dir),
        "tag": tag,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"run_metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path = args.out_dir / f"summary_by_dataset_{tag}.csv"
    if summary_path.exists():
        print(pd.read_csv(summary_path).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
