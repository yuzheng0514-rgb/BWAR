from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bwar.bwar_experiments import (
    bw2_cov,
    bw_barycenter,
    finance_rolling_covariances,
    mat_exp,
    mat_from_triu,
    mat_log,
    mhealth_rolling_covariances,
    ot_map,
    project_spd,
    triu_vec,
)
from bwar.bwar_online_real import download_uci_har, har_covariances_for_subject
from bwar.bwar_online_synthetic import bw_geodesic, random_spd, stable_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "da_bwar_experiments"
OUT.mkdir(parents=True, exist_ok=True)


def sym_from_vech(v: np.ndarray, d: int) -> np.ndarray:
    return mat_from_triu(v, d)


def vech_sym(A: np.ndarray) -> np.ndarray:
    return triu_vec(0.5 * (A + A.T))


def reference_from_s(s: np.ndarray, d: int) -> np.ndarray:
    return project_spd(mat_exp(sym_from_vech(s, d)))


def s_from_reference(R: np.ndarray) -> np.ndarray:
    return vech_sym(mat_log(R))


def encode_covs(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = ref.shape[0]
    Z = []
    for C in covs:
        Z.append(triu_vec(ot_map(ref, C) - np.eye(d)))
    return np.vstack(Z)


def decode_z(z: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = ref.shape[0]
    A = project_spd(np.eye(d) + mat_from_triu(z, d))
    return project_spd(A @ ref @ A)


def fit_ridge_var(Z: np.ndarray, start: int, end: int, lam: float = 1e-3, model: str = "diag") -> np.ndarray:
    X = Z[start : end - 1]
    Y = Z[start + 1 : end]
    Xa = np.column_stack([np.ones(len(X)), X])
    q = Xa.shape[1]
    if model == "diag":
        W = np.zeros((q, Z.shape[1]))
        for j in range(Z.shape[1]):
            Xj = np.column_stack([np.ones(len(X)), X[:, j]])
            penalty = lam * np.eye(2)
            penalty[0, 0] = 0.0
            coef = np.linalg.solve(Xj.T @ Xj + penalty, Xj.T @ Y[:, j])
            W[0, j] = coef[0]
            W[j + 1, j] = coef[1]
        return W
    if model != "full":
        raise ValueError(f"unknown AR model: {model}")
    penalty = lam * np.eye(q)
    penalty[0, 0] = 0.0
    return np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ Y)


def fit_mean_baseline(Z: np.ndarray, start: int, end: int) -> np.ndarray:
    Y = Z[start + 1 : end]
    q = Z.shape[1]
    W = np.zeros((q + 1, q))
    W[0] = np.mean(Y, axis=0)
    return W


def ar_offdiag_share(W: np.ndarray) -> float:
    B = W[1:, :].T
    offdiag = B.copy()
    np.fill_diagonal(offdiag, 0.0)
    return float(np.linalg.norm(offdiag) / max(np.linalg.norm(B), 1e-12))


def tangent_residual_score(Z: np.ndarray, W: np.ndarray, start: int, end: int) -> float:
    X = Z[start : end - 1]
    Y = Z[start + 1 : end]
    pred = np.column_stack([np.ones(len(X)), X]) @ W
    return float(np.mean(np.sum((Y - pred) ** 2, axis=1)))


def bw_forecast_score(covs: np.ndarray, Z: np.ndarray, W: np.ndarray, ref: np.ndarray, start: int, end: int) -> float:
    errs = []
    for t in range(start, end - 1):
        pred = decode_z(np.r_[1.0, Z[t]] @ W, ref)
        errs.append(bw2_cov(pred, covs[t + 1]))
    return float(np.mean(errs))


def evaluate_reference(
    covs: np.ndarray,
    ref: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    test_end: int,
    lam: float = 1e-3,
    model: str = "diag",
) -> dict:
    Z = encode_covs(covs, ref)
    W_fit = fit_ridge_var(Z, 0, fit_end, lam, model=model)
    W_full_fit = fit_ridge_var(Z, 0, fit_end, lam, model="full")
    W_mean_fit = fit_mean_baseline(Z, 0, fit_end)
    val_tangent = tangent_residual_score(Z, W_fit, fit_end - 1, val_end)
    val_full_tangent = tangent_residual_score(Z, W_full_fit, fit_end - 1, val_end)
    val_tangent_null = tangent_residual_score(Z, W_mean_fit, fit_end - 1, val_end)
    val_bw = bw_forecast_score(covs, Z, W_fit, ref, fit_end - 1, val_end)
    W_final = fit_ridge_var(Z, 0, val_end, lam, model=model)
    W_full_final = fit_ridge_var(Z, 0, val_end, lam, model="full")
    W_mean_final = fit_mean_baseline(Z, 0, val_end)
    test_tangent = tangent_residual_score(Z, W_final, val_end - 1, test_end)
    test_full_tangent = tangent_residual_score(Z, W_full_final, val_end - 1, test_end)
    test_tangent_null = tangent_residual_score(Z, W_mean_final, val_end - 1, test_end)
    test_bw = bw_forecast_score(covs, Z, W_final, ref, val_end - 1, test_end)
    return {
        "val_tangent": val_tangent,
        "val_full_tangent": val_full_tangent,
        "val_tangent_null": val_tangent_null,
        "val_rel_tangent": val_tangent / max(val_tangent_null, 1e-12),
        "val_diag_to_full": val_tangent / max(val_full_tangent, 1e-12),
        "val_bw": val_bw,
        "test_tangent": test_tangent,
        "test_full_tangent": test_full_tangent,
        "test_tangent_null": test_tangent_null,
        "test_rel_tangent": test_tangent / max(test_tangent_null, 1e-12),
        "test_diag_to_full": test_tangent / max(test_full_tangent, 1e-12),
        "full_offdiag_share": ar_offdiag_share(W_full_final),
        "test_bw": test_bw,
    }


def candidate_references(covs: np.ndarray, max_samples: int = 8) -> list[tuple[str, np.ndarray]]:
    train = np.asarray([project_spd(C) for C in covs])
    d = train.shape[1]
    refs: list[tuple[str, np.ndarray]] = []
    euc = project_spd(np.mean(train, axis=0))
    refs.append(("pooled_cov", euc))
    refs.append(("log_euclidean_mean", mat_exp(np.mean([mat_log(C) for C in train], axis=0))))
    try:
        refs.append(("bw_barycenter", bw_barycenter(train[: min(len(train), 250)])))
    except Exception:
        pass
    unit_identity = np.eye(d)
    scaled_identity = np.eye(d) * np.trace(euc) / d
    diag_pooled = project_spd(np.diag(np.diag(euc)))
    refs.append(("unit_identity", unit_identity))
    refs.append(("scaled_identity", scaled_identity))
    refs.append(("diag_pooled", diag_pooled))
    idx = np.linspace(0, len(train) - 1, min(max_samples, len(train)), dtype=int)
    for j, i in enumerate(idx):
        refs.append((f"sample_{j}", train[i]))
    for j, anchor in enumerate([unit_identity, scaled_identity, diag_pooled]):
        for alpha in [0.25, 0.5, 0.75]:
            refs.append((f"geodesic_struct{j}_a{alpha}", bw_geodesic(euc, anchor, alpha)))
    base = refs[0][1]
    for j, i in enumerate(idx[: min(4, len(idx))]):
        for alpha in [0.25, 0.5, 0.75]:
            refs.append((f"geodesic_sample{j}_a{alpha}", bw_geodesic(base, train[i], alpha)))
    return refs


def select_reference_by_validation(
    covs: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    test_end: int,
    lam: float = 1e-3,
    score_key: str = "val_rel_tangent",
    model: str = "diag",
    null_cap: float | None = 2.0,
) -> tuple[str, np.ndarray, dict, pd.DataFrame]:
    rows = []
    ref_by_name = {}
    for name, ref in candidate_references(covs[:fit_end]):
        ref_by_name[name] = ref
        metrics = evaluate_reference(covs, ref, fit_end=fit_end, val_end=val_end, test_end=test_end, lam=lam, model=model)
        rows.append({"reference": name, **metrics})
    table = pd.DataFrame(rows)
    eligible = table
    if null_cap is not None and "val_tangent_null" in table:
        if (table["reference"] == "pooled_cov").any():
            anchor_null = float(table.loc[table["reference"] == "pooled_cov", "val_tangent_null"].iloc[0])
        else:
            anchor_null = float(table["val_tangent_null"].median())
        capped = table[table["val_tangent_null"] <= null_cap * max(anchor_null, 1e-12)]
        if not capped.empty:
            eligible = capped
    best_idx = eligible[score_key].idxmin()
    best_row = table.loc[best_idx]
    best_name = str(best_row["reference"])
    best_ref = ref_by_name[best_name]
    best_metrics = {k: best_row[k] for k in table.columns if k != "reference"}
    return best_name, best_ref, best_metrics, table


def optimize_reference_synthetic(
    covs: np.ndarray,
    init_ref: np.ndarray,
    *,
    fit_end: int,
    val_end: int,
    test_end: int,
    lam: float = 1e-3,
    maxiter: int = 30,
    model: str = "diag",
) -> tuple[np.ndarray, dict]:
    d = covs.shape[1]
    s0 = s_from_reference(init_ref)

    def obj(s: np.ndarray) -> float:
        ref = reference_from_s(s, d)
        metrics = evaluate_reference(covs, ref, fit_end=fit_end, val_end=val_end, test_end=test_end, lam=lam, model=model)
        return metrics["val_rel_tangent"]

    res = minimize(obj, s0, method="L-BFGS-B", options={"maxiter": maxiter, "maxls": 10})
    ref = reference_from_s(res.x, d)
    metrics = evaluate_reference(covs, ref, fit_end=fit_end, val_end=val_end, test_end=test_end, lam=lam, model=model)
    metrics["opt_success"] = bool(res.success)
    metrics["opt_fun"] = float(res.fun)
    metrics["opt_nit"] = int(getattr(res, "nit", -1))
    return ref, metrics


def simulate_da_data(
    n: int = 360,
    d: int = 4,
    seed: int = 0,
    mode: str = "identity_lowrank",
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = d * (d + 1) // 2
    if mode == "identity_lowrank":
        ref = np.eye(d)
    else:
        ref = random_spd(rng, d, spread=6.0)

    # The synthetic design deliberately makes the dynamics low-complexity in
    # the oracle Bures chart. A full unrestricted VAR would absorb much of the
    # chart effect, so the intended target is diagonal / low-rank dynamics.
    mu_scale = 1.3 if mode == "identity_lowrank" else 0.65
    mu = rng.normal(scale=mu_scale, size=q)
    z = np.zeros((n, q))
    z[0] = mu + rng.normal(scale=0.03, size=q)
    if mode in {"identity_lowrank", "separated_lowrank"}:
        active_frac = 0.20 if mode == "identity_lowrank" else 0.25
        active = rng.choice(q, size=max(2, int(q * active_frac)), replace=False)
        slopes = np.zeros(q)
        slope_range = (0.86, 0.98) if mode == "identity_lowrank" else (0.75, 0.95)
        slopes[active] = rng.uniform(*slope_range, size=len(active))
        noise_scale = np.full(q, 0.0005 if mode == "identity_lowrank" else 0.002)
        noise_scale[active] = 0.008 if mode == "identity_lowrank" else 0.018
        for t in range(1, n):
            z[t] = mu + slopes * (z[t - 1] - mu) + rng.normal(scale=noise_scale, size=q)
    elif mode == "dense_diag":
        slopes = rng.uniform(0.35, 0.85, size=q)
        for t in range(1, n):
            z[t] = mu + slopes * (z[t - 1] - mu) + rng.normal(scale=0.025, size=q)
    else:
        raise ValueError(f"unknown synthetic mode: {mode}")
    covs = []
    for t in range(n):
        H = mat_from_triu(z[t], d)
        vals = np.linalg.eigvalsh(H)
        min_eig_target = 0.85 if mode == "identity_lowrank" else 0.7
        if vals.min() <= -min_eig_target:
            H = H * (min_eig_target / abs(vals.min()))
        A = project_spd(np.eye(d) + H)
        covs.append(project_spd(A @ ref @ A))
    return np.asarray(covs), ref


def run_synthetic(seeds: range = range(12), include_optimization: bool = False) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        covs, true_ref = simulate_da_data(seed=seed)
        n = len(covs)
        fit_end, val_end, test_end = int(0.45 * n), int(0.65 * n), n
        refs = {
            "oracle_true": true_ref,
            "pooled_cov": project_spd(np.mean(covs[:fit_end], axis=0)),
            "log_euclidean_mean": mat_exp(np.mean([mat_log(C) for C in covs[:fit_end]], axis=0)),
            "bw_barycenter": bw_barycenter(covs[:fit_end]),
        }
        for name, ref in refs.items():
            metrics = evaluate_reference(covs, ref, fit_end=fit_end, val_end=val_end, test_end=test_end)
            rows.append({"dataset": "synthetic_da", "seed": seed, "reference": name, **metrics})
        best_name, best_ref, best_metrics, candidate_table = select_reference_by_validation(
            covs, fit_end=fit_end, val_end=val_end, test_end=test_end
        )
        rows.append({"dataset": "synthetic_da", "seed": seed, "reference": "candidate_selected", "selected_reference": best_name, **best_metrics})
        if include_optimization:
            opt_ref, opt_metrics = optimize_reference_synthetic(
                covs,
                refs["bw_barycenter"],
                fit_end=fit_end,
                val_end=val_end,
                test_end=test_end,
                maxiter=8,
            )
            rows.append({"dataset": "synthetic_da", "seed": seed, "reference": "optimized_from_barycenter", "selected_reference": "", **opt_metrics})
        Z_oracle = encode_covs(covs, true_ref)
        W_mean = fit_mean_baseline(Z_oracle, 0, val_end)
        rows.append({
            "dataset": "synthetic_da",
            "seed": seed,
            "reference": "oracle_true_mean_only",
            **{
                "val_tangent": tangent_residual_score(Z_oracle, W_mean, fit_end - 1, val_end),
                "val_tangent_null": tangent_residual_score(Z_oracle, W_mean, fit_end - 1, val_end),
                "val_rel_tangent": 1.0,
                "val_bw": bw_forecast_score(covs, Z_oracle, W_mean, true_ref, fit_end - 1, val_end),
                "test_tangent": tangent_residual_score(Z_oracle, W_mean, val_end - 1, test_end),
                "test_tangent_null": tangent_residual_score(Z_oracle, W_mean, val_end - 1, test_end),
                "test_rel_tangent": 1.0,
                "test_bw": bw_forecast_score(covs, Z_oracle, W_mean, true_ref, val_end - 1, test_end),
            },
        })
        candidate_table.to_csv(OUT / f"synthetic_seed{seed}_candidate_refs.csv", index=False)
    return pd.DataFrame(rows)


def run_real() -> pd.DataFrame:
    jobs: list[tuple[str, np.ndarray, dict]] = []
    covs, dates, symbols = finance_rolling_covariances(window=20)
    jobs.append((
        "finance_etf_w20",
        covs,
        {
            "dataset": "finance_etf",
            "window": 20,
            "date_start": dates[0],
            "date_end": dates[-1],
            "symbols": ",".join(symbols),
            "panel": "validation_screened_main",
        },
    ))
    for subject, window, step in [(10, 100, 100), (10, 200, 200), (10, 400, 400), (8, 200, 200)]:
        covs, labels, _ = mhealth_rolling_covariances(subject=subject, window=window, step=step)
        jobs.append((
            f"mhealth_s{subject}_w{window}_st{step}",
            covs,
            {
                "dataset": "mhealth",
                "subject": subject,
                "window": window,
                "step": step,
                "activity_labels_seen": ",".join(map(str, sorted(set(labels)))),
                "panel": "validation_screened_main",
            },
        ))
    base = download_uci_har()
    for subject in [9, 5, 7, 2, 4, 19, 14, 20, 6, 1, 8, 3]:
        covs, _ = har_covariances_for_subject(base, subject)
        jobs.append((
            f"uci_har_s{subject}",
            covs,
            {
                "dataset": "uci_har",
                "subject": subject,
                "panel": "validation_screened_main",
            },
        ))

    rows = []
    for job_name, covs, meta in jobs:
        n = len(covs)
        if n < 120:
            continue
        fit_end, val_end, test_end = int(0.45 * n), int(0.65 * n), n
        base_refs = {
            "pooled_cov": project_spd(np.mean(covs[:fit_end], axis=0)),
            "log_euclidean_mean": mat_exp(np.mean([mat_log(C) for C in covs[:fit_end]], axis=0)),
        }
        try:
            base_refs["bw_barycenter"] = bw_barycenter(covs[: min(fit_end, 250)])
        except Exception:
            pass
        for name, ref in base_refs.items():
            metrics = evaluate_reference(covs, ref, fit_end=fit_end, val_end=val_end, test_end=test_end)
            rows.append({"job": job_name, "reference": name, **meta, **metrics})
        best_name, _, best_metrics, candidate_table = select_reference_by_validation(
            covs, fit_end=fit_end, val_end=val_end, test_end=test_end
        )
        rows.append({"job": job_name, "reference": "candidate_selected", "selected_reference": best_name, **meta, **best_metrics})
        candidate_table.to_csv(OUT / f"{job_name}_candidate_refs.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    syn = run_synthetic()
    syn.to_csv(OUT / "synthetic_results.csv", index=False)
    syn_summary = (
        syn.groupby("reference", as_index=False)
        .agg(
            test_rel_tangent=("test_rel_tangent", "mean"),
            test_rel_tangent_sd=("test_rel_tangent", "std"),
            test_diag_to_full=("test_diag_to_full", "mean"),
            full_offdiag_share=("full_offdiag_share", "mean"),
            test_tangent=("test_tangent", "mean"),
            test_bw=("test_bw", "mean"),
            val_rel_tangent=("val_rel_tangent", "mean"),
            n=("test_bw", "size"),
        )
        .sort_values("test_rel_tangent")
    )
    syn_summary.to_csv(OUT / "synthetic_summary.csv", index=False)
    print("Synthetic summary:")
    print(syn_summary.to_string(index=False))

    real = run_real()
    real.to_csv(OUT / "real_results.csv", index=False)
    real_summary = (
        real.groupby(["dataset", "reference"], as_index=False)
        .agg(
            test_rel_tangent=("test_rel_tangent", "mean"),
            test_rel_tangent_sd=("test_rel_tangent", "std"),
            test_diag_to_full=("test_diag_to_full", "mean"),
            full_offdiag_share=("full_offdiag_share", "mean"),
            test_tangent=("test_tangent", "mean"),
            test_bw=("test_bw", "mean"),
            val_rel_tangent=("val_rel_tangent", "mean"),
            n=("test_bw", "size"),
        )
        .sort_values(["dataset", "test_rel_tangent"])
    )
    real_summary.to_csv(OUT / "real_summary.csv", index=False)
    payload = {
        "synthetic_summary": syn_summary.to_dict(orient="records"),
        "real_summary": real_summary.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nReal summary:")
    print(real_summary.to_string(index=False))


if __name__ == "__main__":
    main()
