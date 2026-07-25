from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bwar.bwar_experiments import (
    bw2_cov,
    bw_barycenter,
    mat_exp,
    mat_from_triu,
    mat_log,
    ot_map,
    project_spd,
    triu_vec,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bwar_online_synthetic"
OUT.mkdir(parents=True, exist_ok=True)


def random_spd(rng: np.random.Generator, d: int, spread: float = 1.6) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    vals = np.exp(np.linspace(-spread, spread, d))
    rng.shuffle(vals)
    return project_spd((Q * vals) @ Q.T)


def stable_matrix(rng: np.random.Generator, q: int, rho: float = 0.68) -> np.ndarray:
    B = rng.normal(size=(q, q))
    radius = max(abs(np.linalg.eigvals(B)))
    return rho * B / max(radius, 1e-8)


def bw_geodesic(S0: np.ndarray, S1: np.ndarray, alpha: float) -> np.ndarray:
    A = ot_map(S0, S1)
    M = (1.0 - alpha) * np.eye(S0.shape[0]) + alpha * A
    return project_spd(M @ S0 @ M)


def bw_feature(ref: np.ndarray, C: np.ndarray) -> np.ndarray:
    return triu_vec(ot_map(ref, C) - np.eye(ref.shape[0]))


def bw_exp_feature(ref: np.ndarray, z: np.ndarray) -> np.ndarray:
    d = ref.shape[0]
    A = project_spd(np.eye(d) + mat_from_triu(z, d))
    return project_spd(A @ ref @ A)


def simulate_sequence(
    scenario: str,
    *,
    n: int = 420,
    d: int = 5,
    seed: int = 0,
    latent_noise: float = 0.045,
    tangent_scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed covariance sequence and the latent local reference path."""
    rng = np.random.default_rng(seed)
    q = d * (d + 1) // 2
    S0 = random_spd(rng, d)
    S1 = random_spd(rng, d)
    B = stable_matrix(rng, q)
    z = np.zeros((n, q))
    for t in range(1, n):
        z[t] = B @ z[t - 1] + latent_noise * rng.normal(size=q)

    refs = []
    covs = []
    for t in range(n):
        if scenario == "stationary":
            alpha = 0.0
        elif scenario == "slow_drift":
            alpha = 0.85 * t / max(n - 1, 1)
        elif scenario == "cyclic_drift":
            alpha = 0.5 + 0.35 * np.sin(2.0 * np.pi * t / max(n - 1, 1))
        elif scenario == "abrupt_shift":
            alpha = 0.0 if t < int(0.58 * n) else 0.9
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        ref = bw_geodesic(S0, S1, alpha)
        H = mat_from_triu(tangent_scale * z[t], d)
        A = mat_exp(0.5 * H)
        covs.append(project_spd(A @ ref @ A))
        refs.append(ref)
    return np.asarray(covs), np.asarray(refs)


@dataclass
class OnlineLinear:
    dim: int
    forget: float = 0.995
    uncertainty: float = 100.0

    def __post_init__(self) -> None:
        self.P = self.uncertainty * np.eye(self.dim + 1)
        self.W = np.zeros((self.dim + 1, self.dim))

    def predict(self, x: np.ndarray) -> np.ndarray:
        xa = np.r_[1.0, x]
        return xa @ self.W

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        xa = np.r_[1.0, x]
        denom = self.forget + xa @ self.P @ xa
        K = (self.P @ xa) / max(denom, 1e-12)
        err = y - xa @ self.W
        self.W += np.outer(K, err)
        self.P = (self.P - np.outer(K, xa @ self.P)) / self.forget


def online_log_euclidean(covs: np.ndarray, warmup: int) -> dict:
    n, d, _ = covs.shape
    q = d * (d + 1) // 2
    model = OnlineLinear(q)
    errs = []
    for t in range(n - 1):
        x = triu_vec(mat_log(covs[t]))
        pred_z = model.predict(x)
        pred = mat_exp(mat_from_triu(pred_z, d))
        if t >= warmup:
            errs.append(bw2_cov(pred, covs[t + 1]))
        y = triu_vec(mat_log(covs[t + 1]))
        model.update(x, y)
    return summarize_errors(errs)


def online_euclidean(covs: np.ndarray, warmup: int) -> dict:
    n, d, _ = covs.shape
    q = d * (d + 1) // 2
    model = OnlineLinear(q)
    errs = []
    for t in range(n - 1):
        x = triu_vec(covs[t])
        pred_z = model.predict(x)
        pred = project_spd(mat_from_triu(pred_z, d))
        if t >= warmup:
            errs.append(bw2_cov(pred, covs[t + 1]))
        y = triu_vec(covs[t + 1])
        model.update(x, y)
    return summarize_errors(errs)


def online_bwar(
    covs: np.ndarray,
    warmup: int,
    *,
    mode: str,
    true_refs: np.ndarray | None = None,
) -> dict:
    n, d, _ = covs.shape
    q = d * (d + 1) // 2
    model = OnlineLinear(q)
    if mode == "fixed_warmup":
        ref = bw_barycenter(covs[:warmup])
    else:
        ref = covs[0]
    errs = []
    etas = []
    recent_errs: list[float] = []
    for t in range(n - 1):
        if mode == "oracle_local":
            if true_refs is None:
                raise ValueError("oracle_local requires true_refs")
            ref = true_refs[t]
        x = bw_feature(ref, covs[t])
        pred_z = model.predict(x)
        pred = bw_exp_feature(ref, pred_z)
        err = bw2_cov(pred, covs[t + 1])
        if t >= warmup:
            errs.append(err)
        y = bw_feature(ref, covs[t + 1])
        model.update(x, y)

        eta = 0.0
        if mode == "online_mean":
            eta = min(0.2, 1.0 / (t + 2.0))
        elif mode == "forgetting":
            eta = 0.025
        elif mode == "residual_eta":
            ref_dist = bw2_cov(ref, covs[t + 1])
            eta = 0.003 + 0.10 * err / (ref_dist + 1e-8)
            eta = float(np.clip(eta, 0.003, 0.09))
        elif mode == "residual_reset":
            if len(recent_errs) >= 25:
                med = float(np.median(recent_errs[-50:]))
                mad = float(np.median(np.abs(np.asarray(recent_errs[-50:]) - med))) + 1e-10
                shock = err > med + 5.0 * 1.4826 * mad
            else:
                shock = False
            eta = 0.25 if shock else 0.015
        if mode in {"online_mean", "forgetting", "residual_eta", "residual_reset"}:
            ref = bw_geodesic(ref, covs[t + 1], eta)
            etas.append(eta)
        if t >= warmup:
            recent_errs.append(err)
    out = summarize_errors(errs)
    if etas:
        out["eta_mean"] = float(np.mean(etas))
        out["eta_median"] = float(np.median(etas))
    return out


def online_persistence(covs: np.ndarray, warmup: int) -> dict:
    errs = [bw2_cov(covs[t], covs[t + 1]) for t in range(warmup, len(covs) - 1)]
    return summarize_errors(errs)


def summarize_errors(errs: list[float]) -> dict:
    arr = np.asarray(errs, dtype=float)
    return {
        "bw2_mean": float(arr.mean()),
        "bw2_median": float(np.median(arr)),
        "bw2_q90": float(np.quantile(arr, 0.9)),
        "n_eval": int(arr.size),
    }


def run_one(scenario: str, seed: int, warmup: int = 80) -> list[dict]:
    covs, refs = simulate_sequence(scenario, seed=seed)
    methods = {
        "persistence": online_persistence(covs, warmup),
        "online_euclidean": online_euclidean(covs, warmup),
        "online_log_euclidean": online_log_euclidean(covs, warmup),
        "bwar_fixed_warmup_ref": online_bwar(covs, warmup, mode="fixed_warmup"),
        "bwar_online_mean_ref": online_bwar(covs, warmup, mode="online_mean"),
        "bwar_forgetting_ref": online_bwar(covs, warmup, mode="forgetting"),
        "bwar_residual_eta_ref": online_bwar(covs, warmup, mode="residual_eta"),
        "bwar_residual_reset_ref": online_bwar(covs, warmup, mode="residual_reset"),
        "bwar_oracle_local_ref": online_bwar(covs, warmup, mode="oracle_local", true_refs=refs),
    }
    rows = []
    for method, metrics in methods.items():
        rows.append({"scenario": scenario, "seed": seed, "method": method, **metrics})
    return rows


def main() -> None:
    scenarios = ["stationary", "slow_drift", "cyclic_drift", "abrupt_shift"]
    seeds = list(range(8))
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            rows.extend(run_one(scenario, seed))
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_results.csv", index=False)

    summary = (
        raw.groupby(["scenario", "method"], as_index=False)
        .agg(
            bw2_mean=("bw2_mean", "mean"),
            bw2_mean_sd=("bw2_mean", "std"),
            bw2_median=("bw2_median", "mean"),
            bw2_q90=("bw2_q90", "mean"),
        )
        .sort_values(["scenario", "bw2_mean"])
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    best = summary.loc[summary.groupby("scenario")["bw2_mean"].idxmin()].copy()
    best.to_csv(OUT / "best_by_scenario.csv", index=False)

    payload = {
        "scenarios": scenarios,
        "seeds": seeds,
        "best_by_scenario": best.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nBest by scenario:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
