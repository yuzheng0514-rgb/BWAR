from __future__ import annotations

import csv
import io
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bwar_experiments"
OUT.mkdir(parents=True, exist_ok=True)


def sym(A):
    return 0.5 * (A + A.T)


def project_spd(A, eps=1e-7):
    A = sym(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def mat_power(A, power, eps=1e-9):
    A = project_spd(A, eps)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None) ** power
    return (vecs * vals) @ vecs.T


def mat_log(A):
    A = project_spd(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.log(np.clip(vals, 1e-12, None))
    return sym((vecs * vals) @ vecs.T)


def mat_exp(A):
    A = sym(A)
    vals, vecs = np.linalg.eigh(A)
    vals = np.exp(np.clip(vals, -40, 40))
    return sym((vecs * vals) @ vecs.T)


def bw2_cov(A, B):
    A = project_spd(A)
    B = project_spd(B)
    As = mat_power(A, 0.5)
    inner = As @ B @ As
    val = float(np.trace(A + B - 2.0 * mat_power(inner, 0.5)))
    return max(val, 0.0)


def bw_barycenter(covs, max_iter=35, tol=1e-9):
    S = project_spd(np.mean(covs, axis=0))
    for _ in range(max_iter):
        S_s = mat_power(S, 0.5)
        S_is = mat_power(S, -0.5)
        M = np.zeros_like(S)
        for C in covs:
            M += mat_power(S_s @ C @ S_s, 0.5)
        M /= len(covs)
        S_new = project_spd(S_is @ M @ M @ S_is)
        rel = np.linalg.norm(S_new - S, "fro") / max(np.linalg.norm(S, "fro"), 1e-12)
        S = S_new
        if rel < tol:
            break
    return S


def ot_map(S0, S1):
    S0 = project_spd(S0)
    S1 = project_spd(S1)
    S0_s = mat_power(S0, 0.5)
    S0_is = mat_power(S0, -0.5)
    A = S0_is @ mat_power(S0_s @ S1 @ S0_s, 0.5) @ S0_is
    return sym(A)


def triu_vec(A):
    idx = np.triu_indices(A.shape[0])
    return A[idx]


def mat_from_triu(v, d):
    A = np.zeros((d, d), dtype=float)
    idx = np.triu_indices(d)
    A[idx] = v
    A[(idx[1], idx[0])] = v
    return sym(A)


def ridge_fit_predict(Z, train_end, ridge_grid=None):
    """Predict Z[t+1] from Z[t]. train_end is number of matrices used for train."""
    if ridge_grid is None:
        ridge_grid = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    X_all = Z[:-1]
    Y_all = Z[1:]
    n_train_pairs = train_end - 1
    n_val = max(10, int(0.2 * n_train_pairs))
    fit_n = max(5, n_train_pairs - n_val)
    X_fit, Y_fit = X_all[:fit_n], Y_all[:fit_n]
    X_val, Y_val = X_all[fit_n:n_train_pairs], Y_all[fit_n:n_train_pairs]

    def fit(lam, X, Y):
        Xa = np.column_stack([np.ones(len(X)), X])
        I = np.eye(Xa.shape[1])
        I[0, 0] = 0.0
        return np.linalg.solve(Xa.T @ Xa + lam * I, Xa.T @ Y)

    best_lam = ridge_grid[0]
    best_score = np.inf
    for lam in ridge_grid:
        coef = fit(lam, X_fit, Y_fit)
        pred = np.column_stack([np.ones(len(X_val)), X_val]) @ coef if len(X_val) else Y_fit
        score = float(np.mean((pred - Y_val) ** 2)) if len(X_val) else 0.0
        if score < best_score:
            best_score = score
            best_lam = lam
    coef = fit(best_lam, X_all[:n_train_pairs], Y_all[:n_train_pairs])
    pred_all = np.column_stack([np.ones(len(X_all)), X_all]) @ coef
    return pred_all, best_lam


def ridge_fit_predict_horizon(Z, train_end, horizon=1, ridge_grid=None):
    if ridge_grid is None:
        ridge_grid = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    X_all = Z[:-horizon]
    Y_all = Z[horizon:]
    n_train_pairs = train_end - horizon
    n_val = max(10, int(0.2 * n_train_pairs))
    fit_n = max(5, n_train_pairs - n_val)
    X_fit, Y_fit = X_all[:fit_n], Y_all[:fit_n]
    X_val, Y_val = X_all[fit_n:n_train_pairs], Y_all[fit_n:n_train_pairs]

    def fit(lam, X, Y):
        Xa = np.column_stack([np.ones(len(X)), X])
        I = np.eye(Xa.shape[1])
        I[0, 0] = 0.0
        return np.linalg.solve(Xa.T @ Xa + lam * I, Xa.T @ Y)

    best_lam = ridge_grid[0]
    best_score = np.inf
    for lam in ridge_grid:
        coef = fit(lam, X_fit, Y_fit)
        pred = np.column_stack([np.ones(len(X_val)), X_val]) @ coef if len(X_val) else Y_fit
        score = float(np.mean((pred - Y_val) ** 2)) if len(X_val) else 0.0
        if score < best_score:
            best_score = score
            best_lam = lam
    coef = fit(best_lam, X_all[:n_train_pairs], Y_all[:n_train_pairs])
    pred_all = np.column_stack([np.ones(len(X_all)), X_all]) @ coef
    return pred_all, best_lam


def evaluate_predictions(covs, preds, start_pair_idx):
    bw = []
    fro = []
    for i in range(start_pair_idx, len(covs) - 1):
        P = project_spd(preds[i])
        T = covs[i + 1]
        bw.append(bw2_cov(P, T))
        fro.append(np.linalg.norm(P - T, "fro") ** 2 / max(np.linalg.norm(T, "fro") ** 2, 1e-12))
    return {
        "bw2_mean": float(np.mean(bw)),
        "bw2_median": float(np.median(bw)),
        "rel_fro_mean": float(np.mean(fro)),
        "n_test_pairs": int(len(bw)),
    }


def evaluate_horizon_predictions(covs, preds, start_idx, horizon):
    bw = []
    fro = []
    for i in range(start_idx, len(covs) - horizon):
        P = project_spd(preds[i])
        T = covs[i + horizon]
        bw.append(bw2_cov(P, T))
        fro.append(np.linalg.norm(P - T, "fro") ** 2 / max(np.linalg.norm(T, "fro") ** 2, 1e-12))
    return {
        "bw2_mean": float(np.mean(bw)),
        "bw2_median": float(np.median(bw)),
        "rel_fro_mean": float(np.mean(fro)),
        "n_test_pairs": int(len(bw)),
    }


def run_direct_horizon_methods(covs, horizon=5, train_frac=0.65, name="experiment"):
    covs = np.array([project_spd(C) for C in covs])
    n, d, _ = covs.shape
    train_end = int(n * train_frac)
    train_end = max(20, min(train_end, n - 20))
    test_start = train_end - horizon
    results = {}

    # h-step persistence.
    preds = np.array([covs[i] for i in range(n - horizon)])
    results["persistence"] = evaluate_horizon_predictions(covs, preds, test_start, horizon)

    Z = np.vstack([triu_vec(C) for C in covs])
    pred_Z, lam = ridge_fit_predict_horizon(Z, train_end, horizon=horizon)
    preds = np.array([project_spd(mat_from_triu(z, d)) for z in pred_Z])
    results["euclidean_cov_ar"] = evaluate_horizon_predictions(covs, preds, test_start, horizon)
    results["euclidean_cov_ar"]["ridge"] = lam

    Z = np.vstack([triu_vec(mat_log(C)) for C in covs])
    pred_Z, lam = ridge_fit_predict_horizon(Z, train_end, horizon=horizon)
    preds = np.array([mat_exp(mat_from_triu(z, d)) for z in pred_Z])
    results["log_euclidean_ar"] = evaluate_horizon_predictions(covs, preds, test_start, horizon)
    results["log_euclidean_ar"]["ridge"] = lam

    ref = bw_barycenter(covs[:train_end])
    Z = []
    for C in covs:
        A = ot_map(ref, C)
        Z.append(triu_vec(A - np.eye(d)))
    Z = np.vstack(Z)
    pred_Z, lam = ridge_fit_predict_horizon(Z, train_end, horizon=horizon)
    preds = []
    for z in pred_Z:
        A = project_spd(np.eye(d) + mat_from_triu(z, d))
        preds.append(project_spd(A @ ref @ A))
    preds = np.array(preds)
    results["bw_fixed_ref_ar"] = evaluate_horizon_predictions(covs, preds, test_start, horizon)
    results["bw_fixed_ref_ar"]["ridge"] = lam

    return {
        "name": f"{name}_h{horizon}",
        "horizon": int(horizon),
        "n_matrices": int(n),
        "dimension": int(d),
        "train_end": int(train_end),
        "results": results,
    }


def run_methods(covs, train_frac=0.65, name="experiment"):
    covs = np.array([project_spd(C) for C in covs])
    n, d, _ = covs.shape
    train_end = int(n * train_frac)
    train_end = max(20, min(train_end, n - 20))
    test_start_pair = train_end - 1
    results = {}
    pred_store = {}

    # Persistence.
    preds = np.array([covs[i] for i in range(n - 1)])
    results["persistence"] = evaluate_predictions(covs, preds, test_start_pair)
    pred_store["persistence"] = preds

    # Euclidean AR on upper-triangular covariance entries.
    Z = np.vstack([triu_vec(C) for C in covs])
    pred_Z, lam = ridge_fit_predict(Z, train_end)
    preds = np.array([project_spd(mat_from_triu(z, d)) for z in pred_Z])
    results["euclidean_cov_ar"] = evaluate_predictions(covs, preds, test_start_pair)
    results["euclidean_cov_ar"]["ridge"] = lam

    # Log-Euclidean AR.
    Z = np.vstack([triu_vec(mat_log(C)) for C in covs])
    pred_Z, lam = ridge_fit_predict(Z, train_end)
    preds = np.array([mat_exp(mat_from_triu(z, d)) for z in pred_Z])
    results["log_euclidean_ar"] = evaluate_predictions(covs, preds, test_start_pair)
    results["log_euclidean_ar"]["ridge"] = lam

    # Fixed-reference BW tangent AR.
    ref = bw_barycenter(covs[:train_end])
    Z = []
    for C in covs:
        A = ot_map(ref, C)
        Z.append(triu_vec(A - np.eye(d)))
    Z = np.vstack(Z)
    pred_Z, lam = ridge_fit_predict(Z, train_end)
    preds = []
    for z in pred_Z:
        A = project_spd(np.eye(d) + mat_from_triu(z, d))
        preds.append(project_spd(A @ ref @ A))
    preds = np.array(preds)
    results["bw_fixed_ref_ar"] = evaluate_predictions(covs, preds, test_start_pair)
    results["bw_fixed_ref_ar"]["ridge"] = lam

    # BW increment persistence and AR: transport map from S_t to S_{t+1}.
    inc = []
    for i in range(n - 1):
        inc.append(triu_vec(ot_map(covs[i], covs[i + 1]) - np.eye(d)))
    inc = np.vstack(inc)
    # Repeat previous increment.
    preds = []
    for i in range(n - 1):
        if i == 0:
            A = np.eye(d)
        else:
            A = project_spd(np.eye(d) + mat_from_triu(inc[i - 1], d))
        preds.append(project_spd(A @ covs[i] @ A))
    preds = np.array(preds)
    results["bw_increment_persistence"] = evaluate_predictions(covs, preds, test_start_pair)

    # AR on increments: inc[t] from inc[t-1], then apply predicted increment to cov[t].
    pred_inc, lam = ridge_fit_predict(inc, train_end - 1)
    preds = []
    for i in range(n - 1):
        if i == 0:
            z = inc[0] * 0
        else:
            z = pred_inc[i - 1]
        A = project_spd(np.eye(d) + mat_from_triu(z, d))
        preds.append(project_spd(A @ covs[i] @ A))
    preds = np.array(preds)
    results["bw_increment_ar"] = evaluate_predictions(covs, preds, test_start_pair)
    results["bw_increment_ar"]["ridge"] = lam

    return {
        "name": name,
        "n_matrices": int(n),
        "dimension": int(d),
        "train_end": int(train_end),
        "results": results,
    }


def simulate_bwar_covariances(n=420, d=6, seed=123, noise=0.055):
    rng = np.random.default_rng(seed)
    q = d * (d + 1) // 2
    M = rng.normal(size=(d, d))
    ref = project_spd(M @ M.T / d + np.eye(d))
    B = rng.normal(size=(q, q))
    eig = max(abs(np.linalg.eigvals(B)))
    B = 0.72 * B / eig
    z = np.zeros((n, q))
    eps_scale = noise
    for t in range(1, n):
        z[t] = B @ z[t - 1] + eps_scale * rng.normal(size=q)
    covs = []
    for t in range(n):
        H = mat_from_triu(z[t], d)
        A = mat_exp(0.5 * H)
        covs.append(project_spd(A @ ref @ A))
    return np.array(covs)


def fetch_yahoo_adjclose(symbol, start="2014-01-01", end="2026-05-31"):
    def ts(date):
        y, m, d = map(int, date.split("-"))
        return int(time.mktime((y, m, d, 0, 0, 0, 0, 0, 0)))
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={ts(start)}&period2={ts(end)}&interval=1d&events=history&includeAdjustedClose=true"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urlopen(req, timeout=30).read().decode("utf-8")
    obj = json.loads(raw)
    res = obj["chart"]["result"][0]
    timestamps = res["timestamp"]
    quote = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
    dates = pd.to_datetime(timestamps, unit="s").date
    return pd.Series(adj, index=pd.to_datetime(dates), name=symbol).dropna()


def finance_rolling_covariances(symbols=None, window=60):
    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "USO"]
    prices = []
    for s in symbols:
        ser = fetch_yahoo_adjclose(s)
        prices.append(ser)
    px = pd.concat(prices, axis=1).dropna()
    # Work with percentage log returns so covariance matrices have a numerically
    # comfortable scale for repeated matrix square-root calculations.
    rets = 100.0 * np.log(px).diff().dropna()
    covs = []
    dates = []
    for i in range(window, len(rets)):
        C = np.cov(rets.iloc[i - window : i].to_numpy(), rowvar=False)
        C = project_spd(C + 1e-7 * np.eye(len(symbols)))
        covs.append(C)
        dates.append(str(rets.index[i].date()))
    return np.array(covs), dates, symbols


def mhealth_rolling_covariances(subject=1, window=100, step=25):
    path = ROOT / "data" / "mhealth" / "MHEALTHDATASET" / f"mHealth_subject{subject}.log"
    if not path.exists():
        raise FileNotFoundError(f"MHEALTH file not found: {path}")
    arr = np.loadtxt(path)
    # Use three accelerometers: chest columns 1-3, ankle 6-8, arm 15-17.
    cols = [0, 1, 2, 5, 6, 7, 14, 15, 16]
    X = arr[:, cols].astype(float)
    labels = arr[:, -1].astype(int)
    train_raw_end = int(0.65 * len(X))
    mu = X[:train_raw_end].mean(axis=0)
    sd = X[:train_raw_end].std(axis=0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    covs, labs, starts = [], [], []
    for start in range(0, len(X) - window + 1, step):
        W = X[start : start + window]
        C = np.cov(W, rowvar=False)
        C = project_spd(C + 1e-5 * np.eye(len(cols)))
        covs.append(C)
        counts = np.bincount(labels[start : start + window], minlength=13)
        labs.append(int(np.argmax(counts)))
        starts.append(int(start))
    return np.array(covs), labs, starts


def make_bar_chart(summary, path):
    rows = []
    for exp in summary["experiments"]:
        for method, metrics in exp["results"].items():
            rows.append((exp["name"], method, metrics["bw2_mean"]))
    W, H = 1500, 850
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 36)
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 22)
        small = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 18)
    except Exception:
        title_font = font = small = ImageFont.load_default()
    draw.text((50, 35), "BW-AR preliminary experiments: mean one-step BW^2 error", fill=(11, 37, 69), font=title_font)
    experiments = summary["experiments"]
    colors = [(203, 213, 225), (147, 197, 253), (96, 165, 250), (37, 99, 235), (52, 211, 153), (5, 150, 105)]
    y0 = 120
    for ei, exp in enumerate(experiments):
        x0 = 70 + ei * 720
        draw.text((x0, y0 - 35), exp["name"], fill=(15, 23, 42), font=font)
        vals = [(m, exp["results"][m]["bw2_mean"]) for m in exp["results"]]
        maxv = max(v for _, v in vals)
        chart_h = 470
        chart_w = 620
        draw.line((x0, y0 + chart_h, x0 + chart_w, y0 + chart_h), fill=(51, 65, 85), width=3)
        draw.line((x0, y0, x0, y0 + chart_h), fill=(51, 65, 85), width=3)
        bar_w = 72
        gap = 24
        for i, (m, v) in enumerate(vals):
            bh = int(chart_h * v / maxv) if maxv > 0 else 0
            bx = x0 + 35 + i * (bar_w + gap)
            by = y0 + chart_h - bh
            draw.rounded_rectangle((bx, by, bx + bar_w, y0 + chart_h), radius=8, fill=colors[i % len(colors)])
            draw.text((bx - 6, by - 24), f"{v:.2e}", fill=(15, 23, 42), font=small)
            label = m.replace("_", "\n")
            ly = y0 + chart_h + 15
            for line in label.split("\n")[:3]:
                draw.text((bx - 8, ly), line[:10], fill=(51, 65, 85), font=small)
                ly += 20
    img.save(path)


def main():
    experiments = []
    synth = simulate_bwar_covariances()
    experiments.append(run_methods(synth, train_frac=0.65, name="synthetic_fixed_ref"))

    try:
        covs, dates, symbols = finance_rolling_covariances()
        fin_res = run_methods(covs, train_frac=0.65, name="finance_etf_rolling_cov")
        fin_res["symbols"] = symbols
        fin_res["date_start"] = dates[0]
        fin_res["date_end"] = dates[-1]
        experiments.append(fin_res)
    except Exception as e:
        experiments.append({"name": "finance_etf_rolling_cov", "error": repr(e)})

    try:
        covs, labs, starts = mhealth_rolling_covariances(subject=1)
        mh_res = run_methods(covs, train_frac=0.65, name="mhealth_subject1_accel_cov")
        mh_res["window_samples"] = 100
        mh_res["step_samples"] = 25
        mh_res["sampling_rate_hz"] = 50
        mh_res["activity_labels_seen"] = sorted(set(labs))
        experiments.append(mh_res)
    except Exception as e:
        experiments.append({"name": "mhealth_subject1_accel_cov", "error": repr(e)})

    summary = {"experiments": experiments}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for exp in experiments:
        if "error" in exp:
            rows.append({"experiment": exp["name"], "method": "ERROR", "bw2_mean": exp["error"], "rel_fro_mean": ""})
            continue
        for method, metrics in exp["results"].items():
            rows.append({
                "experiment": exp["name"],
                "method": method,
                "bw2_mean": metrics["bw2_mean"],
                "bw2_median": metrics["bw2_median"],
                "rel_fro_mean": metrics["rel_fro_mean"],
                "n_test_pairs": metrics["n_test_pairs"],
                "ridge": metrics.get("ridge", ""),
            })
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    make_bar_chart(summary, OUT / "summary_bar.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
