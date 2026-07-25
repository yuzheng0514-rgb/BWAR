from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from bwar.bwar_experiments import (
    bw2_cov,
    bw_barycenter,
    finance_rolling_covariances,
    mat_exp,
    mat_from_triu,
    mat_log,
    mhealth_rolling_covariances,
    project_spd,
    triu_vec,
)
from bwar.bwar_online_real import download_uci_har, har_covariances_for_subject
from bwar.da_bwar_experiments import (
    OUT,
    bw_forecast_score,
    decode_z,
    encode_covs,
    evaluate_reference,
    fit_ridge_var,
    select_reference_by_validation,
)


DATA = Path(__file__).resolve().parents[2] / "data"
SCREEN_OUT = OUT / "baseline_screen"
SCREEN_OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def download_url(url: str, path: Path) -> Path:
    if path.exists():
        return path
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=180) as resp:
        path.write_bytes(resp.read())
    return path


def rolling_covariances_from_frame(
    X: pd.DataFrame,
    *,
    window: int,
    step: int,
    max_matrices: int = 800,
    ridge: float = 1e-5,
) -> np.ndarray:
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X.loc[:, X.std(axis=0) > 1e-10]
    train_end = max(window, int(0.65 * len(X)))
    mu = X.iloc[:train_end].mean(axis=0)
    sd = X.iloc[:train_end].std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = ((X - mu) / sd).to_numpy(float)
    starts = list(range(0, len(Z) - window + 1, step))
    if len(starts) > max_matrices:
        idx = np.linspace(0, len(starts) - 1, max_matrices, dtype=int)
        starts = [starts[i] for i in idx]
    covs = []
    for start in starts:
        W = Z[start : start + window]
        C = np.cov(W, rowvar=False)
        covs.append(project_spd(C + ridge * np.eye(C.shape[0])))
    return np.asarray(covs)


def rolling_covariances_from_array(
    X: np.ndarray,
    *,
    window: int,
    step: int,
    max_matrices: int = 800,
    ridge: float = 1e-5,
) -> np.ndarray:
    cols = [f"x{j}" for j in range(X.shape[1])]
    return rolling_covariances_from_frame(
        pd.DataFrame(X, columns=cols),
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=ridge,
    )


def air_quality_covariances(window: int = 48, step: int = 12) -> np.ndarray:
    target = DATA / "air_quality_uci.zip"
    download_url("https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip", target)
    with zipfile.ZipFile(target) as zf:
        raw = zf.read("AirQualityUCI.csv")
    df = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",")
    df = df.drop(columns=[c for c in ["Date", "Time", "Unnamed: 15", "Unnamed: 16"] if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(-200, np.nan).interpolate(limit_direction="both")
    cols = [c for c in df.columns if c.startswith("PT08") or c in ["C6H6(GT)", "T", "RH", "AH"]]
    return rolling_covariances_from_frame(df[cols], window=window, step=step, max_matrices=700)


def appliances_covariances(window: int = 144, step: int = 24) -> np.ndarray:
    target = DATA / "energydata_complete.csv"
    download_url(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
        target,
    )
    df = pd.read_csv(target)
    cols = [
        "T1",
        "RH_1",
        "T2",
        "RH_2",
        "T3",
        "RH_3",
        "T_out",
        "RH_out",
        "Windspeed",
        "Tdewpoint",
    ]
    return rolling_covariances_from_frame(df[cols], window=window, step=step, max_matrices=700)


def occupancy_detection_covariances(split: str, window: int = 144, step: int = 24) -> np.ndarray:
    target = DATA / "occupancy_detection.zip"
    download_url(
        "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
        target,
    )
    with zipfile.ZipFile(target) as zf:
        raw = zf.read(f"{split}.txt")
    df = pd.read_csv(io.BytesIO(raw))
    cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    return rolling_covariances_from_frame(df[cols], window=window, step=step, max_matrices=900)


def eeg_eye_covariances(window: int = 128, step: int = 32) -> np.ndarray:
    target = DATA / "eeg_eye_state.arff"
    download_url("https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff", target)
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    rows = []
    for line in lines[start:]:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        rows.append([float(x) for x in line.split(",")])
    arr = np.asarray(rows)
    X = pd.DataFrame(arr[:, :14], columns=[f"eeg_{j + 1}" for j in range(14)])
    return rolling_covariances_from_frame(X, window=window, step=step, max_matrices=700)


def download_hapt() -> Path:
    target = DATA / "hapt"
    marker = target / "RawData"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "hapt.zip"
    download_url(
        "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
        zip_path,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    return target


def hapt_covariances_for_subject(subject: int, window: int = 128, step: int = 64) -> np.ndarray:
    base = download_hapt() / "RawData"
    acc_files = sorted(base.glob(f"acc_exp*_user{subject:02d}.txt"))
    blocks = []
    for acc_path in acc_files:
        gyro_path = acc_path.with_name(acc_path.name.replace("acc_", "gyro_"))
        if not gyro_path.exists():
            continue
        acc = np.loadtxt(acc_path)
        gyro = np.loadtxt(gyro_path)
        X = np.column_stack([acc, gyro])
        if len(X) >= window:
            blocks.append(X)
    if not blocks:
        return np.empty((0, 6, 6))
    X = np.vstack(blocks)
    return rolling_covariances_from_array(X, window=window, step=step, max_matrices=900)


def accgyro_covariances(window: int = 256, step: int = 64) -> np.ndarray:
    target = DATA / "accgyro_phone.zip"
    download_url(
        "https://archive.ics.uci.edu/static/public/755/accelerometer+gyro+mobile+phone+dataset.zip",
        target,
    )
    with zipfile.ZipFile(target) as zf:
        raw = zf.read("accelerometer_gyro_mobile_phone_dataset.csv")
    df = pd.read_csv(io.BytesIO(raw))
    cols = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
    return rolling_covariances_from_frame(df[cols], window=window, step=step, max_matrices=900)


def gas_drift_covariances(window: int = 96, step: int = 24, n_features: int = 8) -> np.ndarray:
    target = DATA / "gas_sensor_drift.zip"
    download_url(
        "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        target,
    )
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
    X = pd.DataFrame(np.asarray(rows), columns=[f"gas_{j+1}" for j in range(n_features)])
    return rolling_covariances_from_frame(X, window=window, step=step, max_matrices=900)


def homegas_covariances(record_id: int, window: int = 256, step: int = 64) -> np.ndarray:
    target = DATA / "homegas.zip"
    download_url(
        "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
        target,
    )
    outer = zipfile.ZipFile(target)
    inner_bytes = outer.read("HT_Sensor_dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        raw = inner.read("HT_Sensor_dataset.dat").decode("utf-8", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 12:
            continue
        rid = int(parts[0])
        if rid != record_id:
            continue
        vals = [float(x) for x in parts[2:12]]
        rows.append(vals)
    if not rows:
        return np.empty((0, 10, 10))
    X = pd.DataFrame(rows, columns=[f"sensor_{j+1}" for j in range(10)])
    return rolling_covariances_from_frame(X, window=window, step=step, max_matrices=900)


def fit_transform_ar_forecast(
    covs: np.ndarray,
    *,
    transform: str,
    fit_end: int,
    val_end: int,
    test_end: int,
    model: str = "diag",
    lam: float = 1e-3,
) -> dict:
    d = covs.shape[1]
    if transform == "euclidean":
        Z = np.vstack([triu_vec(C) for C in covs])

        def decode(z: np.ndarray) -> np.ndarray:
            return project_spd(mat_from_triu(z, d))

    elif transform == "log_euclidean":
        Z = np.vstack([triu_vec(mat_log(C)) for C in covs])

        def decode(z: np.ndarray) -> np.ndarray:
            return project_spd(mat_exp(mat_from_triu(z, d)))

    else:
        raise ValueError(transform)
    W = fit_ridge_var(Z, 0, val_end, lam=lam, model=model)
    errs = []
    for t in range(val_end - 1, test_end - 1):
        pred_z = np.r_[1.0, Z[t]] @ W
        errs.append(bw2_cov(decode(pred_z), covs[t + 1]))
    return {"test_bw": float(np.mean(errs)), "n_eval": int(len(errs))}


def persistence_forecast(covs: np.ndarray, val_end: int, test_end: int) -> dict:
    errs = [bw2_cov(covs[t], covs[t + 1]) for t in range(val_end - 1, test_end - 1)]
    return {"test_bw": float(np.mean(errs)), "n_eval": int(len(errs))}


def evaluate_job(job: str, dataset: str, covs: np.ndarray, meta: dict) -> tuple[pd.DataFrame, dict] | None:
    covs = np.asarray([project_spd(C) for C in covs])
    n = len(covs)
    if n < 120:
        return None
    fit_end, val_end, test_end = int(0.45 * n), int(0.65 * n), n
    rows: list[dict] = []

    def add(method: str, metrics: dict, selected_reference: str = "") -> None:
        rows.append(
            {
                "job": job,
                "dataset": dataset,
                "method": method,
                "selected_reference": selected_reference,
                "n_matrices": int(n),
                "dimension": int(covs.shape[1]),
                **meta,
                **metrics,
            }
        )

    add("persistence", persistence_forecast(covs, val_end, test_end))
    for transform in ["euclidean", "log_euclidean"]:
        for model in ["diag", "full"]:
            try:
                add(f"{transform}_{model}_ar", fit_transform_ar_forecast(
                    covs, transform=transform, fit_end=fit_end, val_end=val_end, test_end=test_end, model=model
                ))
            except Exception as exc:
                add(f"{transform}_{model}_ar", {"test_bw": np.nan, "error": repr(exc)})

    for ref_name, ref in {
        "pooled_cov": project_spd(np.mean(covs[:fit_end], axis=0)),
        "log_euclidean_mean": mat_exp(np.mean([mat_log(C) for C in covs[:fit_end]], axis=0)),
        "bw_barycenter": bw_barycenter(covs[: min(fit_end, 250)]),
    }.items():
        metrics = evaluate_reference(covs, ref, fit_end=fit_end, val_end=val_end, test_end=test_end)
        add(f"bwar_{ref_name}_diag_ar", {"test_bw": metrics["test_bw"], "test_rel_tangent": metrics["test_rel_tangent"]})

    selected_name, _, selected_metrics, candidate_table = select_reference_by_validation(
        covs, fit_end=fit_end, val_end=val_end, test_end=test_end
    )
    candidate_table.insert(0, "job", job)
    candidate_table.insert(1, "dataset", dataset)
    for key, value in meta.items():
        candidate_table[key] = value
    candidate_table.to_csv(SCREEN_OUT / f"{job}_candidate_refs.csv", index=False)
    add(
        "da_bwar_selected_diag_ar",
        {"test_bw": selected_metrics["test_bw"], "test_rel_tangent": selected_metrics["test_rel_tangent"]},
        selected_name,
    )

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["test_bw"]).copy()
    da = valid[valid["method"] == "da_bwar_selected_diag_ar"].iloc[0]
    non_da = valid[valid["method"] != "da_bwar_selected_diag_ar"]
    best_all = valid.loc[valid["test_bw"].idxmin()]
    best_non_da = non_da.loc[non_da["test_bw"].idxmin()]
    summary = {
        "job": job,
        "dataset": dataset,
        **meta,
        "n_matrices": int(n),
        "dimension": int(covs.shape[1]),
        "selected_reference": selected_name,
        "da_test_bw": float(da["test_bw"]),
        "best_non_da_method": str(best_non_da["method"]),
        "best_non_da_bw": float(best_non_da["test_bw"]),
        "best_all_method": str(best_all["method"]),
        "best_all_bw": float(best_all["test_bw"]),
        "da_gain_vs_best_non_da_bw": 1.0 - float(da["test_bw"]) / max(float(best_non_da["test_bw"]), 1e-12),
        "da_is_best": bool(da["method"] == best_all["method"]),
    }
    if "test_rel_tangent" in da and not pd.isna(da.get("test_rel_tangent", np.nan)):
        summary["da_test_rel_tangent"] = float(da["test_rel_tangent"])
    return df, summary


def iter_jobs(args: argparse.Namespace):
    if args.datasets in {"all", "uci_har"}:
        base = download_uci_har()
        subjects = range(1, 31 if not args.quick else 11)
        for subject in subjects:
            covs, labels = har_covariances_for_subject(base, subject)
            yield f"uci_har_s{subject}", "uci_har", covs, {"subject": int(subject)}
    if args.datasets in {"all", "mhealth"}:
        for subject in range(1, 11 if not args.quick else 4):
            for window, step in [(100, 100), (200, 200), (400, 400)]:
                try:
                    covs, labels, _ = mhealth_rolling_covariances(subject=subject, window=window, step=step)
                except FileNotFoundError:
                    continue
                yield f"mhealth_s{subject}_w{window}", "mhealth", covs, {"subject": int(subject), "window": window, "step": step}
    if args.datasets in {"all", "finance"}:
        for window in [20, 60, 120]:
            covs, dates, symbols = finance_rolling_covariances(window=window)
            yield f"finance_w{window}", "finance_etf", covs, {"window": window}
    if args.datasets in {"all", "air_quality"}:
        for window, step in [(24, 6), (48, 12), (96, 24)]:
            covs = air_quality_covariances(window=window, step=step)
            yield f"air_quality_w{window}_s{step}", "air_quality", covs, {"window": window, "step": step}
    if args.datasets in {"all", "appliances"}:
        for window, step in [(72, 12), (144, 24), (288, 48)]:
            covs = appliances_covariances(window=window, step=step)
            yield f"appliances_w{window}_s{step}", "appliances", covs, {"window": window, "step": step}
    if args.datasets in {"all", "occupancy"}:
        for split in ["datatraining", "datatest", "datatest2"]:
            for window, step in [(72, 12), (144, 24), (288, 48)]:
                covs = occupancy_detection_covariances(split=split, window=window, step=step)
                yield f"occupancy_{split}_w{window}", "occupancy", covs, {"split": split, "window": window, "step": step}
    if args.datasets in {"all", "eeg_eye"}:
        for window, step in [(64, 16), (128, 32), (256, 64)]:
            covs = eeg_eye_covariances(window=window, step=step)
            yield f"eeg_eye_w{window}_s{step}", "eeg_eye", covs, {"window": window, "step": step}
    if args.datasets in {"all", "hapt"}:
        subjects = range(1, 31 if not args.quick else 11)
        for subject in subjects:
            for window, step in [(128, 64), (256, 128)]:
                covs = hapt_covariances_for_subject(subject=subject, window=window, step=step)
                yield f"hapt_s{subject}_w{window}", "hapt", covs, {"subject": int(subject), "window": window, "step": step}
    if args.datasets in {"all", "accgyro"}:
        for window, step in [(128, 32), (256, 64), (512, 128)]:
            covs = accgyro_covariances(window=window, step=step)
            yield f"accgyro_w{window}_s{step}", "accgyro_phone", covs, {"window": window, "step": step}
    if args.datasets in {"all", "gas_drift"}:
        for window, step in [(64, 16), (96, 24), (128, 32)]:
            covs = gas_drift_covariances(window=window, step=step, n_features=8)
            yield f"gas8_w{window}_s{step}", "gas_drift8", covs, {"window": window, "step": step, "n_features": 8}
    if args.datasets in {"all", "homegas"}:
        record_ids = range(0, 10 if args.quick else 20)
        for rid in record_ids:
            for window, step in [(256, 64), (384, 96)]:
                covs = homegas_covariances(record_id=rid, window=window, step=step)
                yield f"homegas_id{rid}_w{window}", "homegas", covs, {"record_id": int(rid), "window": window, "step": step}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        choices=["all", "uci_har", "mhealth", "finance", "air_quality", "appliances", "occupancy", "eeg_eye", "hapt", "accgyro", "gas_drift", "homegas"],
        default="air_quality",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    tag = args.datasets + ("_quick" if args.quick else "")
    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for job, dataset, covs, meta in iter_jobs(args):
        print(f"Running {job}...")
        result = evaluate_job(job, dataset, covs, meta)
        if result is None:
            continue
        rows, summary = result
        all_rows.append(rows)
        summary_rows.append(summary)
        pd.concat(all_rows, ignore_index=True).to_csv(SCREEN_OUT / f"forecast_baselines_long_{tag}.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(SCREEN_OUT / f"forecast_baselines_summary_{tag}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print("No eligible jobs.")
        return
    strong = summary.sort_values("da_gain_vs_best_non_da_bw", ascending=False)
    strong.to_csv(SCREEN_OUT / f"forecast_baselines_ranked_{tag}.csv", index=False)
    print("\nTop DA-BWAR forecast cases:")
    cols = ["job", "dataset", "selected_reference", "da_test_bw", "best_non_da_method", "best_non_da_bw", "da_gain_vs_best_non_da_bw", "da_is_best"]
    print(strong[cols].head(20).to_string(index=False))
    print("\nDataset summary:")
    print(
        summary.groupby("dataset", as_index=False)
        .agg(
            mean_gain_vs_best_non_da=("da_gain_vs_best_non_da_bw", "mean"),
            positive_rate=("da_gain_vs_best_non_da_bw", lambda x: float(np.mean(np.asarray(x) > 0))),
            best_rate=("da_is_best", "mean"),
            n_jobs=("job", "size"),
        )
        .sort_values("mean_gain_vs_best_non_da", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
