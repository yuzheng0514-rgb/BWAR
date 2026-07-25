from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from bwar.bwar_experiments import (
    finance_rolling_covariances,
    mhealth_rolling_covariances,
    project_spd,
)
from bwar.bwar_online_synthetic import (
    online_bwar,
    online_euclidean,
    online_log_euclidean,
    online_persistence,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bwar_online_real"
DATA = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


METHODS = {
    "persistence": online_persistence,
    "online_euclidean": online_euclidean,
    "online_log_euclidean": online_log_euclidean,
    "bwar_fixed_warmup_ref": lambda covs, warmup: online_bwar(covs, warmup, mode="fixed_warmup"),
    "bwar_online_mean_ref": lambda covs, warmup: online_bwar(covs, warmup, mode="online_mean"),
    "bwar_forgetting_ref": lambda covs, warmup: online_bwar(covs, warmup, mode="forgetting"),
    "bwar_residual_eta_ref": lambda covs, warmup: online_bwar(covs, warmup, mode="residual_eta"),
    "bwar_residual_reset_ref": lambda covs, warmup: online_bwar(covs, warmup, mode="residual_reset"),
}


def choose_warmup(n: int) -> int:
    return int(max(50, min(250, 0.25 * n)))


def run_methods(dataset: str, covs: np.ndarray, meta: dict | None = None) -> list[dict]:
    covs = np.asarray([project_spd(C) for C in covs])
    n, d, _ = covs.shape
    if n < 90:
        return []
    warmup = choose_warmup(n)
    rows = []
    for method, fn in METHODS.items():
        try:
            metrics = fn(covs, warmup)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_matrices": int(n),
                    "dimension": int(d),
                    "warmup": int(warmup),
                    **(meta or {}),
                    **metrics,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_matrices": int(n),
                    "dimension": int(d),
                    "warmup": int(warmup),
                    **(meta or {}),
                    "error": repr(exc),
                }
            )
    return rows


def run_finance() -> list[dict]:
    rows = []
    for window in [20, 60, 120]:
        covs, dates, symbols = finance_rolling_covariances(window=window)
        rows.extend(
            run_methods(
                "finance_etf",
                covs,
                {
                    "config": f"window_{window}",
                    "window": int(window),
                    "date_start": dates[0],
                    "date_end": dates[-1],
                    "symbols": ",".join(symbols),
                },
            )
        )
    return rows


def run_mhealth() -> list[dict]:
    rows = []
    for subject in [1, 2, 3]:
        for window, step in [(100, 100), (200, 100), (400, 200)]:
            covs, labels, starts = mhealth_rolling_covariances(subject=subject, window=window, step=step)
            rows.extend(
                run_methods(
                    "mhealth",
                    covs,
                    {
                        "config": f"subject_{subject}_w{window}_s{step}",
                        "subject": int(subject),
                        "window": int(window),
                        "step": int(step),
                        "activity_labels_seen": ",".join(map(str, sorted(set(labels)))),
                    },
                )
            )
    return rows


def download_uci_har() -> Path:
    target = DATA / "uci_har"
    marker = target / "UCI HAR Dataset"
    if marker.exists():
        return marker
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "uci_har.zip"
    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as resp:
            zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    nested_zip = target / "UCI HAR Dataset.zip"
    if nested_zip.exists() and not marker.exists():
        with zipfile.ZipFile(nested_zip) as zf:
            zf.extractall(target)
    return marker


def read_har_split(base: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal_dir = base / split / "Inertial Signals"
    names = [
        f"body_acc_x_{split}.txt",
        f"body_acc_y_{split}.txt",
        f"body_acc_z_{split}.txt",
        f"body_gyro_x_{split}.txt",
        f"body_gyro_y_{split}.txt",
        f"body_gyro_z_{split}.txt",
        f"total_acc_x_{split}.txt",
        f"total_acc_y_{split}.txt",
        f"total_acc_z_{split}.txt",
    ]
    signals = [np.loadtxt(signal_dir / name) for name in names]
    X = np.stack(signals, axis=2)
    subjects = np.loadtxt(base / split / f"subject_{split}.txt").astype(int)
    labels = np.loadtxt(base / split / f"y_{split}.txt").astype(int)
    return X, subjects, labels


def har_covariances_for_subject(base: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    all_covs = []
    all_labels = []
    for split in ["train", "test"]:
        X, subjects, labels = read_har_split(base, split)
        idx = np.where(subjects == subject)[0]
        for row_idx in idx:
            W = X[row_idx]
            C = np.cov(W, rowvar=False)
            all_covs.append(project_spd(C + 1e-6 * np.eye(W.shape[1])))
            all_labels.append(labels[row_idx])
    return np.asarray(all_covs), np.asarray(all_labels)


def run_har() -> list[dict]:
    rows = []
    base = download_uci_har()
    for subject in range(1, 31):
        covs, labels = har_covariances_for_subject(base, subject)
        if len(covs) < 90:
            continue
        rows.extend(
            run_methods(
                "uci_har",
                covs,
                {
                    "config": f"subject_{subject}",
                    "subject": int(subject),
                    "activity_labels_seen": ",".join(map(str, sorted(set(labels.tolist())))),
                },
            )
        )
    return rows


def write_outputs(rows: list[dict]) -> None:
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw_results.csv", index=False)
    ok = raw[~raw.get("bw2_mean", pd.Series(index=raw.index)).isna()].copy()
    group_cols = ["dataset", "config", "method"]
    summary = (
        ok.groupby(group_cols, as_index=False)
        .agg(
            bw2_mean=("bw2_mean", "mean"),
            bw2_median=("bw2_median", "mean"),
            bw2_q90=("bw2_q90", "mean"),
            n_matrices=("n_matrices", "mean"),
            dimension=("dimension", "mean"),
        )
        .sort_values(["dataset", "config", "bw2_mean"])
    )
    summary.to_csv(OUT / "summary_by_config.csv", index=False)
    dataset_summary = (
        ok.groupby(["dataset", "method"], as_index=False)
        .agg(
            bw2_mean=("bw2_mean", "mean"),
            bw2_mean_sd=("bw2_mean", "std"),
            bw2_median=("bw2_median", "mean"),
            bw2_q90=("bw2_q90", "mean"),
            n_runs=("bw2_mean", "size"),
        )
        .sort_values(["dataset", "bw2_mean"])
    )
    dataset_summary.to_csv(OUT / "summary_by_dataset.csv", index=False)
    best = dataset_summary.loc[dataset_summary.groupby("dataset")["bw2_mean"].idxmin()]
    best.to_csv(OUT / "best_by_dataset.csv", index=False)
    payload = {
        "best_by_dataset": best.to_dict(orient="records"),
        "summary_by_dataset": dataset_summary.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Best by dataset:")
    print(best.to_string(index=False))
    print("\nSummary by dataset:")
    print(dataset_summary.to_string(index=False))


def main() -> None:
    rows: list[dict] = []
    for name, fn in [("finance", run_finance), ("mhealth", run_mhealth), ("uci_har", run_har)]:
        print(f"Running {name}...")
        try:
            rows.extend(fn())
        except Exception as exc:
            rows.append({"dataset": name, "method": "ERROR", "error": repr(exc)})
            print(f"{name} failed: {exc!r}")
        write_outputs(rows)


if __name__ == "__main__":
    main()
