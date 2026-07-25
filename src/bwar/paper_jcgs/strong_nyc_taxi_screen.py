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
    DATA,
    DEFAULT_OUT,
    rolling_gaussians_from_array,
    run_single_series,
    set_reference_library_mode,
    summarize_outputs,
)
from bwar.paper_jcgs.rolling_origin_backtest import (  # noqa: E402
    run_rolling_origin_series,
    write_outputs as write_rolling_outputs,
)


TAXI_DATA = DATA / "nyc_taxi"
SCREEN_OUT = DEFAULT_OUT / "domain_metric_screen"
ROLLING_OUT = DEFAULT_OUT / "rolling_origin_domain_final"


def taxi_trip_path(service: str, month: str) -> Path:
    service = service.lower()
    if service not in {"yellow", "green"}:
        raise ValueError("service must be 'yellow' or 'green'")
    return TAXI_DATA / f"{service}_tripdata_{month}.parquet"


def taxi_pickup_time_col(service: str) -> str:
    return "tpep_pickup_datetime" if service.lower() == "yellow" else "lpep_pickup_datetime"


def load_taxi_hourly_matrix(path: Path, *, pickup_time_col: str) -> pd.DataFrame:
    trips = pd.read_parquet(path, columns=[pickup_time_col, "PULocationID"])
    trips = trips.dropna(subset=[pickup_time_col, "PULocationID"]).copy()
    trips[pickup_time_col] = pd.to_datetime(trips[pickup_time_col], errors="coerce")
    trips = trips.dropna(subset=[pickup_time_col])
    month_start = trips[pickup_time_col].min().floor("D")
    month_end = trips[pickup_time_col].max().floor("h") + pd.Timedelta(hours=1)
    trips = trips[(trips[pickup_time_col] >= month_start) & (trips[pickup_time_col] < month_end)].copy()
    trips["hour"] = trips[pickup_time_col].dt.floor("h")
    trips["zone"] = trips["PULocationID"].astype(int).astype(str)
    counts = trips.groupby(["hour", "zone"], observed=True).size().rename("pickups")
    matrix = counts.unstack().fillna(0.0).sort_index().astype(float)
    full_index = pd.date_range(month_start, month_end - pd.Timedelta(hours=1), freq="h")
    matrix = matrix.reindex(full_index, fill_value=0.0)
    matrix.columns = [str(col) for col in matrix.columns]
    return matrix


def cached_taxi_hourly_matrix(service: str, month: str, *, refresh: bool = False) -> pd.DataFrame:
    cache = TAXI_DATA / f"{service}_hourly_pickups_{month}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    path = taxi_trip_path(service, month)
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = load_taxi_hourly_matrix(path, pickup_time_col=taxi_pickup_time_col(service))
    TAXI_DATA.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(cache)
    return matrix


def taxi_gaussians(
    service: str,
    month: str,
    *,
    window: int,
    step: int,
    n_zones: int,
    max_matrices: int = 750,
    refresh: bool = False,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    matrix = cached_taxi_hourly_matrix(service, month, refresh=refresh)
    early_end = max(window, int(0.45 * len(matrix)))
    early = matrix.iloc[:early_end]
    nonzero_rate = (early > 0.0).mean(axis=0)
    variability = np.log1p(early.clip(lower=0.0)).std(axis=0)
    score = (nonzero_rate * variability).replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < n_zones:
        raise ValueError(f"NYC taxi {service} {month} has only {len(score)} usable pickup zones; requested {n_zones}")
    selected = list(score.sort_values(ascending=False).head(n_zones).index)
    X = np.log1p(matrix[selected].clip(lower=0.0)).to_numpy(dtype=float)
    result = rolling_gaussians_from_array(
        X,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
        return_windows=return_windows,
    )
    meta = {
        "window": int(window),
        "step": int(step),
        "n_zones": int(n_zones),
        "service": service,
        "month": month,
        "transform": "log1p hourly taxi pickup-zone counts",
        "source": "NYC TLC trip records",
        "selected_zone_ids": ",".join(selected[: min(len(selected), 20)]),
        "n_source_hours": int(len(matrix)),
        "n_source_zones": int(matrix.shape[1]),
    }
    if return_windows:
        means, covs, raw_windows, starts = result
        return means, covs, raw_windows, starts, meta
    means, covs = result
    return means, covs, meta


def iter_taxi_jobs(service: str, month: str, *, quick: bool, refresh: bool = False):
    configs = (
        [(24, 6, 40), (48, 12, 60)]
        if quick
        else [(24, 6, 40), (24, 6, 60), (48, 12, 40), (48, 12, 60), (72, 12, 60), (72, 24, 80)]
    )
    for window, step, n_zones in configs:
        means, covs, meta = taxi_gaussians(
            service,
            month,
            window=window,
            step=step,
            n_zones=n_zones,
            refresh=refresh,
        )
        yield f"nyc_taxi_{service}_{month}_w{window}_s{step}_d{n_zones}", "nyc_taxi", means, covs, meta


def run_fixed_screen(
    *,
    service: str,
    month: str,
    horizons: list[int],
    quick: bool,
    ar_model: str,
    reference_library: str,
    out_dir: Path,
    refresh: bool = False,
) -> dict[str, object]:
    set_reference_library_mode(reference_library)
    tag = (
        f"nyc_taxi_{service}_{month.replace('-', '')}_h{'-'.join(map(str, horizons))}_{ar_model}"
        + ("_quick" if quick else "")
    )
    if reference_library != "full":
        tag += f"_{reference_library}"

    started = time.time()
    rows: list[pd.DataFrame] = []
    ref_tables: list[pd.DataFrame] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for job, dataset, means, covs, meta in iter_taxi_jobs(service, month, quick=quick, refresh=refresh):
        print(f"Running {job} horizons={horizons} n={len(covs)} d={covs.shape[1]}", flush=True)
        raw, refs = run_single_series(
            job=job,
            dataset=dataset,
            means=means,
            covs=covs,
            meta=meta,
            horizons=horizons,
            ar_model=ar_model,
        )
        if not raw.empty:
            rows.append(raw)
            raw_all = pd.concat(rows, ignore_index=True)
            raw_all.to_csv(out_dir / f"raw_results_{tag}.csv", index=False)
            summary, summary_h = summarize_outputs(raw_all)
            summary.to_csv(out_dir / f"summary_by_dataset_{tag}.csv", index=False)
            summary_h.to_csv(out_dir / f"summary_by_dataset_horizon_{tag}.csv", index=False)
            print(summary.to_string(index=False))
        if not refs.empty:
            ref_tables.append(refs)
            pd.concat(ref_tables, ignore_index=True).to_csv(out_dir / f"reference_table_{tag}.csv", index=False)

    raw_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary, summary_h = summarize_outputs(raw_all) if not raw_all.empty else (pd.DataFrame(), pd.DataFrame())
    metadata = {
        "service": service,
        "month": month,
        "horizons": horizons,
        "quick": bool(quick),
        "ar_model": ar_model,
        "reference_library": reference_library,
        "tag": tag,
        "elapsed_seconds": round(time.time() - started, 3),
        "out_dir": str(out_dir),
    }
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary_by_dataset": summary.to_dict(orient="records"),
                "summary_by_dataset_horizon": summary_h.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata


def run_rolling_screen(
    *,
    service: str,
    month: str,
    window: int,
    step: int,
    n_zones: int,
    horizons: list[int],
    ar_model: str,
    reference_library: str,
    max_origins: int,
    out_dir: Path,
    refresh: bool = False,
) -> dict[str, object]:
    set_reference_library_mode(reference_library)
    tag = (
        f"nyc_taxi_{service}_{month.replace('-', '')}_w{window}_s{step}_d{n_zones}_"
        f"h{'-'.join(map(str, horizons))}_{ar_model}_orig{max_origins}"
    )
    if reference_library != "full":
        tag += f"_{reference_library}"

    means, covs, raw_windows, starts, meta = taxi_gaussians(
        service,
        month,
        window=window,
        step=step,
        n_zones=n_zones,
        refresh=refresh,
        return_windows=True,
    )
    started = time.time()
    print(f"Rolling-origin nyc_taxi_{service}_{month}_w{window}_s{step}_d{n_zones} n={len(covs)} d={covs.shape[1]}", flush=True)
    raw, refs = run_rolling_origin_series(
        job=f"nyc_taxi_{service}_{month}_w{window}_s{step}_d{n_zones}",
        dataset="nyc_taxi",
        means=means,
        covs=covs,
        meta=meta,
        horizons=horizons,
        raw_windows=raw_windows,
        window_starts=starts,
        window_size=window,
        ar_model=ar_model,
        max_origins=max_origins,
    )
    rows = [raw] if not raw.empty else []
    ref_tables = [refs] if not refs.empty else []
    write_rolling_outputs(rows, ref_tables, out_dir=out_dir, tag=tag)
    metadata = {
        "service": service,
        "month": month,
        "window": int(window),
        "step": int(step),
        "n_zones": int(n_zones),
        "horizons": horizons,
        "ar_model": ar_model,
        "reference_library": reference_library,
        "max_origins": int(max_origins),
        "tag": tag,
        "elapsed_seconds": round(time.time() - started, 3),
        "out_dir": str(out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"run_metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary_path = out_dir / f"summary_by_dataset_{tag}.csv"
    if summary_path.exists():
        print(pd.read_csv(summary_path).to_string(index=False))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen NYC taxi hourly pickup-zone streams for BWAR.")
    parser.add_argument("--mode", choices=["fixed", "rolling"], default="fixed")
    parser.add_argument("--service", choices=["yellow", "green"], default="yellow")
    parser.add_argument("--month", default="2024-01")
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--ar-model", choices=["diag", "full"], default="diag")
    parser.add_argument("--reference-library", choices=["full", "no_barycenter", "fast"], default="fast")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--n-zones", type=int, default=40)
    parser.add_argument("--max-origins", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if args.mode == "fixed":
        metadata = run_fixed_screen(
            service=args.service,
            month=args.month,
            horizons=horizons,
            quick=bool(args.quick),
            ar_model=args.ar_model,
            reference_library=args.reference_library,
            out_dir=args.out_dir or SCREEN_OUT,
            refresh=bool(args.refresh),
        )
    else:
        metadata = run_rolling_screen(
            service=args.service,
            month=args.month,
            window=args.window,
            step=args.step,
            n_zones=args.n_zones,
            horizons=horizons,
            ar_model=args.ar_model,
            reference_library=args.reference_library,
            max_origins=args.max_origins,
            out_dir=args.out_dir or ROLLING_OUT,
            refresh=bool(args.refresh),
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
