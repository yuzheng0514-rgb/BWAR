from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.real_bwar_theory_matched import (  # noqa: E402
    DEFAULT_OUT,
    rolling_gaussians_from_array,
    run_single_series,
    set_reference_library_mode,
    summarize_outputs,
)


SCREEN_OUT = DEFAULT_OUT / "strong_screen"

BIKESHARE_MONTH_URLS = {
    "divvy": "https://divvy-tripdata.s3.amazonaws.com/{month}-divvy-tripdata.zip",
    "capital_bikeshare": "https://s3.amazonaws.com/capitalbikeshare-data/{month}-capitalbikeshare-tripdata.zip",
    "baywheels": "https://s3.amazonaws.com/baywheels-data/{month}-baywheels-tripdata.csv.zip",
    "bluebikes": "https://s3.amazonaws.com/hubway-data/{month}-bluebikes-tripdata.zip",
    "citibike": "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.zip",
    "citibike_jc": "https://s3.amazonaws.com/tripdata/JC-{month}-citibike-tripdata.csv.zip",
}


def download_url(url: str, path: Path, *, timeout: int = 240) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        path.write_bytes(resp.read())
    return path


def bikeshare_zip_path(system: str, month: str) -> Path:
    if system not in BIKESHARE_MONTH_URLS:
        raise ValueError(f"unknown bikeshare system: {system}")
    url = BIKESHARE_MONTH_URLS[system].format(month=month)
    return download_url(url, ROOT / "data" / system / f"{month}.zip")


def _read_monthly_station_counts(path: Path) -> pd.Series:
    counts: pd.Series | None = None
    with zipfile.ZipFile(path) as zf:
        csv_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".csv")
            and "__macosx/" not in name.lower()
            and not Path(name).name.startswith("._")
        ]
        if not csv_names:
            raise ValueError(f"no CSV file in {path}")
        for csv_name in csv_names:
            with zf.open(csv_name) as fh:
                header = pd.read_csv(fh, nrows=0)
            if "started_at" not in header.columns:
                continue
            station_col = "start_station_id" if "start_station_id" in header.columns else "start_station_name"
            if station_col not in header.columns:
                continue
            usecols = ["started_at", station_col]
            with zf.open(csv_name) as fh:
                for chunk in pd.read_csv(fh, usecols=usecols, chunksize=350_000):
                    chunk = chunk.dropna(subset=["started_at", station_col]).copy()
                    if chunk.empty:
                        continue
                    chunk["hour"] = pd.to_datetime(chunk["started_at"], errors="coerce").dt.floor("h")
                    chunk = chunk.dropna(subset=["hour"])
                    chunk[station_col] = chunk[station_col].astype(str)
                    part = chunk.groupby(["hour", station_col], observed=True).size()
                    counts = part if counts is None else counts.add(part, fill_value=0.0)
    if counts is None:
        return pd.Series(dtype=float, name="count")
    counts.name = "count"
    return counts.astype(float)


def load_bikeshare_hourly(system: str, months: tuple[str, ...]) -> pd.DataFrame:
    cache_path = ROOT / "data" / system / f"hourly_{'_'.join(months)}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    monthly = [_read_monthly_station_counts(bikeshare_zip_path(system, month)) for month in months]
    counts = monthly[0]
    for part in monthly[1:]:
        counts = counts.add(part, fill_value=0.0)
    frame = counts.unstack(fill_value=0.0).sort_index()
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(full_index, fill_value=0.0)
    frame.columns = [str(col) for col in frame.columns]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)
    return frame


def bikeshare_gaussians(
    system: str,
    months: tuple[str, ...],
    *,
    window: int = 48,
    step: int = 12,
    n_stations: int = 60,
    max_matrices: int = 750,
    return_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = load_bikeshare_hourly(system, months)
    early_end = max(window, int(0.45 * len(frame)))
    early = frame.iloc[:early_end]
    transformed = np.log1p(early.clip(lower=0.0))
    nonzero_rate = (early > 0.0).mean(axis=0)
    variability = transformed.std(axis=0)
    score = (nonzero_rate * variability).replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < n_stations:
        raise ValueError(f"{system} has only {len(score)} usable stations; requested {n_stations}")
    selected = list(score.sort_values(ascending=False).head(n_stations).index)
    X = np.log1p(frame[selected].clip(lower=0.0)).to_numpy(dtype=float)
    return rolling_gaussians_from_array(
        X,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=True,
        return_windows=return_windows,
    )


def parse_config_grid(text: str | None, *, quick: bool) -> list[tuple[int, int, int]]:
    if not text:
        return [(48, 12, 60)] if quick else [
            (24, 6, 40),
            (48, 12, 40),
            (48, 12, 60),
            (72, 12, 60),
            (72, 24, 80),
        ]
    configs: list[tuple[int, int, int]] = []
    for chunk in text.split(","):
        parts = [part.strip() for part in chunk.split(":")]
        if len(parts) != 3:
            raise ValueError(f"invalid config {chunk!r}; expected window:step:n_stations")
        window, step, n_stations = (int(part) for part in parts)
        configs.append((window, step, n_stations))
    return configs


def iter_bikeshare_jobs(
    system: str,
    months: tuple[str, ...],
    *,
    quick: bool,
    configs: list[tuple[int, int, int]] | None = None,
):
    configs = configs or parse_config_grid(None, quick=quick)
    for window, step, n_stations in configs:
        means, covs = bikeshare_gaussians(
            system,
            months,
            window=window,
            step=step,
            n_stations=n_stations,
        )
        yield (
            f"{system}_w{window}_s{step}_d{n_stations}",
            system,
            means,
            covs,
            {
                "window": int(window),
                "step": int(step),
                "n_stations": int(n_stations),
                "months": ",".join(months),
                "transform": "log1p hourly station trip counts",
            },
        )


def run_screen(
    *,
    systems: tuple[str, ...],
    months: tuple[str, ...],
    horizons: list[int],
    quick: bool,
    ar_model: str,
    reference_library: str,
    out_dir: Path,
    configs: list[tuple[int, int, int]] | None = None,
) -> dict[str, object]:
    set_reference_library_mode(reference_library)
    started = time.time()
    rows: list[pd.DataFrame] = []
    ref_tables: list[pd.DataFrame] = []
    tag = (
        f"bikeshare_{'-'.join(systems)}_{months[0]}_{months[-1]}_"
        f"h{'-'.join(map(str, horizons))}_{ar_model}"
        + ("_quick" if quick else "")
    )
    if reference_library != "full":
        tag += f"_{reference_library}"

    out_dir.mkdir(parents=True, exist_ok=True)
    for system in systems:
        for job, dataset, means, covs, meta in iter_bikeshare_jobs(system, months, quick=quick, configs=configs):
            print(f"Running {job} horizons={horizons} n={len(covs)} d={covs.shape[1] if len(covs) else 'NA'}")
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
                pd.concat(rows, ignore_index=True).to_csv(out_dir / f"raw_results_{tag}.csv", index=False)
                summary, summary_h = summarize_outputs(pd.concat(rows, ignore_index=True))
                summary.to_csv(out_dir / f"summary_by_dataset_{tag}.csv", index=False)
                summary_h.to_csv(out_dir / f"summary_by_dataset_horizon_{tag}.csv", index=False)
                print(summary.to_string(index=False))
            if not refs.empty:
                ref_tables.append(refs)
                pd.concat(ref_tables, ignore_index=True).to_csv(out_dir / f"reference_table_{tag}.csv", index=False)

    raw_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary, summary_h = summarize_outputs(raw_all) if not raw_all.empty else (pd.DataFrame(), pd.DataFrame())
    metadata = {
        "systems": systems,
        "months": months,
        "horizons": horizons,
        "quick": bool(quick),
        "ar_model": ar_model,
        "reference_library": reference_library,
        "configs": configs or parse_config_grid(None, quick=quick),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen additional real-data streams for BWAR.")
    parser.add_argument("--systems", default="divvy,capital_bikeshare")
    parser.add_argument("--months", default="202401,202402,202403,202404,202405,202406")
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--ar-model", choices=["diag", "full"], default="diag")
    parser.add_argument("--reference-library", choices=["full", "no_barycenter"], default="no_barycenter")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--configs", default="", help="Optional comma-separated window:step:n_stations triples.")
    parser.add_argument("--out-dir", type=Path, default=SCREEN_OUT)
    args = parser.parse_args()
    configs = parse_config_grid(args.configs, quick=bool(args.quick))
    metadata = run_screen(
        systems=tuple(x.strip() for x in args.systems.split(",") if x.strip()),
        months=tuple(x.strip() for x in args.months.split(",") if x.strip()),
        horizons=[int(x) for x in args.horizons.split(",") if x.strip()],
        quick=bool(args.quick),
        ar_model=args.ar_model,
        reference_library=args.reference_library,
        out_dir=args.out_dir,
        configs=configs,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
