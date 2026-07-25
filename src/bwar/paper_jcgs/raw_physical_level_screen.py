from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bwar.paper_jcgs.real_bwar_theory_matched import (  # noqa: E402
    BEIJING_AIR_FEATURES,
    BEIJING_AIR_STATIONS,
    DATA,
    FINANCE_ETF_HD_SYMBOLS,
    HYDRAULIC_SENSOR_FILES,
    INTEL_LAB_MEASURES,
    _ensure_beijing_air_files,
    _ensure_hydraulic_archive,
    _ensure_mhealth_dir,
    download_url,
    fetch_yahoo_adjclose_cached,
    load_electricity_matrix,
    load_hai21_process_frame,
    load_intel_lab_sensor_frame,
    load_solar_energy_matrix,
    project_spd,
    rolling_gaussians_from_array,
    set_reference_library_mode,
)
from bwar.paper_jcgs.rolling_origin_backtest import (  # noqa: E402
    make_rolling_origin_splits,
    run_rolling_origin_series,
    summarize_rolling,
)
from bwar.paper_jcgs.strong_nyc_taxi_screen import cached_taxi_hourly_matrix  # noqa: E402
from bwar.paper_jcgs.strong_realdata_screen import load_bikeshare_hourly  # noqa: E402


OUT_DIR = DATA.parent / "paper_jcgs" / "outputs" / "real_bwar_theory_matched" / "raw_physical_level_screen"
TABLE_DIR = DATA.parent / "paper_jcgs" / "tables"


@dataclass(frozen=True)
class RawPhysicalCandidate:
    name: str
    label: str
    configs: tuple[tuple[int, int, int], ...]


CANDIDATES: dict[str, RawPhysicalCandidate] = {
    "household_power": RawPhysicalCandidate(
        name="household_power",
        label="household electric-load level",
        configs=((720, 180, 7), (960, 240, 7), (1440, 360, 7)),
    ),
    "appliances_energy": RawPhysicalCandidate(
        name="appliances_energy",
        label="appliances building-environment level",
        configs=((144, 24, 10), (288, 48, 10), (576, 96, 10)),
    ),
    "hai21": RawPhysicalCandidate(
        name="hai21",
        label="industrial process-variable level",
        configs=((600, 600, 30), (1200, 600, 30), (1800, 900, 30)),
    ),
    "intel_lab": RawPhysicalCandidate(
        name="intel_lab",
        label="environmental sensor level",
        configs=((72, 12, 10), (144, 24, 12), (216, 36, 15)),
    ),
    "sml2010": RawPhysicalCandidate(
        name="sml2010",
        label="building sensor level",
        configs=((48, 8, 21), (96, 16, 21), (192, 32, 21)),
    ),
    "solar_energy": RawPhysicalCandidate(
        name="solar_energy",
        label="solar generation level",
        configs=((144, 24, 30), (288, 48, 40), (576, 96, 40)),
    ),
    "electricity": RawPhysicalCandidate(
        name="electricity",
        label="electricity demand level",
        configs=((168, 24, 30), (336, 48, 40), (720, 72, 40)),
    ),
    "beijing_air": RawPhysicalCandidate(
        name="beijing_air",
        label="pollutant concentration level",
        configs=((168, 24, 36), (336, 48, 48), (720, 72, 48)),
    ),
    "occupancy_detection": RawPhysicalCandidate(
        name="occupancy_detection",
        label="room occupancy sensor level",
        configs=((144, 24, 5), (288, 48, 5), (576, 96, 5)),
    ),
    "room_occupancy": RawPhysicalCandidate(
        name="room_occupancy",
        label="room multi-sensor level",
        configs=((96, 24, 16), (192, 48, 16), (384, 96, 16)),
    ),
    "finance_etf": RawPhysicalCandidate(
        name="finance_etf",
        label="ETF return level",
        configs=((60, 5, 20), (120, 10, 20), (252, 21, 20)),
    ),
    "gas_drift": RawPhysicalCandidate(
        name="gas_drift",
        label="gas sensor response level",
        configs=((96, 24, 16), (192, 48, 16), (384, 96, 16)),
    ),
    "hydraulic": RawPhysicalCandidate(
        name="hydraulic",
        label="hydraulic sensor-cycle level",
        configs=((30, 5, 17), (60, 10, 17), (120, 20, 17)),
    ),
    "mhealth_s1": RawPhysicalCandidate(
        name="mhealth_s1",
        label="wearable motion-sensor level",
        configs=((200, 50, 21), (400, 100, 21), (800, 200, 21)),
    ),
    "mhealth_s2": RawPhysicalCandidate(
        name="mhealth_s2",
        label="wearable motion-sensor level",
        configs=((200, 50, 21), (400, 100, 21), (800, 200, 21)),
    ),
    "mhealth_s3": RawPhysicalCandidate(
        name="mhealth_s3",
        label="wearable motion-sensor level",
        configs=((200, 50, 21), (400, 100, 21), (800, 200, 21)),
    ),
    "melbourne_pedestrian": RawPhysicalCandidate(
        name="melbourne_pedestrian",
        label="pedestrian count level",
        configs=((24, 6, 40), (48, 12, 40), (168, 24, 40)),
    ),
    "capital_bikeshare": RawPhysicalCandidate(
        name="capital_bikeshare",
        label="bike-share station trip-count level",
        configs=((24, 6, 40), (48, 12, 40), (72, 24, 60)),
    ),
    "baywheels": RawPhysicalCandidate(
        name="baywheels",
        label="bike-share station trip-count level",
        configs=((24, 6, 40), (48, 12, 40), (72, 24, 60)),
    ),
    "bluebikes": RawPhysicalCandidate(
        name="bluebikes",
        label="bike-share station trip-count level",
        configs=((24, 6, 40), (48, 12, 40), (72, 24, 60)),
    ),
    "divvy": RawPhysicalCandidate(
        name="divvy",
        label="bike-share station trip-count level",
        configs=((24, 6, 40), (48, 12, 40), (72, 24, 60)),
    ),
    "nyc_taxi_green": RawPhysicalCandidate(
        name="nyc_taxi_green",
        label="taxi pickup-count level",
        configs=((24, 6, 15), (24, 6, 30), (48, 12, 40)),
    ),
    "nyc_taxi_yellow": RawPhysicalCandidate(
        name="nyc_taxi_yellow",
        label="taxi pickup-count level",
        configs=((24, 6, 30), (48, 12, 40), (72, 24, 40)),
    ),
    "mta_subway": RawPhysicalCandidate(
        name="mta_subway",
        label="subway station ridership level",
        configs=((24, 6, 40), (48, 12, 40), (168, 24, 40)),
    ),
    "citibike_jc": RawPhysicalCandidate(
        name="citibike_jc",
        label="bike-share station trip-count level",
        configs=((24, 6, 40), (48, 12, 40), (72, 24, 60)),
    ),
    "casey_pedestrian": RawPhysicalCandidate(
        name="casey_pedestrian",
        label="pedestrian count level",
        configs=((24, 6, 40), (48, 12, 40), (168, 24, 40)),
    ),
    "sydney_pedestrian": RawPhysicalCandidate(
        name="sydney_pedestrian",
        label="pedestrian count level",
        configs=((24, 6, 4), (48, 12, 4), (168, 24, 4)),
    ),
    "dublin_cycle_counts": RawPhysicalCandidate(
        name="dublin_cycle_counts",
        label="cycle-count level",
        configs=((24, 6, 25), (48, 12, 25), (168, 24, 25)),
    ),
}


def _initial_fit_raw_end(*, n_raw: int, window: int, step: int, max_matrices: int) -> int:
    starts = list(range(0, n_raw - window + 1, step))
    if len(starts) > max_matrices:
        idx = np.linspace(0, len(starts) - 1, max_matrices, dtype=int)
        starts = [starts[i] for i in idx]
    splits = make_rolling_origin_splits(len(starts), max_origins=1)
    if not splits:
        return min(n_raw, max(window, 1))
    fit_end = splits[0][0]
    return int(starts[fit_end - 1] + window)


def _standardized_stream_with_profile(
    X: np.ndarray,
    *,
    window: int,
    step: int,
    max_matrices: int,
    metric: str,
    label: str,
    fit_raw_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    X = np.asarray(X, dtype=float)
    X = X[:, np.nanstd(X, axis=0) > 1e-10]
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("raw physical matrix has no usable columns")
    X_filled = pd.DataFrame(X).interpolate(limit_direction="both").dropna(axis=0, how="any").to_numpy(float)
    scale_end = (
        _initial_fit_raw_end(
            n_raw=len(X_filled),
            window=window,
            step=step,
            max_matrices=max_matrices,
        )
        if fit_raw_end is None
        else min(len(X_filled), max(1, int(fit_raw_end)))
    )
    center = X_filled[:scale_end].mean(axis=0)
    scale = X_filled[:scale_end].std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    X_standardized = (X_filled - center) / scale
    means, covs, raw_windows, starts = rolling_gaussians_from_array(
        X_standardized,
        window=window,
        step=step,
        max_matrices=max_matrices,
        ridge=1e-5,
        standardize=False,
        return_windows=True,
    )
    profile = {
        "metric": metric,
        "label": label,
        "kind": "mean_rmse",
        "center": center[: means.shape[1]],
        "scale": scale[: means.shape[1]],
        "fit_raw_end": int(scale_end),
    }
    return means, np.asarray([project_spd(C, eps=1e-8) for C in covs]), raw_windows, starts, profile


def household_power_matrix() -> np.ndarray:
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
    return df[cols].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").dropna().to_numpy(float)


def appliances_energy_matrix() -> np.ndarray:
    target = DATA / "energydata_complete.csv"
    download_url(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv",
        target,
    )
    df = pd.read_csv(target)
    cols = ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T_out", "RH_out", "Windspeed", "Tdewpoint"]
    return df[cols].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").dropna().to_numpy(float)


def hai21_matrix(n_features: int) -> np.ndarray:
    df = load_hai21_process_frame()
    selected = list(df.std(axis=0).sort_values(ascending=False).head(n_features).index)
    return df[selected].to_numpy(float)


def intel_lab_matrix(n_motes: int) -> np.ndarray:
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
    selected_motes = [int(x) for x in rank.sort_values("score", ascending=False).head(n_motes)["mote"]]
    cols = [f"mote{mote}_{measure}" for mote in selected_motes for measure in INTEL_LAB_MEASURES]
    return df[cols].interpolate(limit_direction="both").dropna(axis=0, how="any").to_numpy(float)


def sml2010_matrix() -> np.ndarray:
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
            frame = frame.iloc[:, 2:23].copy()
            frame.columns = cols
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)


def _select_variable_indices(X: np.ndarray, n_features: int, *, early_end: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < n_features:
        raise ValueError(f"matrix has shape {X.shape}, cannot select {n_features} columns")
    if early_end is None:
        early_end = max(10, int(0.45 * len(X)))
    early_end = min(len(X), max(1, int(early_end)))
    early = X[:early_end]
    nonzero = np.nanmean(early > 0.0, axis=0)
    variability = np.nanstd(early, axis=0)
    scores = np.nan_to_num(nonzero * variability, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    return np.argsort(scores)[::-1][:n_features]


def _select_variable_columns(X: np.ndarray, n_features: int, *, early_end: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    selected = _select_variable_indices(X, n_features, early_end=early_end)
    return X[:, selected]


def solar_energy_matrix(n_sensors: int) -> np.ndarray:
    return _select_variable_columns(np.clip(load_solar_energy_matrix(), 0.0, None), n_sensors)


def electricity_matrix(n_clients: int) -> np.ndarray:
    return _select_variable_columns(np.clip(load_electricity_matrix(), 0.0, None), n_clients)


def beijing_air_matrix(max_features: int) -> np.ndarray:
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
        sub.index = dt
        sub.columns = [f"{station}_{col}" for col in BEIJING_AIR_FEATURES]
        series.append(sub)
    frame = pd.concat(series, axis=1).sort_index()
    frame = frame.interpolate(method="time", limit_direction="both").dropna(axis=1, how="all").dropna(axis=0, how="any")
    return _select_variable_columns(frame.clip(lower=0.0).to_numpy(float), max_features)


def occupancy_detection_matrix() -> np.ndarray:
    target = DATA / "occupancy_detection.zip"
    download_url("https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip", target)
    frames = []
    with zipfile.ZipFile(target) as zf:
        for split in ["datatraining.txt", "datatest.txt", "datatest2.txt"]:
            with zf.open(split) as fh:
                frames.append(pd.read_csv(fh))
    df = pd.concat(frames, ignore_index=True)
    cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    return df[cols].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").dropna().to_numpy(float)


def room_occupancy_matrix() -> np.ndarray:
    target = DATA / "room_occupancy_estimation.zip"
    download_url("https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip", target)
    with zipfile.ZipFile(target) as zf:
        with zf.open("Occupancy_Estimation.csv") as fh:
            df = pd.read_csv(fh)
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
    return df[cols].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").dropna().to_numpy(float)


def finance_etf_return_matrix(n_assets: int) -> np.ndarray:
    series = [fetch_yahoo_adjclose_cached(symbol) for symbol in FINANCE_ETF_HD_SYMBOLS]
    prices = pd.concat(series, axis=1, join="inner").sort_index()
    prices.columns = list(FINANCE_ETF_HD_SYMBOLS)
    returns = 100.0 * np.log(prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
    return _select_variable_columns(returns.to_numpy(float), n_assets)


def gas_drift_matrix(n_features: int) -> np.ndarray:
    target = DATA / "gas_sensor_drift.zip"
    download_url("https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip", target)
    rows = []
    with zipfile.ZipFile(target) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".dat")):
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
    if not rows:
        raise ValueError("gas drift matrix has no complete rows")
    return np.asarray(rows, dtype=float)


def hydraulic_cycle_matrix(n_features: int) -> np.ndarray:
    archive = _ensure_hydraulic_archive()
    channels = []
    with zipfile.ZipFile(archive) as zf:
        for name in HYDRAULIC_SENSOR_FILES[:n_features]:
            with zf.open(name) as fh:
                values = pd.read_csv(fh, sep="\t", header=None).to_numpy(float)
            channels.append(values.mean(axis=1))
    return np.column_stack(channels)


def mhealth_matrix(subject: int, n_features: int) -> np.ndarray:
    base = _ensure_mhealth_dir()
    path = base / f"mHealth_subject{subject}.log"
    if not path.exists():
        raise FileNotFoundError(f"MHEALTH subject file not found: {path}")
    arr = np.loadtxt(path)
    labels = arr[:, -1].astype(int)
    motion_cols = [0, 1, 2] + list(range(5, 23))
    X = arr[labels > 0][:, motion_cols].astype(float)
    if X.shape[1] > n_features:
        return _select_variable_columns(X, n_features)
    return X


def melbourne_pedestrian_matrix(n_sensors: int) -> np.ndarray:
    from bwar.paper_jcgs.real_bwar_theory_matched import load_melbourne_pedestrian_frame

    frame = load_melbourne_pedestrian_frame()
    return _select_variable_columns(frame.clip(lower=0.0).interpolate(limit_direction="both").dropna().to_numpy(float), n_sensors)


def bikeshare_matrix(
    system: str,
    n_stations: int,
    *,
    months: tuple[str, ...] = ("202401", "202402", "202403", "202404", "202405", "202406"),
    window: int | None = None,
    step: int | None = None,
    max_matrices: int | None = None,
    fit_raw_end: int | None = None,
    return_columns: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[str, ...]]:
    frame = load_bikeshare_hourly(system, months)
    early_end = fit_raw_end
    if early_end is None and window is not None and step is not None and max_matrices is not None:
        early_end = _initial_fit_raw_end(
            n_raw=len(frame),
            window=window,
            step=step,
            max_matrices=max_matrices,
        )
    X = frame.clip(lower=0.0).to_numpy(float)
    selected = _select_variable_indices(X, n_stations, early_end=early_end)
    matrix = X[:, selected]
    if return_columns:
        columns = tuple(str(frame.columns[i]) for i in selected)
        return matrix, columns
    return matrix


def taxi_green_matrix(n_zones: int) -> np.ndarray:
    matrix = cached_taxi_hourly_matrix("green", "2024-01")
    return _select_variable_columns(matrix.clip(lower=0.0).to_numpy(float), n_zones)


def taxi_yellow_matrix(n_zones: int) -> np.ndarray:
    cache = ROOT / "data" / "nyc_taxi" / "yellow_hourly_pickups_2024-01_clean.parquet"
    if cache.exists():
        matrix = pd.read_parquet(cache)
    else:
        trips = pd.read_parquet(
            ROOT / "data" / "nyc_taxi" / "yellow_tripdata_2024-01.parquet",
            columns=["tpep_pickup_datetime", "PULocationID"],
        )
        trips = trips.dropna(subset=["tpep_pickup_datetime", "PULocationID"]).copy()
        trips["tpep_pickup_datetime"] = pd.to_datetime(trips["tpep_pickup_datetime"], errors="coerce")
        start = pd.Timestamp("2024-01-01 00:00:00")
        stop = pd.Timestamp("2024-02-01 00:00:00")
        trips = trips[(trips["tpep_pickup_datetime"] >= start) & (trips["tpep_pickup_datetime"] < stop)]
        trips["hour"] = trips["tpep_pickup_datetime"].dt.floor("h")
        trips["zone"] = trips["PULocationID"].astype(int).astype(str)
        counts = trips.groupby(["hour", "zone"], observed=True).size().rename("pickups")
        matrix = counts.unstack(fill_value=0.0).sort_index().astype(float)
        matrix = matrix.reindex(pd.date_range(start, stop - pd.Timedelta(hours=1), freq="h"), fill_value=0.0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(cache)
    return _select_variable_columns(matrix.clip(lower=0.0).to_numpy(float), n_zones)


def parquet_matrix(relative_path: str, n_features: int) -> np.ndarray:
    frame = pd.read_parquet(ROOT / relative_path)
    numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction="both").dropna(axis=1, how="all").dropna(axis=0, how="any")
    return _select_variable_columns(numeric.clip(lower=0.0).to_numpy(float), n_features)


def dublin_cycle_counts_matrix(n_features: int) -> np.ndarray:
    frame = pd.read_csv(ROOT / "data" / "dublin_cycle_counts" / "cycle_counts_2024.csv")
    numeric = frame.drop(columns=["Time"], errors="ignore").replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction="both").dropna(axis=1, how="all").dropna(axis=0, how="any")
    return _select_variable_columns(numeric.clip(lower=0.0).to_numpy(float), n_features)


def candidate_matrix(name: str, dimension_arg: int) -> np.ndarray:
    if name == "household_power":
        return household_power_matrix()
    if name == "appliances_energy":
        return appliances_energy_matrix()
    if name == "hai21":
        return hai21_matrix(dimension_arg)
    if name == "intel_lab":
        return intel_lab_matrix(dimension_arg)
    if name == "sml2010":
        return sml2010_matrix()
    if name == "solar_energy":
        return solar_energy_matrix(dimension_arg)
    if name == "electricity":
        return electricity_matrix(dimension_arg)
    if name == "beijing_air":
        return beijing_air_matrix(dimension_arg)
    if name == "occupancy_detection":
        return occupancy_detection_matrix()
    if name == "room_occupancy":
        return room_occupancy_matrix()
    if name == "finance_etf":
        return finance_etf_return_matrix(dimension_arg)
    if name == "gas_drift":
        return gas_drift_matrix(dimension_arg)
    if name == "hydraulic":
        return hydraulic_cycle_matrix(dimension_arg)
    if name.startswith("mhealth_s"):
        return mhealth_matrix(int(name.rsplit("s", 1)[1]), dimension_arg)
    if name == "melbourne_pedestrian":
        return melbourne_pedestrian_matrix(dimension_arg)
    if name in {"capital_bikeshare", "baywheels", "bluebikes", "divvy"}:
        return bikeshare_matrix(name, dimension_arg)
    if name == "nyc_taxi_green":
        return taxi_green_matrix(dimension_arg)
    if name == "nyc_taxi_yellow":
        return taxi_yellow_matrix(dimension_arg)
    if name == "mta_subway":
        return parquet_matrix("data/mta_subway_hourly/hourly_matrix_20240101_20240701.parquet", dimension_arg)
    if name == "citibike_jc":
        return parquet_matrix("data/citibike_jc/hourly_202401_202402_202403_202404_202405_202406.parquet", dimension_arg)
    if name == "casey_pedestrian":
        return parquet_matrix("data/casey_pedestrian/hourly_matrix_2024.parquet", dimension_arg)
    if name == "sydney_pedestrian":
        return parquet_matrix("data/sydney_pedestrian/hourly_2024.parquet", dimension_arg)
    if name == "dublin_cycle_counts":
        return dublin_cycle_counts_matrix(dimension_arg)
    raise ValueError(f"unknown candidate: {name}")


def best_single_rows(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    rows = comp.sort_values("gain_vs_best_non_bwar", ascending=False).groupby("screen_candidate", as_index=False).head(1)
    keep = [
        "screen_candidate",
        "dataset",
        "domain_metric_label",
        "job",
        "origin",
        "horizon",
        "dimension",
        "window",
        "step",
        "selected_reference",
        "target_loss",
        "best_non_bwar_loss",
        "best_non_bwar_method",
        "gain_vs_best_non_bwar",
    ]
    return rows[[col for col in keep if col in rows.columns]].sort_values("gain_vs_best_non_bwar", ascending=False)


def method_loss_table(
    raw: pd.DataFrame,
    best: pd.DataFrame,
    *,
    loss_col: str = "test_domain_loss_mean",
    target_col_name: str = "Physical target",
    target_suffix: str | None = None,
) -> pd.DataFrame:
    display_names = {
        "nyc_taxi_green": "NYC Green Taxi",
        "household_power": "Household Power",
        "appliances_energy": "Appliances Energy",
        "hai21": "HAI 21.03",
        "intel_lab": "Intel Lab",
        "sml2010": "SML2010",
        "solar_energy": "Solar Energy",
        "electricity": "Electricity",
        "beijing_air": "Beijing Air",
        "occupancy_detection": "Occupancy Detection",
        "room_occupancy": "Room Occupancy",
        "finance_etf": "Finance ETF",
        "gas_drift": "Gas Sensor Drift",
        "hydraulic": "Hydraulic Systems",
        "mhealth_s1": "MHEALTH S1",
        "mhealth_s2": "MHEALTH S2",
        "mhealth_s3": "MHEALTH S3",
        "melbourne_pedestrian": "Melbourne Pedestrian",
        "capital_bikeshare": "Capital Bikeshare",
        "baywheels": "Bay Wheels",
        "bluebikes": "Bluebikes",
        "divvy": "Divvy",
    }
    target_names = {
        "nyc_taxi_green": "出租车上车量均值 RMSE",
        "household_power": "家庭用电负荷均值 RMSE",
        "appliances_energy": "建筑环境变量均值 RMSE",
        "hai21": "工业过程变量均值 RMSE",
        "intel_lab": "环境传感器均值 RMSE",
        "sml2010": "建筑传感器均值 RMSE",
        "solar_energy": "太阳能发电量均值 RMSE",
        "electricity": "电力需求均值 RMSE",
        "beijing_air": "污染物浓度均值 RMSE",
        "occupancy_detection": "占用传感器均值 RMSE",
        "room_occupancy": "房间传感器均值 RMSE",
        "finance_etf": "ETF return 原始均值 RMSE",
        "gas_drift": "气体传感器均值 RMSE",
        "hydraulic": "液压传感器均值 RMSE",
        "mhealth_s1": "运动传感器均值 RMSE",
        "mhealth_s2": "运动传感器均值 RMSE",
        "mhealth_s3": "运动传感器均值 RMSE",
        "melbourne_pedestrian": "行人计数均值 RMSE",
        "capital_bikeshare": "共享单车出行量均值 RMSE",
        "baywheels": "共享单车出行量均值 RMSE",
        "bluebikes": "共享单车出行量均值 RMSE",
        "divvy": "共享单车出行量均值 RMSE",
    }
    methods = [
        ("persistence", "Persistence"),
        ("raw_var_window_ar", "Raw VAR"),
        ("euclidean_gaussian_ar", "Euclidean AR"),
        ("cholesky_gaussian_ar", "Cholesky AR"),
        ("log_euclidean_gaussian_ar", "Log-Euclidean AR"),
        ("bwar_selected_ref", "BWAR-selected"),
    ]
    rows = []
    for _, case in best.iterrows():
        sub = raw[
            (raw["screen_candidate"] == case["screen_candidate"])
            & (raw["job"] == case["job"])
            & (raw["origin"] == case["origin"])
            & (raw["horizon"] == case["horizon"])
        ]
        row = {
            "Dataset": display_names.get(str(case["screen_candidate"]), str(case["screen_candidate"])),
            target_col_name: target_suffix
            or target_names.get(str(case["screen_candidate"]), case["domain_metric_label"]),
            "Setting": f"w={int(case['window'])}, step={int(case['step'])}, d={int(case['dimension'])}, "
            f"h={int(case['horizon'])}, origin={int(case['origin'])}",
            "Selected R": str(case["selected_reference"]),
        }
        for method, label in methods:
            vals = sub.loc[sub["method"] == method, loss_col]
            row[label] = float(vals.iloc[0]) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_beamer_table(table: pd.DataFrame, path: Path, *, target_col: str = "Physical target") -> None:
    def tex_escape(value: object) -> str:
        text = str(value)
        return (
            text.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
        )

    fmt = table.copy()
    for col in ["Persistence", "Raw VAR", "Euclidean AR", "Cholesky AR", "Log-Euclidean AR", "BWAR-selected"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"数据集 & 指标 & 设置 & Pers. & Raw VAR & Euc. & Chol. & LogEuc. & BWAR \\",
        r"\midrule",
    ]
    for _, r in fmt.iterrows():
        lines.append(
            f"{tex_escape(r['Dataset'])} & {tex_escape(r[target_col])} & {tex_escape(r['Setting'])} & {r['Persistence']} & "
            f"{r['Raw VAR']} & {r['Euclidean AR']} & {r['Cholesky AR']} & {r['Log-Euclidean AR']} & "
            f"\\textbf{{{r['BWAR-selected']}}} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    candidates: list[str],
    *,
    horizons: list[int],
    max_origins: int,
    reference_library: str,
    ar_model: str,
    out_dir: Path,
) -> None:
    set_reference_library_mode(reference_library)
    rows = []
    refs = []
    for name in candidates:
        candidate = CANDIDATES[name]
        for window, step, dimension_arg in candidate.configs:
            X = candidate_matrix(name, dimension_arg)
            means, covs, raw_windows, starts, profile = _standardized_stream_with_profile(
                X,
                window=window,
                step=step,
                max_matrices=900,
                metric=f"{name}_raw_physical_mean_rmse",
                label=f"{candidate.label} raw mean RMSE",
            )
            job = f"{name}_w{window}_s{step}_d{means.shape[1]}"
            print(f"[{name}] {job}: n={len(covs)} d={covs.shape[1]} horizons={horizons}", flush=True)
            raw, ref = run_rolling_origin_series(
                job=job,
                dataset=name,
                means=means,
                covs=covs,
                meta={"window": window, "step": step, "physical_units": "raw"},
                horizons=horizons,
                raw_windows=raw_windows,
                window_starts=starts,
                window_size=window,
                ar_model=ar_model,
                max_origins=max_origins,
                domain_profile_override=profile,
            )
            if not raw.empty:
                raw = raw.copy()
                raw["screen_candidate"] = name
                rows.append(raw)
            if not ref.empty:
                ref = ref.copy()
                ref["screen_candidate"] = name
                refs.append(ref)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{'-'.join(candidates)}_raw_physical_h{'-'.join(map(str, horizons))}_{ar_model}_orig{max_origins}_{reference_library}"
    raw_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    raw_all.to_csv(out_dir / f"raw_results_{tag}.csv", index=False)
    if refs:
        pd.concat(refs, ignore_index=True).to_csv(out_dir / f"reference_table_{tag}.csv", index=False)
    comp, summary, summary_h = summarize_rolling(raw_all)
    if not comp.empty and "screen_candidate" not in comp.columns:
        labels = raw_all[["job", "origin", "horizon", "screen_candidate"]].drop_duplicates()
        comp = comp.merge(labels, on=["job", "origin", "horizon"], how="left")
    comp.to_csv(out_dir / f"comparison_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"summary_by_dataset_{tag}.csv", index=False)
    summary_h.to_csv(out_dir / f"summary_by_dataset_horizon_{tag}.csv", index=False)
    best = best_single_rows(comp)
    best.to_csv(out_dir / f"best_single_{tag}.csv", index=False)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    method_table = method_loss_table(raw_all, best)
    method_table.to_csv(TABLE_DIR / "raw_physical_level_method_losses.csv", index=False)
    write_beamer_table(method_table, TABLE_DIR / "raw_physical_level_method_losses_beamer.tex")
    w2_method_table = method_loss_table(
        raw_all,
        best,
        loss_col="test_w2_mean",
        target_col_name="Distribution target",
        target_suffix="Gaussian W2^2 平均损失",
    )
    w2_method_table.to_csv(TABLE_DIR / "raw_physical_w2_method_losses.csv", index=False)
    write_beamer_table(
        w2_method_table,
        TABLE_DIR / "raw_physical_w2_method_losses_beamer.tex",
        target_col="Distribution target",
    )
    print(best.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen BWAR on direct raw physical mean prediction tasks.")
    parser.add_argument("--candidates", default="household_power,hai21,intel_lab,sml2010")
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--ar-model", choices=["diag", "full"], default="diag")
    parser.add_argument("--max-origins", type=int, default=3)
    parser.add_argument("--reference-library", choices=["full", "no_barycenter", "fast"], default="fast")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    started = time.time()
    names = [name.strip() for name in args.candidates.split(",") if name.strip()]
    missing = sorted(set(names) - set(CANDIDATES))
    if missing:
        raise ValueError(f"unknown candidates: {', '.join(missing)}")
    run(
        names,
        horizons=[int(x) for x in args.horizons.split(",") if x.strip()],
        max_origins=args.max_origins,
        reference_library=args.reference_library,
        ar_model=args.ar_model,
        out_dir=args.out_dir,
    )
    print(json.dumps({"candidates": names, "elapsed_seconds": round(time.time() - started, 3)}, indent=2))


if __name__ == "__main__":
    main()
