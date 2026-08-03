# Reproducibility map

Commands are run from the repository root after `make install`.

## Article-item map

| Article item | Compact source | Rebuild command | Generated artifact |
|---|---|---|---|
| Table 1: fixed-reference geometry study | `results/reference/s1_geometry/` | `make figures` | `artifacts/generated/tables/geometry_robustness_rebuild.tex` |
| Figure 3: matched-start reference adaptation | `results/reference/s2_reference_adaptation/` | `make figures` | `artifacts/generated/figures/s2_matched_start_loss.pdf` |
| Table 2: Divvy rolling-origin results | `results/reference/divvy/` | `make figures` | `artifacts/generated/tables/divvy_full_results_rebuild.tex` |
| Table 3: PEMS-BAY rolling-origin results | `results/reference/pems_bay/` | `make figures` | `artifacts/generated/tables/pems_full_results.tex` |

Frozen manuscript-ready versions are under `artifacts/reference/`. Run

```bash
make figures
make test
```

to rebuild the artifacts and compare the generated tables with the frozen
versions.

## S1: fixed-reference geometry study

```bash
make s1
```

The study uses 80 replications per setting, one-step forecasts, a
45%--20%--35% chronological split, the ridge grid
`1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10`, and an unrestricted centered ridge
lag-design VAR(1). The baseline mechanisms use `T=320`, `d=8`, and
`phi=0.70`; the table also reports the prespecified Bures finite-sample
variations.

## S2: matched-start reference adaptation

```bash
make s2
```

The frozen configuration is `configs/matched_reference_drift.json`. It uses
100 paired replications, `T=260`, `d=5`, fit/validation boundaries 117/169, a
36-observation rolling lag-design window, horizons 1, 3, and 6, and displacement
grid `0, 0.6, 1.2, 1.8, 2.4`. The primary local fit uses the ridge selected for
fixed BWAR; only the reference and re-encoding change in the primary contrast.

## Divvy

```bash
PYTHONPATH=src python scripts/download_divvy.py
make divvy
```

The analysis constructs 364 Gaussian states from 72-hour windows advanced by
24 hours, selects 30 stations using only the initial fitting data, and evaluates
horizons 3, 4, and 5 on 183 common targets per horizon. The local window grid is
`24, 48, 72`, its refresh period is 1, and selection uses validation raw RMSE.
The primary moving-block length is 3, the sensitivity length is 6, and the
bootstrap uses 10,000 resamples.

The compact reference directory contains only the seven methods reported in
the article.

## PEMS-BAY

```bash
make pems
```

The raw inputs are `data/pems_bay/pems-bay.h5` and
`data/pems_bay/graph_sensor_locations_bay.csv`. The analysis forms four
training-defined 20-sensor panels and 360 nonoverlapping 12-hour states per
panel. The first 216 states fit the models, the next 72 select the ridge, and
the remainder are held out. All coordinate methods use a diagonal ridge VAR(1)
with grid `1e-4, 1e-3, 1e-2, 1e-1, 1, 10`. Moving-block intervals use circular
blocks of length 7 and 2,000 draws.

## Data and versioning

Raw Divvy and PEMS-BAY data are not committed. Compact derived results,
protocol manifests, selected settings, and paired effects are versioned. The
release tag and commit hash identify the exact code/results snapshot used by
the article.
