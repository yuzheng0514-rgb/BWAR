# Reproducibility map

Commands are run from the repository root after `make install`.

| Article item | Source results | Reproduction command | Generated file |
|---|---|---|---|
| Figure 3 | fixed simulation summary | `make figures` | `artifacts/generated/figures/synthetic_transport_mechanism.pdf` |
| Figure 4 | rolling-drift results | `make figures` | `artifacts/generated/figures/rolling_refit_reference_shift.pdf` |
| Figure 5 | descriptive Divvy structure | `make data-figure` | `artifacts/generated/figures/divvy_data_background.pdf` |
| Figure 6 | Divvy forecast summaries | `make figures` | `artifacts/generated/figures/redone_realdata_application.pdf` |
| Tables 2--3 | fixed simulation summary | `make figures` | `artifacts/generated/tables/synthetic_transport_*.tex` |
| Table 4 | rolling-drift summary | `make figures` | `artifacts/generated/tables/rolling_refit_reference_shift.tex` |
| Tables 5--6 | Divvy summaries | `make figures` | `artifacts/generated/tables/redone_realdata_*.tex` |

## Full simulation rerun

```bash
make simulations
```

This runs 50 independent replications of the fixed-reference Bures,
Log-Euclidean, and Cholesky generating mechanisms, 50 independent replications
of the rolling-refit reference-drift study, and its prespecified gradual-drift
strength grid.
For a quick implementation check:

```bash
make smoke
```

## Full Divvy rerun

```bash
PYTHONPATH=src python scripts/download_divvy.py
make divvy
```

The first command downloads approximately 222 MB of official source archives
and verifies their SHA-256 values. The analysis then constructs hourly station
counts, performs training-only station selection and scaling, runs all
rolling-origin forecasts, and applies the origin-preserving moving-block
bootstrap with 10,000 replicates.
