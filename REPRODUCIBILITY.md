# Reproducibility map

Commands are run from the repository root after `make install`.

| Article item | Source results | Reproduction command | Generated file |
|---|---|---|---|
| Figure 3 | fixed simulation summary | `make figures` | `artifacts/generated/figures/synthetic_transport_mechanism.pdf` |
| Figure 4 | rolling-drift results | `make figures` | `artifacts/generated/figures/rolling_refit_reference_shift.pdf` |
| Figure 5 | Divvy summaries | `make figures` | `artifacts/generated/figures/redone_realdata_application.pdf` |
| Tables 2--3 | fixed simulation summary | `make figures` | `artifacts/generated/tables/synthetic_transport_*.tex` |
| Table 4 | archived frozen artifact | `make figures` | `artifacts/generated/tables/local_reference_shift_main.tex` |
| Table 5 | rolling-drift summary | `make figures` | `artifacts/generated/tables/rolling_refit_reference_shift.tex` |
| Tables 6--8 | Divvy summaries | `make figures` | `artifacts/generated/tables/redone_realdata_*.tex` |

## Full simulation rerun

```bash
make simulations
```

This runs 50 independent replications of the fixed-reference mechanism study
and 50 independent replications of the rolling-refit reference-drift study.
For a quick implementation check:

```bash
make smoke
```

The ten-replication frozen-chart stress test in Table 4 predates the final
replication pipeline. Its accepted table is archived with an explicit
provenance note in `results/reference/frozen_drift/`; the original raw
generator is not represented as available.

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
