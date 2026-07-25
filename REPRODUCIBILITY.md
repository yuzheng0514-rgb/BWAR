# Reproducibility map

All commands below are run from the repository root after `make install`.
Archived reference outputs are retained so that tables and figures can be
audited without rerunning the computationally intensive experiments.

| Article item | Source results | Reproduction command | Generated file |
|---|---|---|---|
| Figure 3 | `results/reference/fixed_simulation/` | `make artifacts` | `artifacts/generated/figures/synthetic_transport_mechanism.pdf` |
| Figure 4 | `results/reference/rolling_drift/` | `make artifacts` | `artifacts/generated/figures/rolling_refit_reference_shift.pdf` |
| Figure 5 | `results/reference/divvy/` | `make artifacts` | `artifacts/generated/figures/redone_realdata_application.pdf` |
| Tables 2--3 | fixed simulation summary | `make artifacts` | `artifacts/generated/tables/synthetic_transport_*.tex` |
| Table 4 | archived frozen artifact | `make artifacts` | `artifacts/generated/tables/local_reference_shift_main.tex` |
| Table 5 | rolling-drift summary | `make artifacts` | `artifacts/generated/tables/rolling_refit_reference_shift.tex` |
| Tables 6 and 8 | Divvy method summary | `make artifacts` | `artifacts/generated/tables/redone_realdata_{application,horizon}.tex` |
| Table 7 | Divvy paired inference | `make artifacts` | `artifacts/generated/tables/redone_realdata_inference.tex` |

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

## Verification

```bash
make verify
```

This regenerates all empirical tables and figures from the archived numerical
outputs, checks key sample sizes and headline values, and compares every
generated table byte-for-byte with the submitted artifact.
