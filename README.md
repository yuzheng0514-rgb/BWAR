# BWAR

Replication code and numerical artifacts for **Bures--Wasserstein
Autoregression for Gaussian Distributional Time Series** by Yuzheng Dong,
Junlie Huang, and Cheng Meng.

BWAR represents a multivariate Gaussian time series in affine
Bures--Wasserstein transport coordinates, fits autoregressive dynamics in the
resulting Euclidean chart, and reconstructs Gaussian forecasts through the
inverse chart. The repository contains the fixed-reference and local-reference
implementations, comparator methods, simulations, Divvy rolling-origin
analysis, paired inference, and figure/table builders used for the article.

## Contents

- `src/bwar/`: validated numerical implementation.
- `scripts/`: clean public entry points for simulations, data retrieval,
  Divvy analysis, artifact generation, and verification.
- `configs/`: machine-readable records of the reported experimental settings.
- `results/reference/`: archived numerical outputs underlying the article.
- `artifacts/submitted/`: exact empirical figures and tables used in the
  submitted version.
- `tests/`: focused tests for geometry, local references, rolling evaluation,
  inference, and graphical output.
- `REPRODUCIBILITY.md`: command-to-figure/table map.

Raw Divvy trip archives, unrelated raw datasets, exploratory outputs,
development notes, and document build products are intentionally excluded.

## Installation

The archived results were produced with Python 3.13.9. Create an isolated
environment and install the locked dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The validated numerical stack is recorded in `requirements-lock.txt`.

## Quick verification

```bash
make test
make verify
```

`make verify` rebuilds every empirical table and figure from the archived
result files and checks the reported sample sizes and key numerical values.
A one-replication implementation smoke test is available through:

```bash
make smoke
```

## Reproduce the simulations

```bash
make simulations
```

This runs the 50-replication transport-linear simulation and the
50-replication rolling-refit reference-drift experiment. Outputs are written
under `results/generated/` and `artifacts/generated/`.

The earlier ten-replication frozen-chart stress-test table is retained as an
archived provenance artifact because its original raw generator is absent from
the final development archive. This limitation is stated explicitly rather
than replacing the original computation with a reconstructed approximation.

## Reproduce the Divvy analysis

The analysis uses the public January--December 2024 Divvy trip records:

```bash
PYTHONPATH=src python scripts/download_divvy.py
make divvy
make artifacts
```

The download script retrieves the official monthly archives and checks the
SHA-256 values recorded in `data/divvy_source_manifest.json`. Raw archives and
the generated hourly cache are ignored by Git. The full analysis may take
several hours depending on hardware.

See `data/README.md` for the source license and `REPRODUCIBILITY.md` for the
mapping from commands to article figures and tables.

## Reference outputs

The repository includes the archived simulation summaries, replication-level
losses, Divvy target-level loss panel, selected settings, bootstrap inference,
protocol manifest, and submitted empirical artifacts. These small files allow
the article to be audited without downloading third-party data or rerunning
the full experiments.

## Software and data availability wording

After a versioned Zenodo archive has been created, the article can state:

> The Python implementation of BWAR and the code reproducing the simulation
> studies, Divvy analysis, statistical inference, tables, and figures are
> available at <https://github.com/yuzheng0514-rgb/BWAR>. The version used for
> the article is archived at the associated Zenodo DOI under the MIT License.

The Divvy source data are available from
<https://divvybikes.com/system-data> and remain subject to the
[Divvy Data License Agreement](https://divvybikes.com/data-license-agreement).

## License and citation

The authors' code is released under the MIT License. Third-party data retain
their original terms. Citation metadata are provided in `CITATION.cff`; a
version-specific DOI will be added after archiving the GitHub release with
Zenodo.
