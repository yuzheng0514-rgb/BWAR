# Data

The real-data analysis uses the public 2024 Divvy trip archives. The raw
archives are not redistributed in this repository. Run

```bash
PYTHONPATH=src python scripts/download_divvy.py
```

to retrieve the twelve official monthly files and verify the SHA-256 values in
`divvy_source_manifest.json`.

The source records are governed by the
[Divvy Data License Agreement](https://divvybikes.com/data-license-agreement).
The repository contains selected station identifiers, preprocessing metadata,
target-level losses, and other derived numerical results needed to audit the
article, but not a stand-alone copy of the trip dataset.
