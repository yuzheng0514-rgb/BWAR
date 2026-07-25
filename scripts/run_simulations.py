#!/usr/bin/env python3
"""Run the reproducible BWAR simulations reported in the article."""

from __future__ import annotations

import argparse
from pathlib import Path

from bwar.paper_jcgs import build_jcgs_simulation_artifacts as simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the transport-linear and rolling-refit reference-drift "
            "simulations."
        )
    )
    parser.add_argument("--fixed-reps", type=int, default=50)
    parser.add_argument("--drift-reps", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=simulation.ROOT / "results" / "generated",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=simulation.ROOT / "artifacts" / "generated",
    )
    args = parser.parse_args()

    simulation.FIXED_OUT_DIR = args.output_root / "fixed_simulation"
    simulation.DRIFT_OUT_DIR = args.output_root / "rolling_drift"
    simulation.build_all(
        fixed_reps=args.fixed_reps,
        drift_reps=args.drift_reps,
        target_dir=args.artifact_root,
    )
    print(f"Simulation results: {args.output_root}")
    print(f"Generated tables and figures: {args.artifact_root}")


if __name__ == "__main__":
    main()
