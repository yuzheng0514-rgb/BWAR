#!/usr/bin/env python3
"""Run and persist the matched-start S2 reference-adaptation study."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bwar.paper_jcgs.matched_reference_drift import (  # noqa: E402
    MatchedReferenceConfig,
    PATH_REGIMES,
    REGIME_LABELS,
    make_figure,
    run_replication,
    summarize_performance,
    validate_results,
)


def _worker(job):
    return run_replication(*job)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _run_jobs(reps: int, workers: int, config: MatchedReferenceConfig):
    jobs = [(replication, config) for replication in range(reps)]
    performance, origins, references = [], [], []
    if workers <= 1:
        iterator = map(_worker, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_worker, jobs)
    try:
        for completed, result in enumerate(iterator, start=1):
            performance.append(result[0])
            origins.append(result[1])
            references.append(result[2])
            if completed == 1 or completed % max(1, reps // 10) == 0 or completed == reps:
                print(f"[matched-s2] completed {completed}/{reps}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    return (
        pd.concat(performance, ignore_index=True),
        pd.concat(origins, ignore_index=True),
        pd.concat(references, ignore_index=True),
    )


def _write_report(path, *, config, reps, workers, elapsed, summary, validation, figure):
    primary = summary[summary.method.isin(("fixed", "local_shared"))]
    lines = [
        "# S2 matched-start reference-adaptation run report",
        "",
        f"- UTC completion: {datetime.now(timezone.utc).isoformat()}",
        f"- Replications: {reps}",
        f"- Workers: {workers}",
        f"- Elapsed seconds: {elapsed:.2f}",
        f"- Estimator: centered full ridge VAR(1)",
        f"- Dimensions/time points: d={config.d}, T={config.n}",
        f"- Fit/validation/test boundaries: {config.fit_end}/{config.val_end}/{config.n}",
        f"- Rolling window: {config.window_length}; refresh period: {config.refresh_period}",
        f"- Target displacement grid at first test origin: {list(config.target_deltas)}",
        f"- Continuing increment during test: {config.continuation_increment}",
        f"- Numerical gate status: {'PASS' if validation['passed'] else 'FAIL'}",
        "",
        "## Numerical gates",
        "",
    ]
    for gate, passed in validation["numerical_gates"].items():
        lines.append(f"- {gate}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Primary direct Gaussian $W_2^2$ losses", "", "Entries are mean test losses with 95% Monte Carlo intervals; lower values are better.", ""])
    for horizon in config.horizons:
        lines.append(f"### h={horizon}")
        for regime in PATH_REGIMES:
            cell = primary[(primary.regime.eq(regime)) & (primary.horizon.eq(horizon))].sort_values(["target_delta", "method"])
            text = ", ".join(
                f"Delta={row.target_delta:g}, {row.method}: {row.test_w2_mean:.4f} [{row.test_w2_ci_low:.4f},{row.test_w2_ci_high:.4f}]"
                for row in cell.itertuples()
            )
            lines.append(f"- {REGIME_LABELS[regime]}: {text}")
        lines.append("")
    lines.extend(["## Figure exports", ""])
    for extension, figure_path in figure["outputs"].items():
        lines.append(f"- {extension.upper()}: `{figure_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "matched_reference_drift.json")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tag", default="matched_start_full")
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="Recompute validation and figure from an existing run.",
    )
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = MatchedReferenceConfig.from_mapping(payload["design"])
    reps = int(payload.get("replications", 100) if args.reps is None else args.reps)
    if reps < 2:
        raise ValueError("at least two replications are required")
    workers = max(1, int(args.workers))
    result_dir = ROOT / "results" / "generated" / "matched_reference_drift" / args.tag
    artifact_dir = ROOT / "artifacts" / "generated" / "matched_reference_drift" / args.tag
    report_dir = result_dir
    for directory in (result_dir, artifact_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.audit_existing:
        raw = pd.read_csv(result_dir / "replication_results.csv.gz")
        origins = pd.read_csv(result_dir / "origin_results.csv.gz")
        references = pd.read_csv(result_dir / "reference_diagnostics.csv.gz")
        # Recompute the summary so newly added uncertainty columns and the
        # direct-loss figure always reflect the persisted replication rows.
        summary = summarize_performance(raw)
        summary.to_csv(result_dir / "summary.csv", index=False)
        validation = validate_results(raw, origins, references, summary, config)
        (result_dir / "validation.json").write_text(
            json.dumps(validation, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        figure = make_figure(
            summary,
            artifact_dir / "s2_matched_start_loss",
            reps=int(raw.replication.nunique()),
        )
        report_path = report_dir / "run_report.md"
        _write_report(
            report_path,
            config=config,
            reps=int(raw.replication.nunique()),
            workers=0,
            elapsed=0.0,
            summary=summary,
            validation=validation,
            figure=figure,
        )
        manifest_path = report_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest["validation"] = validation
        manifest["figure"] = figure
        manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(f"[matched-s2] audited numerical={'PASS' if validation['passed'] else 'FAIL'}", flush=True)
        print(f"[matched-s2] report={report_path}", flush=True)
        return 0 if validation["passed"] else 2

    if any(result_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {result_dir}")

    started = time.perf_counter()
    raw, origins, references = _run_jobs(reps, workers, config)
    summary = summarize_performance(raw)
    validation = validate_results(raw, origins, references, summary, config)
    raw.to_csv(result_dir / "replication_results.csv.gz", index=False, compression="gzip")
    origins.to_csv(result_dir / "origin_results.csv.gz", index=False, compression="gzip")
    references.to_csv(result_dir / "reference_diagnostics.csv.gz", index=False, compression="gzip")
    summary.to_csv(result_dir / "summary.csv", index=False)
    (result_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=_json_default) + "\n", encoding="utf-8")
    figure = make_figure(summary, artifact_dir / "s2_matched_start_loss", reps=reps)
    elapsed = time.perf_counter() - started
    report_path = report_dir / "run_report.md"
    _write_report(report_path, config=config, reps=reps, workers=workers, elapsed=elapsed, summary=summary, validation=validation, figure=figure)
    manifest = {
        "experiment": "S2 matched-start reference adaptation",
        "tag": args.tag,
        "replications": reps,
        "workers": workers,
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "design": asdict(config),
        "elapsed_seconds": elapsed,
        "validation": validation,
        "result_dir": str(result_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "report_path": str(report_path.resolve()),
        "figure": figure,
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"[matched-s2] numerical={'PASS' if validation['passed'] else 'FAIL'}", flush=True)
    print(f"[matched-s2] report={report_path}", flush=True)
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
