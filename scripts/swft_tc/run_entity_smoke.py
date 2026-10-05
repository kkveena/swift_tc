#!/usr/bin/env python3
"""Live entity-identification smoke test on a handful of records (EXP-03).

Runs the ordinary batch pipeline over a FILTERED copy of the experiment input
(a few RECORD_IDs), with the experiment's own caches, and then checks the
run against the expectations a healthy EXP-03 run must meet BEFORE the full
115-record experiment is attempted again:

* Town/Country backend calls == 0 (every address replays from the seeded
  extraction cache, so the Town/Country prompt is never exercised);
* entity backend calls == entity cache misses (one successful call per
  uncached payload: no malformed re-asks, no truncation);
* entity errors == 0, and every non-empty group in the detailed JSON carries
  a parsed, schema-valid entity block;
* the entity cache grew by exactly the number of backend calls (existing
  entries preserved, every new answer persisted).

Only the run artifacts (output CSV, detailed JSON, metrics, errors, reports)
are redirected into a ``smoke/`` folder next to the experiment's outputs. The
extraction cache and the entity cache are the experiment's real files, so a
passing smoke leaves its answers cached for the full run. Nothing is logged
or printed here beyond record ids, counts and the pipeline's own
hash-only diagnostics.

Run from the repository root (credentials via the environment, as for
run_batch.py)::

    python scripts/swft_tc/run_entity_smoke.py \\
        --config config/config_exp03_entity_span_protected_confidence_gated.yaml \\
        --input  data/exp01_rightmost_town_allmatch_country_addresses.csv \\
        --record-ids 2152,1578,1450,1212,1062

Exit codes: 0 every expectation held; 1 the run completed but an expectation
failed; 2 the smoke could not start.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __package__ in (None, ""):  # pragma: no cover - CLI bootstrap
    _root = str(_repository_root())
    if _root not in sys.path:
        sys.path.insert(0, _root)

import pandas as pd  # noqa: E402

from models.swft_tc.src.cache import AddressCache  # noqa: E402
from models.swft_tc.src.entity_detection import ENTITY_TYPES  # noqa: E402
from models.swft_tc.src.settings import MODEL_ROOT, load_config  # noqa: E402
from scripts.swft_tc import run_batch  # noqa: E402

LOGGER = logging.getLogger("swft_tc.run_entity_smoke")

DEFAULT_CONFIG = "config/config_exp03_entity_span_protected_confidence_gated.yaml"
DEFAULT_INPUT = "data/exp01_rightmost_town_allmatch_country_addresses.csv"
DEFAULT_RECORD_IDS = "2152,1578,1450,1212,1062"

EXIT_OK = 0
EXIT_EXPECTATION_FAILED = 1
EXIT_CANNOT_START = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_entity_smoke.py",
        description="Run the EXP-03 pipeline over a few records and check the entity call path.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Experiment config YAML (relative to models/swft_tc/).")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Full experiment input CSV (relative to models/swft_tc/).")
    parser.add_argument("--record-ids", default=DEFAULT_RECORD_IDS,
                        help="Comma-separated RECORD_IDs to run (default: five high-value records).")
    parser.add_argument("--smoke-dir", default=None,
                        help="Where run artifacts go. Default: <experiment outputs dir>/smoke.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Offline stubs instead of the model backend (checks the plumbing only).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _filter_input(input_path: Path, record_column: str, record_ids: list[str], target: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if record_column not in frame.columns:
        raise ValueError(f"input has no {record_column!r} column")
    wanted = set(record_ids)
    subset = frame[frame[record_column].isin(wanted)]
    missing = sorted(wanted - set(subset[record_column]))
    if missing:
        raise ValueError(f"RECORD_ID(s) not found in input: {', '.join(missing)}")
    # Keep the caller's order so the smoke reads like its command line.
    subset = subset.set_index(record_column).loc[record_ids].reset_index()
    target.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(target, index=False, encoding="utf-8")
    return subset


def _derive_smoke_config(config_path: Path, smoke_dir: Path, target: Path) -> dict:
    """Copy the experiment config, redirecting run artifacts (NOT caches) into smoke_dir."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    processing = raw.setdefault("processing", {})
    processing["output_path"] = str(smoke_dir / "smoke_output.csv")
    processing["errors_path"] = str(smoke_dir / "errors" / "smoke_processing_errors.csv")
    processing["metrics_path"] = str(smoke_dir / "smoke_run_metrics.json")
    processing["detailed_json_path"] = str(smoke_dir / "smoke_detailed_output.jsonl")
    processing["detailed_json_format"] = "jsonl"
    processing["detailed_json_enabled"] = True
    reporting = raw.setdefault("reporting", {})
    reporting["reports_dir"] = str(smoke_dir / "reports")
    reporting["charts_dir"] = str(smoke_dir / "charts")
    target.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return raw


def _count_cache_entries(path: Path) -> int:
    return AddressCache(path).load() if path.exists() else 0


def _check(name: str, ok: bool, detail: str, results: list[tuple[str, bool, str]]) -> None:
    results.append((name, ok, detail))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-7s %(name)s: %(message)s", stream=sys.stdout, force=True)

    try:
        config = load_config(args.config, base_dir=MODEL_ROOT)
    except Exception as exc:  # noqa: BLE001 - startup diagnostics
        LOGGER.error("cannot start: %s", exc)
        return EXIT_CANNOT_START
    if not config.entity_detection.enabled:
        LOGGER.error("cannot start: entity_detection.enabled is false in %s", args.config)
        return EXIT_CANNOT_START

    config_path = config.path(args.config)
    input_path = config.path(args.input)
    tc_cache_path = config.path(config.processing.cache_path)
    entity_cache_path = config.path(config.entity_detection.cache_path)
    smoke_dir = Path(args.smoke_dir) if args.smoke_dir else tc_cache_path.parent / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    record_ids = [r.strip() for r in args.record_ids.split(",") if r.strip()]
    if not record_ids:
        LOGGER.error("cannot start: no record ids given")
        return EXIT_CANNOT_START

    try:
        subset = _filter_input(input_path, config.project.record_id_column, record_ids,
                               smoke_dir / "smoke_input.csv")
        _derive_smoke_config(config_path, smoke_dir, smoke_dir / "config_smoke.yaml")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOGGER.error("cannot start: %s", exc)
        return EXIT_CANNOT_START

    tc_entries_before = _count_cache_entries(tc_cache_path)
    entity_entries_before = _count_cache_entries(entity_cache_path)
    if tc_entries_before == 0 and not args.dry_run:
        LOGGER.error(
            "cannot start: the Town/Country extraction cache %s is empty or missing; seed it from "
            "the EXP-01 run first so that Town/Country backend calls can be 0", tc_cache_path,
        )
        return EXIT_CANNOT_START

    print(f"smoke records           : {len(subset)}  {record_ids}")
    print(f"Town/Country cache      : {tc_entries_before} entries  ({tc_cache_path.name})")
    print(f"entity cache (before)   : {entity_entries_before} entries  ({entity_cache_path.name})")
    print(f"entity max_output_tokens: {config.entity_detection.max_output_tokens}")
    print(f"smoke artifacts         : {smoke_dir}")
    print()

    batch_args = ["--config", str(smoke_dir / "config_smoke.yaml"),
                  "--input", str(smoke_dir / "smoke_input.csv"),
                  "--no-reports", "--log-level", args.log_level]
    if args.dry_run:
        batch_args.append("--dry-run")
    batch_exit = run_batch.main(batch_args)
    print()

    metrics_path = smoke_dir / "smoke_run_metrics.json"
    detail_path = smoke_dir / "smoke_detailed_output.jsonl"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        records = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        LOGGER.error("run produced no readable metrics/detail: %s", exc)
        return EXIT_EXPECTATION_FAILED

    eff = metrics.get("efficiency", {})
    tc_calls = int(eff.get("backend_calls", -1))
    entity_calls = int(eff.get("entity_backend_calls", -1))
    entity_hits = int(eff.get("entity_cache_hits", -1))
    entity_misses = int(eff.get("entity_cache_misses", -1))
    entity_errors = int(eff.get("entity_errors", -1))
    entity_entries_after = _count_cache_entries(entity_cache_path)

    statuses: dict[str, int] = {}
    unparsed: list[str] = []
    for record in records:
        for group_id, group in record.get("groups", {}).items():
            block = group.get("entity_detection")
            if block is None:
                continue  # empty group: entity detection does not run
            status = block.get("status", "missing")
            statuses[status] = statuses.get(status, 0) + 1
            valid = status in {"detected", "no_entity"} and block.get("entity_type") in ENTITY_TYPES
            if not valid:
                unparsed.append(f"{record.get('record_id')}/{group_id}:{status}")

    results: list[tuple[str, bool, str]] = []
    _check("run exit code", batch_exit == run_batch.EXIT_OK, f"run_batch exit {batch_exit}", results)
    if args.dry_run:
        _check("Town/Country backend calls", True,
               f"{tc_calls} (not asserted in --dry-run: the stub model name never matches the live cache)", results)
    else:
        _check("Town/Country backend calls == 0", tc_calls == 0, f"backend_calls={tc_calls}", results)
    _check("entity errors == 0", entity_errors == 0, f"entity_errors={entity_errors}", results)
    _check("entity calls == entity cache misses", entity_calls == entity_misses and entity_calls >= 0,
           f"entity_backend_calls={entity_calls} cache_hits={entity_hits} cache_misses={entity_misses}", results)
    _check("every entity response parsed and validated", not unparsed and bool(statuses),
           f"statuses={statuses}" + (f" unparsed={unparsed}" if unparsed else ""), results)
    _check("entity cache preserved and grew by the calls made",
           entity_entries_after == entity_entries_before + entity_calls,
           f"entries before={entity_entries_before} after={entity_entries_after} calls={entity_calls}", results)
    _check("Town/Country cache untouched",
           _count_cache_entries(tc_cache_path) == tc_entries_before or args.dry_run,
           f"entries before={tc_entries_before} after={_count_cache_entries(tc_cache_path)}", results)

    print("entity smoke checks")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
    passed = all(ok for _, ok, _ in results)
    print()
    print("ENTITY SMOKE: " + ("PASS — safe to run the full experiment" if passed
                              else "FAIL — do not run the full experiment; see diagnostics above"))
    return EXIT_OK if passed else EXIT_EXPECTATION_FAILED


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
