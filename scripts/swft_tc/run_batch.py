#!/usr/bin/env python3
"""Batch entry point for the SWIFT Town/Country address model.

This script is a *thin* command-line wrapper. It contains no business logic:
every decision about grouping, extraction, scoring, retraction, HITL routing
and reporting lives in ``models/swft_tc/src/`` and is imported from there, so
the batch run, the notebooks and the tests all exercise the same code.

Run it from the enterprise repository root::

    python scripts/swft_tc/run_batch.py --dry-run
    python scripts/swft_tc/run_batch.py --input models/swft_tc/data/sample_input.csv

Exit codes:
    0  the run completed and every unique address was extracted
    1  the run completed but at least one unique address failed (see the
       processing-errors CSV); output, metrics and reports are still written
    2  the run could not start (bad configuration, missing input or reference
       file, absent credentials without ``--dry-run``)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _repository_root() -> Path:
    """Locate the enterprise repository root from this file's location.

    Anchored on ``__file__``, never on the working directory, so the script
    behaves identically however it is invoked.
    """
    return Path(__file__).resolve().parents[2]


# The repository root is the single import anchor; this is the one bootstrap
# in the script, mirroring the one in each notebook. Everything below is a
# normal project import.
if __package__ in (None, ""):  # pragma: no cover - CLI bootstrap
    _root = str(_repository_root())
    if _root not in sys.path:
        sys.path.insert(0, _root)

from models.swft_tc.src import io as swift_io  # noqa: E402
from models.swft_tc.src import reporting  # noqa: E402
from models.swft_tc.src.cache import AddressCache  # noqa: E402
from models.swft_tc.src.gemini_client import build_client  # noqa: E402
from models.swft_tc.src.grouping import load_group_config  # noqa: E402
from models.swft_tc.src.pipeline import Phase1Pipeline  # noqa: E402
from models.swft_tc.src.reference_data import (  # noqa: E402
    TownCountryReferenceError,
    build_provider,
    build_town_country_provider,
    find_iso_provider,
)
from models.swft_tc.src.schemas import load_prompt_contract  # noqa: E402
from models.swft_tc.src.serialization import write_detailed_json  # noqa: E402
from models.swft_tc.src.settings import (  # noqa: E402
    MODEL_ROOT,
    AppConfig,
    credentials_available,
    dry_run_requested,
    load_config,
    resolve_model_name,
)

LOGGER = logging.getLogger("swft_tc.run_batch")

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_CANNOT_START = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_batch.py",
        description=(
            "Run the Town/Country address extraction pipeline over an input CSV "
            "and write the expanded output, processing errors, run metrics, "
            "detailed audit JSON and executive reports."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Configuration YAML. Relative paths resolve against the model root "
            "(models/swft_tc/). Defaults to config/config.yaml there, or to "
            "SWIFT_ADDRESS_CONFIG when set."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input CSV. Relative paths resolve against the model root. "
            "Defaults to data/sample_input.csv."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Destination for the expanded output CSV. Relative paths resolve "
            "against the model root. Defaults to processing.output_path."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use the offline extraction stub instead of the model backend. No "
            "credentials required and no request leaves the machine."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and do not write the extraction cache for this run.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip the executive report and chart artifacts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Reporting threshold override. Does not change scoring or HITL "
            "routing; it only re-cuts the report bands."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> AppConfig:
    overrides: dict[str, dict[str, object]] = {}
    if args.no_cache:
        overrides["processing"] = {"cache_enabled": False}
    return load_config(args.config, base_dir=MODEL_ROOT, overrides=overrides or None)


def _describe(path: Path) -> str:
    """Render a path relative to the repository root when it lives inside it."""
    root = _repository_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    try:
        config = _resolve_config(args)
        group_config = load_group_config(config.path(config.project.group_config_path))
        prompt = load_prompt_contract(
            config.path(config.project.prompt_path), config.project.prompt_version
        )
        reference_provider = build_provider(
            config.reference_data, base_dir=config.base_dir
        )
        iso_provider = find_iso_provider(reference_provider)
        town_country_provider = build_town_country_provider(
            config.reference_data, base_dir=config.base_dir
        )
    except (FileNotFoundError, TownCountryReferenceError, ValueError) as exc:
        LOGGER.error("cannot start: %s", exc)
        return EXIT_CANNOT_START

    dry_run = args.dry_run or dry_run_requested()
    if not dry_run and not credentials_available():
        LOGGER.error(
            "cannot start: no model credentials in the environment. Provide them "
            "through the approved secret-management mechanism, or pass --dry-run "
            "to use the offline stub."
        )
        return EXIT_CANNOT_START

    model_name = resolve_model_name(config)
    client = build_client(
        config.model,
        prompt,
        model=model_name,
        iso_provider=iso_provider,
        dry_run=dry_run,
    )

    input_path = config.path(args.input or "data/sample_input.csv")
    try:
        frame = swift_io.read_input_csv(
            input_path, record_id_column=config.project.record_id_column
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("cannot start: %s", exc)
        return EXIT_CANNOT_START

    cache = AddressCache(
        config.path(config.processing.cache_path),
        enabled=config.processing.cache_enabled,
    )
    pipeline = Phase1Pipeline(
        config,
        group_config,
        client=client,
        reference_provider=reference_provider,
        town_country_provider=town_country_provider,
        prompt=prompt,
        cache=cache,
        mode="dry_run" if dry_run else "live",
    )
    result = pipeline.run(frame)

    output_path = swift_io.write_output_csv(
        result.frame, config.path(args.output or config.processing.output_path)
    )
    errors_path = swift_io.write_errors_csv(
        result.errors, config.path(config.processing.errors_path)
    )
    metrics_path = swift_io.write_metrics_json(
        result.metrics, config.path(config.processing.metrics_path)
    )

    detail_path = None
    if config.processing.detailed_json_enabled:
        detail_path = write_detailed_json(
            result.frame,
            config.path(config.processing.detailed_json_path),
            config=config,
            group_config=group_config,
            decisions_by_address=result.decisions_by_address,
            iso_provider=pipeline.iso_provider,
            output_format=config.processing.detailed_json_format,
            include_empty_groups=config.processing.detailed_json_include_empty_groups,
        )

    report_paths: dict[str, Path] = {}
    if config.reporting.enabled and not args.no_reports:
        report = reporting.write_reports(
            result.metrics, result.instances, config, threshold=args.threshold
        )
        report_paths = dict(report.paths)

    print()
    print(f"mode          : {'DRY RUN (offline stub)' if dry_run else 'LIVE'}")
    print(f"model         : {model_name}")
    print(f"input         : {_describe(input_path)}  {frame.shape}")
    print(f"output        : {_describe(output_path)}  {result.frame.shape}")
    print(f"backend calls : {client.call_count}")
    print(f"errors        : {len(result.errors)} -> {_describe(errors_path)}")
    print(f"metrics       : {_describe(metrics_path)}")
    if detail_path is not None:
        print(f"detailed JSON : {_describe(Path(detail_path))}")
    for name, path in sorted(report_paths.items()):
        print(f"report/{name:<20} : {_describe(Path(path))}")

    if result.errors:
        LOGGER.warning(
            "%d unique address(es) failed extraction; see %s",
            len(result.errors),
            _describe(errors_path),
        )
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
