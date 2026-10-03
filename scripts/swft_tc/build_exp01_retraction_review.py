#!/usr/bin/env python3
"""Derive the EXP-01 retraction-review artifacts from the finished baseline run.

EXP-01 - Right-Most Town / All-Match Country Retraction Baseline
(slug ``exp01_rightmost_town_allmatch_country``). Town: one right-most verified
occurrence removed per group. Country: every verified occurrence/form removed.
Entity protection: none. This script changes nothing about that: it reads the
canonical output the run already produced and writes, next to it:

* ``<prefix>_retraction_review.csv``       - the compact per-record view
* ``<prefix>_town_in_line1_review.csv``    - records whose predicted Town occurs
                                             as a whole token/phrase in line 1
* ``<prefix>_diagnostics.json``            - the baseline counts

where ``<prefix>`` is the canonical output's stem without ``_output``, so the
Experiment 1 artifacts are named ``exp01_rightmost_town_allmatch_country_*``.

It makes **no model calls** and never modifies the canonical output: every
value is copied from it. The Town-in-line-1 selection uses the pipeline's own
token-safe matcher, so ``BOSTONIAN`` is never counted as a ``BOSTON`` and a
multi-token Town has to match as a whole phrase. Later experiments get their
own review tooling; Experiment 2 behaviour does not belong here.

Run from the repository root, after the pipeline::

    python scripts/swft_tc/build_exp01_retraction_review.py \\
        --config config/config_exp01_rightmost_town_allmatch_country.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # pragma: no cover - CLI bootstrap
    _root = str(Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

from models.swft_tc.src.retraction import token_phrase_matches  # noqa: E402
from models.swft_tc.src.schemas import NO_TOWN  # noqa: E402
from models.swft_tc.src.scoring import (  # noqa: E402
    HITL_AUTO_ACCEPT_CANDIDATE,
    HITL_PROCESSING_ERROR,
)
from models.swft_tc.src.settings import MODEL_ROOT, load_config  # noqa: E402

REVIEW_FIELDS = (
    "combined_address_cleaned",
    "predicted_town",
    "predicted_country",
    "predicted_town_probability",
    "predicted_country_probability",
    "predicted_town_exists",
    "predicted_country_exists",
    "combined_address_retracted",
    "combined_address_retracted_comments",
    "composite_weighted_score",
    "hitl_flag",
    "hitl_state",
    "hitl_state_reason",
)

LINE1_FIELDS = ("predicted_town", "predicted_country", "combined_address_retracted")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def _artifact_prefix(output_path: Path) -> str:
    stem = output_path.stem
    return stem[: -len("_output")] if stem.endswith("_output") else stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config",
        default="config/config_exp01_rightmost_town_allmatch_country.yaml",
        help="Runtime config of the finished run; relative to the model root.",
    )
    parser.add_argument("--group-id", default="1", help="Configured group to review.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, base_dir=MODEL_ROOT)
    group = str(args.group_id)

    canonical = config.path(config.processing.output_path)
    if not canonical.is_file():
        print(f"canonical output not found: {canonical}", file=sys.stderr)
        return 2

    frame = pd.read_csv(canonical, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    record_id = config.project.record_id_column
    col = {key: config.output.column_name(key, group) for key in (*REVIEW_FIELDS, "combined_address")}
    missing = [name for name in col.values() if name not in frame.columns]
    if missing:
        print(f"canonical output lacks expected group-{group} columns: {missing}", file=sys.stderr)
        return 2

    # Source fields are whatever the run's group config says they are.
    from models.swft_tc.src.grouping import load_group_config

    groups = load_group_config(config.path(config.project.group_config_path))
    source_fields = list(next(g for g in groups.groups if g.group_id == group).source_fields)

    prefix = _artifact_prefix(canonical)
    out_dir = canonical.parent

    # --- compact retraction review ----------------------------------------
    review = frame[[record_id, *source_fields, *(col[k] for k in REVIEW_FIELDS)]]
    review_path = out_dir / f"{prefix}_retraction_review.csv"
    review.to_csv(review_path, index=False, encoding="utf-8")

    # --- Town-in-line-1 review ----------------------------------------------
    line1 = source_fields[0]
    towns = frame[col["predicted_town"]].astype(str)

    def town_in_line1(row: pd.Series) -> bool:
        town = str(row[col["predicted_town"]]).strip()
        if not town or town == NO_TOWN:
            return False
        return bool(token_phrase_matches(str(row[line1]), town))

    in_line1_mask = frame.apply(town_in_line1, axis=1)
    in_line1 = frame.loc[in_line1_mask, [record_id, *source_fields, *(col[k] for k in LINE1_FIELDS)]]
    line1_path = out_dir / f"{prefix}_town_in_line1_review.csv"
    in_line1.to_csv(line1_path, index=False, encoding="utf-8")

    # --- diagnostics ----------------------------------------------------------
    cleaned = frame[col["combined_address_cleaned"]].astype(str)
    retracted = frame[col["combined_address_retracted"]].astype(str)
    changed = cleaned.ne(retracted)
    states = frame[col["hitl_state"]].astype(str)
    diagnostics = {
        "group_id": group,
        "canonical_output": str(canonical),
        "total_input_records": int(len(frame)),
        "town_explicit": int(_truthy(frame[col["predicted_town_exists"]]).sum()),
        "country_explicit": int(_truthy(frame[col["predicted_country_exists"]]).sum()),
        "retraction_changed_address": int(changed.sum()),
        "retraction_made_no_change": int((~changed).sum()),
        "hitl_count": int(_truthy(frame[col["hitl_flag"]]).sum()),
        "auto_accept_candidate_count": int(states.eq(HITL_AUTO_ACCEPT_CANDIDATE).sum()),
        "extraction_error_count": int(states.eq(HITL_PROCESSING_ERROR).sum()),
        "predicted_town_in_line1": int(in_line1_mask.sum()),
        "no_town_predicted": int(towns.str.strip().isin({"", NO_TOWN}).sum()),
        "hitl_state_counts": {k: int(v) for k, v in states.value_counts().sort_index().items()},
    }
    diag_path = out_dir / f"{prefix}_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))
    print()
    print(f"retraction review     : {review_path}  ({len(review)} rows)")
    print(f"town-in-line-1 review : {line1_path}  ({len(in_line1)} rows)")
    print(f"diagnostics           : {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
