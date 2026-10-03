#!/usr/bin/env python3
"""Derive the EXP-02 retraction-review artifacts from the finished challenger run.

EXP-02 - Line-1 Protected / Confidence-Gated Retraction
(slug ``exp02_line1_protected_confidence_gated``). Reads the canonical output
and detailed JSONL that run already produced and writes, next to the output:

* ``<prefix>_retraction_review.csv``       - the compact per-record view
* ``<prefix>_town_in_line1_review.csv``    - records whose predicted Town occurs
                                             as a whole token/phrase in line 1
* ``<prefix>_diagnostics.json``            - the counts, including the policy
                                             audit: line-1 protected, Town
                                             gated, Country gated, and - when the
                                             EXP-01 output is present - how many
                                             retracted addresses changed vs EXP-01

It makes **no model calls** and never modifies either experiment's output. The
EXP-01 builder is untouched; this one reads the extra audit fields EXP-02
writes into the detailed JSON retraction block.

Run from the repository root, after the pipeline::

    python scripts/swft_tc/build_exp02_retraction_review.py \\
        --config config/config_exp02_line1_protected_confidence_gated.yaml
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

from models.swft_tc.src.grouping import load_group_config  # noqa: E402
from models.swft_tc.src.retraction import (  # noqa: E402
    SKIP_PROBABILITY,
    SKIP_PROTECTED_ONLY,
    token_phrase_matches,
)
from models.swft_tc.src.schemas import NO_TOWN  # noqa: E402
from models.swft_tc.src.scoring import (  # noqa: E402
    HITL_AUTO_ACCEPT_CANDIDATE,
    HITL_PROCESSING_ERROR,
)
from models.swft_tc.src.serialization import read_detailed_jsonl  # noqa: E402
from models.swft_tc.src.settings import MODEL_ROOT, load_config  # noqa: E402

REVIEW_FIELDS = (
    "combined_address_cleaned", "predicted_town", "predicted_country",
    "predicted_town_probability", "predicted_country_probability",
    "predicted_town_exists", "predicted_country_exists",
    "combined_address_retracted", "combined_address_retracted_comments",
    "composite_weighted_score", "hitl_flag", "hitl_state", "hitl_state_reason",
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
        default="config/config_exp02_line1_protected_confidence_gated.yaml",
        help="Runtime config of the finished EXP-02 run; relative to the model root.",
    )
    parser.add_argument(
        "--baseline-config",
        default="config/config_exp01_rightmost_town_allmatch_country.yaml",
        help="EXP-01 config, used only to locate its output for the changed-vs-EXP01 count.",
    )
    parser.add_argument("--group-id", default="1", help="Configured group to review.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, base_dir=MODEL_ROOT)
    group = str(args.group_id)

    canonical = config.path(config.processing.output_path)
    detailed = config.path(config.processing.detailed_json_path)
    if not canonical.is_file():
        print(f"canonical output not found: {canonical}", file=sys.stderr)
        return 2

    frame = pd.read_csv(canonical, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    record_id = config.project.record_id_column
    col = {key: config.output.column_name(key, group) for key in REVIEW_FIELDS}
    missing = [name for name in col.values() if name not in frame.columns]
    if missing:
        print(f"canonical output lacks expected group-{group} columns: {missing}", file=sys.stderr)
        return 2

    groups = load_group_config(config.path(config.project.group_config_path))
    source_fields = list(next(g for g in groups.groups if g.group_id == group).source_fields)
    prefix = _artifact_prefix(canonical)
    out_dir = canonical.parent

    # --- compact retraction review ----------------------------------------
    review = frame[[record_id, *source_fields, *(col[k] for k in REVIEW_FIELDS)]]
    review_path = out_dir / f"{prefix}_retraction_review.csv"
    review.to_csv(review_path, index=False, encoding="utf-8")

    # --- Town-in-line-1 review (EXP-02: these must all be preserved) --------
    line1 = source_fields[0]

    def town_in_line1(row: pd.Series) -> bool:
        town = str(row[col["predicted_town"]]).strip()
        return bool(town) and town != NO_TOWN and bool(token_phrase_matches(str(row[line1]), town))

    in_line1_mask = frame.apply(town_in_line1, axis=1)
    in_line1 = frame.loc[in_line1_mask, [record_id, *source_fields, *(col[k] for k in LINE1_FIELDS)]]
    line1_path = out_dir / f"{prefix}_town_in_line1_review.csv"
    in_line1.to_csv(line1_path, index=False, encoding="utf-8")

    # --- policy audit from the detailed JSON --------------------------------
    audit = {"line1_protected_town": 0, "line1_protected_country": 0,
             "town_probability_gated": 0, "country_probability_gated": 0,
             "line1_modified": 0, "policy_names": {}}
    if detailed.is_file():
        for doc in read_detailed_jsonl(detailed):
            block = doc.get("groups", {}).get(group, {}).get("retraction", {})
            name = block.get("policy_name", "")
            audit["policy_names"][name] = audit["policy_names"].get(name, 0) + 1
            audit["line1_protected_town"] += block.get("town_skip_reason") == SKIP_PROTECTED_ONLY
            audit["line1_protected_country"] += block.get("country_skip_reason") == SKIP_PROTECTED_ONLY
            audit["town_probability_gated"] += block.get("town_skip_reason") == SKIP_PROBABILITY
            audit["country_probability_gated"] += block.get("country_skip_reason") == SKIP_PROBABILITY
            before = block.get("actual_column_before_retraction", {}).get(line1)
            after = block.get("actual_column_after_retraction", {}).get(line1)
            audit["line1_modified"] += before is not None and before != after
    else:
        audit["note"] = f"detailed JSONL not found: {detailed}"

    # --- changed vs EXP-01, when its output is present ----------------------
    comparison: dict[str, object] = {}
    try:
        baseline = load_config(args.baseline_config, base_dir=MODEL_ROOT)
        baseline_csv = baseline.path(baseline.processing.output_path)
        if baseline_csv.is_file():
            b = pd.read_csv(baseline_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            key = col["combined_address_retracted"]
            merged = frame[[record_id, key]].merge(b[[record_id, key]], on=record_id, how="inner",
                                                   suffixes=("_exp02", "_exp01"))
            comparison = {
                "baseline_output": str(baseline_csv),
                "records_compared": int(len(merged)),
                "retracted_address_changed_vs_exp01": int(
                    merged[f"{key}_exp02"].ne(merged[f"{key}_exp01"]).sum()),
            }
        else:
            comparison = {"note": f"EXP-01 output not found: {baseline_csv}"}
    except FileNotFoundError as exc:
        comparison = {"note": str(exc)}

    cleaned = frame[col["combined_address_cleaned"]].astype(str)
    retracted = frame[col["combined_address_retracted"]].astype(str)
    states = frame[col["hitl_state"]].astype(str)
    changed = cleaned.ne(retracted)
    diagnostics = {
        "experiment": "EXP-02", "policy": "line1_protected_confidence_gated", "group_id": group,
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
        "policy_audit": audit,
        "comparison_with_exp01": comparison,
    }
    diag_path = out_dir / f"{prefix}_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(json.dumps(diagnostics, indent=2))
    print(f"\nretraction review     : {review_path}  ({len(review)} rows)")
    print(f"town-in-line-1 review : {line1_path}  ({len(in_line1)} rows)")
    print(f"diagnostics           : {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
