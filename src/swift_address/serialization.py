"""Nested per-record detail, written as a stream.

The CSV stays the flat compatibility artifact. Audit and evaluation depth lives
here instead of growing dozens more columns: one JSON object per input record,
with every enabled group nested inside it.

**JSON Lines is the default and the writer streams.** One record is built,
serialized, and written before the next is touched, so peak memory is a single
record rather than the whole dataset. A conventional JSON array is available for
small development runs, and even then the records are streamed into the file
rather than assembled into one giant in-memory list.

Sensitivity: this file contains raw address data. It belongs under ``outputs/``,
which is git-ignored.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from .evaluation import null_cross_entropy, null_ground_truth
from .retraction import RetractionResult, null_retraction, retract_group
from .schemas import NO_COUNTRY, NO_TOWN

__all__ = [
    "SCHEMA_VERSION",
    "build_record_document",
    "iter_record_documents",
    "write_detailed_json",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "phase1-detailed-v1"

#: Marks a group short-circuited before any model call.
STATUS_NULL_SKIP = "null_skip"
STATUS_EXTRACTED = "extracted"


def _clean(value: Any) -> Any:
    """Convert a value into something JSON can represent honestly.

    NaN and pandas NA become ``null`` — never the bare token ``NaN``, which is
    not valid JSON and which downstream parsers reject or misread.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (bool, int, str)):
        return value
    if hasattr(value, "item"):          # numpy scalar
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return str(value)
    if pd.isna(value):                  # pragma: no cover - defensive
        return None
    return value


def _float_or_none(value: Any) -> float | None:
    cleaned = _clean(value)
    if cleaned is None or isinstance(cleaned, bool):
        return None if cleaned is None else float(cleaned)
    try:
        result = float(cleaned)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _bool_or_none(value: Any) -> bool | None:
    cleaned = _clean(value)
    return None if cleaned is None else bool(cleaned)


def build_record_document(
    row: Mapping[str, Any],
    *,
    config: Any,
    group_config: Any,
    decisions_by_address: Mapping[str, Any],
    iso_provider: Any = None,
    include_empty_groups: bool = True,
) -> dict[str, Any]:
    """Build the nested document for one input record.

    Run-wide metadata (model, prompt version, reference provenance) is
    deliberately absent: it belongs in ``run_metrics.json`` and
    ``executive_summary.json``, not repeated inside every group of every record.
    """
    record_id = str(row[config.project.record_id_column])
    groups: dict[str, Any] = {}

    for group in group_config.enabled_groups:
        # Look columns up by field key, never by position, so reordering
        # OUTPUT_FIELD_KEYS cannot silently mis-map a value here.
        def column(field_key: str, group_id: str = group.group_id) -> Any:
            return row.get(config.output.column_name(field_key, group_id))

        cleaned = str(column("combined_address_cleaned") or "")

        if not cleaned:
            if include_empty_groups:
                groups[group.group_id] = _null_group_document(group)
            continue

        groups[group.group_id] = _group_document(
            row=row,
            group=group,
            column=column,
            cleaned=cleaned,
            decision=decisions_by_address.get(cleaned),
            iso_provider=iso_provider,
            zero_is_missing=config.input.zero_field_is_missing,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "groups": groups,
    }


def _null_group_document(group: Any) -> dict[str, Any]:
    """A short record for a group skipped before any model call.

    Included for completeness but kept lean: repeating a full set of empty
    prediction/scoring blocks for what is often the large majority of instances
    would bloat the file for no information gain.
    """
    retraction = null_retraction(group.source_fields)
    return {
        "status": STATUS_NULL_SKIP,
        "source_fields": list(group.source_fields),
        "address": {"combined_address": "", "combined_address_cleaned": ""},
        "prediction": {
            "town": NO_TOWN,
            "country": NO_COUNTRY,
            "country_name": NO_COUNTRY,
            "town_probability": 0.0,
            "country_probability": 0.0,
        },
        "text_evidence": {
            "predicted_town_exists": False,
            "predicted_country_exists": False,
        },
        "ground_truth_validation": {
            **null_ground_truth().to_dict(),
            "reference_status": "not_checked",
            "reference_country_codes": [],
        },
        "scoring": {
            "scenario": "null_skip",
            "composite_weighted_score": 0.0,
            "needs_hitl": False,
        },
        "cross_entropy": null_cross_entropy().to_dict(),
        "retraction": retraction.to_dict(),
    }


def _group_document(
    *,
    row: Mapping[str, Any],
    group: Any,
    column: Any,
    cleaned: str,
    decision: Any,
    iso_provider: Any,
    zero_is_missing: bool,
) -> dict[str, Any]:
    source_values = {
        field_name: str(row.get(field_name, "") or "")
        for field_name in group.source_fields
    }

    town = str(column("predicted_town") or "")
    country_value = str(column("predicted_country") or "")
    town_exists = bool(_bool_or_none(column("predicted_town_exists")))
    country_exists = bool(_bool_or_none(column("predicted_country_exists")))

    # Recomputed from the same pure function the CSV columns used, so the two
    # representations cannot drift apart. A test asserts they agree.
    retraction: RetractionResult = retract_group(
        source_values,
        group.source_fields,
        town=town,
        country_value=country_value,
        town_exists=town_exists,
        country_exists=country_exists,
        iso_provider=iso_provider,
        zero_is_missing=zero_is_missing,
    )

    verified = getattr(decision, "verified", None)
    score = getattr(decision, "score", None)
    ground_truth = getattr(decision, "ground_truth", None) or null_ground_truth()
    cross_entropy = getattr(decision, "cross_entropy", None) or null_cross_entropy()

    document: dict[str, Any] = {
        "status": STATUS_EXTRACTED,
        "source_fields": list(group.source_fields),
        "address": {
            "combined_address": str(column("combined_address") or ""),
            "combined_address_cleaned": cleaned,
        },
        "prediction": {
            "town": town,
            "country": country_value,
            "country_name": str(column("predicted_country_name") or ""),
            "town_probability": _float_or_none(column("predicted_town_probability")),
            "country_probability": _float_or_none(
                column("predicted_country_probability")
            ),
        },
        "text_evidence": {
            "predicted_town_exists": town_exists,
            "predicted_country_exists": country_exists,
        },
        "ground_truth_validation": {
            **ground_truth.to_dict(),
            "reference_status": (
                verified.reference_status if verified is not None else "not_checked"
            ),
            "reference_country_codes": (
                list(verified.reference_codes) if verified is not None else []
            ),
        },
        "scoring": _scoring_block(score, column),
        "cross_entropy": cross_entropy.to_dict(),
        "rationale": {
            "town": str(column("rationale_town") or ""),
            "country": str(column("rationale_country") or ""),
        },
        "retraction": retraction.to_dict(),
    }
    return document


def _scoring_block(score: Any, column: Any) -> dict[str, Any]:
    composite = _float_or_none(column("composite_weighted_score"))
    if score is None:
        return {"composite_weighted_score": composite}
    return {
        "scenario": score.scenario,
        "town_weight": _float_or_none(score.town_weight),
        "country_weight": _float_or_none(score.country_weight),
        "adjusted_town_score": _float_or_none(score.adjusted_town_score),
        "adjusted_country_score": _float_or_none(score.adjusted_country_score),
        "composite_weighted_score": composite,
        "needs_hitl": bool(score.needs_hitl),
    }


def iter_record_documents(
    frame: pd.DataFrame,
    *,
    config: Any,
    group_config: Any,
    decisions_by_address: Mapping[str, Any],
    iso_provider: Any = None,
    include_empty_groups: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield one document per input row, building them lazily."""
    columns = list(frame.columns)
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        yield build_record_document(
            row,
            config=config,
            group_config=group_config,
            decisions_by_address=decisions_by_address,
            iso_provider=iso_provider,
            include_empty_groups=include_empty_groups,
        )


def write_detailed_json(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    config: Any,
    group_config: Any,
    decisions_by_address: Mapping[str, Any],
    iso_provider: Any = None,
    output_format: str = "jsonl",
    include_empty_groups: bool = True,
) -> Path:
    """Stream the detailed output to disk. Returns the path written.

    Parent directories are created on demand. Encoding is UTF-8 and keys are
    emitted in a deterministic order so two runs over the same data produce
    byte-identical files.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    documents = iter_record_documents(
        frame,
        config=config,
        group_config=group_config,
        decisions_by_address=decisions_by_address,
        iso_provider=iso_provider,
        include_empty_groups=include_empty_groups,
    )

    written = 0
    with target.open("w", encoding="utf-8") as handle:
        if output_format == "json":
            # Still streamed: the array is punctuated as records are written
            # rather than materialized as one list.
            handle.write("[\n")
            for index, document in enumerate(documents):
                if index:
                    handle.write(",\n")
                handle.write(_dumps(document))
                written += 1
            handle.write("\n]\n")
        else:
            for document in documents:
                handle.write(_dumps(document))
                handle.write("\n")
                written += 1

    logger.info("wrote %d detailed record(s) to %s", written, target)
    return target


def _dumps(document: Mapping[str, Any]) -> str:
    """Serialize one document. ``allow_nan=False`` makes invalid JSON impossible.

    Every numeric field has already been through :func:`_clean`, so a NaN
    reaching this point is a bug rather than a data condition — and it will
    raise here instead of silently emitting the invalid token ``NaN``.
    """
    return json.dumps(document, ensure_ascii=False, allow_nan=False, default=_clean)


def read_detailed_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Read a JSONL detail file back, one document at a time. Test/tooling helper."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
