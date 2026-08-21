"""CSV input/output, the error sidecar, and run metrics.

Input handling is conservative by design: every column is read as a string so
``RECORD_ID`` stays ``CA0000000318`` and a postal code stays ``02111`` rather
than becoming ``2111``. Column order is preserved, unknown columns are carried
through untouched, and no source column is ever mutated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "ERROR_COLUMNS",
    "ProcessingError",
    "read_input_csv",
    "read_output_csv",
    "write_errors_csv",
    "write_metrics_json",
    "write_output_csv",
]

logger = logging.getLogger(__name__)

#: Columns of ``processing_errors.csv``. No raw address: errors are joined back
#: to the cache and the dataframe by ``address_hash``.
ERROR_COLUMNS: tuple[str, ...] = (
    "address_hash",
    "occurrences",
    "group_ids",
    "record_ids",
    "error_type",
    "error_message",
    "model",
    "prompt_version",
    "attempts",
    "timestamp_utc",
)


@dataclass(frozen=True)
class ProcessingError:
    """One failed unique address, with every occurrence it affected."""

    address_hash: str
    occurrences: int
    group_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    error_type: str
    error_message: str
    model: str
    prompt_version: str
    attempts: int = 0
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    #: Cap on record IDs listed per error so one bad address cannot produce an
    #: unbounded cell in the sidecar. `occurrences` still reports the true total.
    MAX_RECORD_IDS = 25

    def to_row(self) -> dict[str, Any]:
        record_ids = self.record_ids[: self.MAX_RECORD_IDS]
        suffix = (
            f" (+{len(self.record_ids) - self.MAX_RECORD_IDS} more)"
            if len(self.record_ids) > self.MAX_RECORD_IDS
            else ""
        )
        return {
            "address_hash": self.address_hash,
            "occurrences": self.occurrences,
            "group_ids": "|".join(self.group_ids),
            "record_ids": "|".join(record_ids) + suffix,
            "error_type": self.error_type,
            # Newlines would break a reader's row alignment expectations.
            "error_message": " ".join(str(self.error_message).split())[:500],
            "model": self.model,
            "prompt_version": self.prompt_version,
            "attempts": self.attempts,
            "timestamp_utc": self.timestamp_utc,
        }


def read_input_csv(
    path: str | Path,
    *,
    record_id_column: str = "RECORD_ID",
) -> pd.DataFrame:
    """Read the input CSV with every column as a string and order preserved.

    ``keep_default_na=False`` stops pandas turning empty cells into NaN, which
    would otherwise make an empty address line indistinguishable from the
    literal string "nan" downstream. Missing values are handled explicitly by
    :func:`models.swft_tc.src.cleaning.is_missing_field`.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"input CSV not found: {csv_path}")

    frame = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
    )

    if record_id_column not in frame.columns:
        raise ValueError(
            f"input CSV {csv_path} has no {record_id_column!r} column; found: "
            f"{', '.join(frame.columns[:10])}..."
        )

    duplicates = [name for name in frame.columns if list(frame.columns).count(name) > 1]
    if duplicates:
        raise ValueError(
            "input CSV has duplicate column name(s): " + ", ".join(sorted(set(duplicates)))
        )

    logger.info("read %d row(s) x %d column(s) from input", len(frame), len(frame.columns))
    return frame


def read_output_csv(
    path: str | Path, *, record_id_column: str = "RECORD_ID"
) -> pd.DataFrame:
    """Read back an output CSV using the same string-preserving settings."""
    return read_input_csv(path, record_id_column=record_id_column)


def write_output_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the expanded dataframe, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(
        "wrote %d row(s) x %d column(s) to %s",
        len(frame),
        len(frame.columns),
        output_path,
    )
    return output_path


def write_errors_csv(errors: Iterable[ProcessingError], path: str | Path) -> Path:
    """Write the processing-error sidecar.

    Always writes the file, even with zero errors: an absent file is ambiguous
    between "clean run" and "sidecar never produced".
    """
    error_path = Path(path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [error.to_row() for error in errors]
    frame = pd.DataFrame(rows, columns=list(ERROR_COLUMNS))
    frame.to_csv(error_path, index=False, encoding="utf-8")
    if rows:
        logger.warning("wrote %d processing error(s) to %s", len(rows), error_path)
    return error_path


def write_metrics_json(metrics: Mapping[str, Any], path: str | Path) -> Path:
    """Write run metrics as JSON."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return metrics_path


def assert_columns_preserved(
    original: Sequence[str], produced: Sequence[str]
) -> None:
    """Verify the output still begins with the input columns, in order."""
    prefix = list(produced)[: len(original)]
    if prefix != list(original):
        differences = [
            f"position {index}: expected {expected!r}, got {actual!r}"
            for index, (expected, actual) in enumerate(zip(original, prefix))
            if expected != actual
        ]
        raise AssertionError(
            "input columns were not preserved in order: " + "; ".join(differences[:5])
        )
