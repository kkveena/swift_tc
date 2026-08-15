"""Address group configuration and combined-address construction.

Group definitions are external data, never code. The pipeline supports any
number of groups and any number of address lines per group; the supplied
sample happens to use 16 groups of 3 lines, and nothing here knows that.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from .cleaning import clean_address, is_missing_field, join_lines, trim_field

__all__ = [
    "AddressGroup",
    "GroupConfig",
    "GroupConfigError",
    "MissingInputColumnsError",
    "build_combined_address",
    "load_group_config",
]

# Columns of the group-config CSV that are not address-line fields.
_RESERVED_COLUMNS = {"group_id", "enabled", "notes", "label", "description"}

_TRUTHY = {"1", "true", "yes", "y", "on", "t"}
_FALSY = {"0", "false", "no", "n", "off", "f", ""}


class GroupConfigError(ValueError):
    """The group configuration is structurally invalid."""


class MissingInputColumnsError(GroupConfigError):
    """Configured source columns are absent from the input dataframe.

    Raised before any model call so a misconfigured run costs nothing.
    """

    def __init__(self, missing_by_group: Mapping[str, Sequence[str]]) -> None:
        self.missing_by_group = {
            group: tuple(columns) for group, columns in missing_by_group.items()
        }
        detail = "; ".join(
            f"group {group}: {', '.join(columns)}"
            for group, columns in sorted(self.missing_by_group.items())
        )
        all_missing = sorted(
            {column for columns in self.missing_by_group.values() for column in columns}
        )
        super().__init__(
            f"{len(all_missing)} configured source column(s) are missing from the "
            f"input: {', '.join(all_missing)}. Affected groups -> {detail}. "
            "No model calls were made."
        )


@dataclass(frozen=True)
class AddressGroup:
    """One configured address group: an ID plus ordered source fields."""

    group_id: str
    source_fields: tuple[str, ...]
    enabled: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.group_id:
            raise GroupConfigError("group_id must not be empty")
        if not self.source_fields:
            raise GroupConfigError(
                f"group {self.group_id} has no source fields configured"
            )
        seen: set[str] = set()
        duplicates: list[str] = []
        for field in self.source_fields:
            if field in seen:
                duplicates.append(field)
            seen.add(field)
        if duplicates:
            raise GroupConfigError(
                f"group {self.group_id} lists duplicate source field(s): "
                f"{', '.join(sorted(set(duplicates)))}"
            )

    @property
    def line_count(self) -> int:
        return len(self.source_fields)


@dataclass(frozen=True)
class GroupConfig:
    """The validated set of address groups, in configuration order."""

    groups: tuple[AddressGroup, ...]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for group in self.groups:
            if group.group_id in seen:
                duplicates.append(group.group_id)
            seen.add(group.group_id)
        if duplicates:
            raise GroupConfigError(
                "duplicate group_id(s) in group configuration: "
                f"{', '.join(sorted(set(duplicates)))}"
            )

    @property
    def enabled_groups(self) -> tuple[AddressGroup, ...]:
        return tuple(group for group in self.groups if group.enabled)

    @property
    def all_source_fields(self) -> tuple[str, ...]:
        """Every source field referenced by an enabled group, deduplicated."""
        ordered: list[str] = []
        seen: set[str] = set()
        for group in self.enabled_groups:
            for field in group.source_fields:
                if field not in seen:
                    seen.add(field)
                    ordered.append(field)
        return tuple(ordered)

    def validate_against_columns(self, columns: Iterable[str]) -> None:
        """Fail fast when an enabled group references an absent input column."""
        available = set(columns)
        missing_by_group: dict[str, list[str]] = {}
        for group in self.enabled_groups:
            missing = [f for f in group.source_fields if f not in available]
            if missing:
                missing_by_group[group.group_id] = missing
        if missing_by_group:
            raise MissingInputColumnsError(missing_by_group)


def load_group_config(path: str | Path) -> GroupConfig:
    """Load a group configuration from CSV or YAML.

    CSV schema (the supplied sample)::

        group_id,address_line_1,address_line_2,address_line_3,enabled,notes

    Any number of ``address_line_*`` columns is accepted, in column order.
    A blank line cell simply means that group has fewer lines.

    YAML schema::

        groups:
          - group_id: "1"
            source_fields: [ADDR_1, ADDR_2]
            enabled: true
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"group configuration not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        groups = _load_yaml_groups(config_path)
    elif suffix == ".csv":
        groups = _load_csv_groups(config_path)
    else:
        raise GroupConfigError(
            f"unsupported group configuration format {suffix!r}: use .csv or .yaml"
        )

    if not groups:
        raise GroupConfigError(f"{config_path} defines no address groups")
    return GroupConfig(groups=tuple(groups), source_path=config_path)


def _load_csv_groups(path: Path) -> list[AddressGroup]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise GroupConfigError(f"{path} has no header row")
        headers = [name.strip() for name in reader.fieldnames]
        if "group_id" not in headers:
            raise GroupConfigError(f"{path} must define a 'group_id' column")

        # Address-line columns are every non-reserved header, kept in file order
        # so line ordering comes from the config rather than from sorting.
        line_columns = [name for name in headers if name.lower() not in _RESERVED_COLUMNS]
        if not line_columns:
            raise GroupConfigError(
                f"{path} defines no address-line columns beside {sorted(_RESERVED_COLUMNS)}"
            )

        groups: list[AddressGroup] = []
        for line_number, row in enumerate(reader, start=2):
            normalized = {
                (key or "").strip(): trim_field(value) for key, value in row.items()
            }
            group_id = normalized.get("group_id", "")
            if not group_id:
                continue  # tolerate trailing blank lines
            fields = tuple(
                normalized[column]
                for column in line_columns
                if normalized.get(column)
            )
            if not fields:
                raise GroupConfigError(
                    f"{path} line {line_number}: group {group_id} has no source fields"
                )
            groups.append(
                AddressGroup(
                    group_id=group_id,
                    source_fields=fields,
                    enabled=_parse_bool(normalized.get("enabled", "true"), path, line_number),
                    notes=normalized.get("notes", ""),
                )
            )
    return groups


def _load_yaml_groups(path: Path) -> list[AddressGroup]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) or "groups" not in data:
        raise GroupConfigError(f"{path} must contain a top-level 'groups' list")

    groups: list[AddressGroup] = []
    for index, entry in enumerate(data["groups"], start=1):
        if not isinstance(entry, dict):
            raise GroupConfigError(f"{path} groups[{index}] must be a mapping")
        group_id = trim_field(entry.get("group_id"))
        fields = entry.get("source_fields") or []
        groups.append(
            AddressGroup(
                group_id=group_id,
                source_fields=tuple(trim_field(f) for f in fields if trim_field(f)),
                enabled=bool(entry.get("enabled", True)),
                notes=trim_field(entry.get("notes", "")),
            )
        )
    return groups


def _parse_bool(value: str, path: Path, line_number: int) -> bool:
    text = value.strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise GroupConfigError(
        f"{path} line {line_number}: cannot interpret enabled={value!r} as a boolean"
    )


def build_combined_address(
    values: Sequence[object],
    *,
    zero_is_missing: bool = True,
    separator: str = " ",
) -> str:
    """Join configured source values into one combined address.

    Each value is trimmed; ``None``/NaN, empty strings and whole-field ``"0"``
    are dropped; the survivors are joined in configuration order with a single
    space. Digits inside legitimate values are never removed.
    """
    kept = [
        trim_field(value)
        for value in values
        if not is_missing_field(value, zero_is_missing=zero_is_missing)
    ]
    return join_lines(kept, separator=separator)


def build_group_addresses(
    row: Mapping[str, object],
    group: AddressGroup,
    *,
    zero_is_missing: bool = True,
) -> tuple[str, str]:
    """Return ``(combined_address, combined_address_cleaned)`` for one row/group."""
    combined = build_combined_address(
        [row.get(field) for field in group.source_fields],
        zero_is_missing=zero_is_missing,
    )
    return combined, clean_address(combined)
