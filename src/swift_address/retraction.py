"""Deterministic removal of verified Town/Country evidence from source fields.

Retraction answers: *if we take out only the location information the address
actually stated and we deterministically verified, what is left?*

Three rules keep this safe, and all three matter:

1. **Only verified evidence is removed.** Town is retracted only when
   ``predicted_town_exists`` is True; Country only when
   ``predicted_country_exists`` is True. A country the model *inferred* from
   reference data was never in the text, so there is nothing to take out — it
   stays a prediction and the address is left alone.

2. **Removal is token-span based, never substring replacement.** ``AERONAUTICA``
   cannot lose ``RONA``; ``CUSTOMS`` cannot lose ``US``; ``IN`` in ordinary prose
   is not removed unless country verification already concluded it really meant
   India. Matching runs on whole tokens over the original text, and the exact
   textual forms come from the same ISO verification that produced
   ``country_exists``.

3. **Work happens at the original source-column level.** Each configured field
   is processed independently, so before/after is reportable per column and the
   retracted combined address is *rebuilt* from the after-values using the same
   Pass 1 conventions — never reverse-engineered from a mutated combined string.

The original input columns are never modified. Nothing here calls a model.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .cleaning import clean_address, normalize_whitespace, trim_field
from .grouping import build_combined_address
from .schemas import NO_COUNTRY, NO_TOWN

__all__ = [
    "RetractionResult",
    "TokenSpan",
    "null_retraction",
    "remove_token_phrases",
    "retract_group",
    "token_spans",
]

#: Characters treated as separators when a removal leaves an orphaned delimiter.
_SEPARATORS = ",;:/|-–—"


@dataclass(frozen=True)
class TokenSpan:
    """One alphanumeric token and its character range in the original text."""

    text: str
    start: int
    end: int

    @property
    def key(self) -> str:
        """NFKC-uppercased comparison form. Spans stay in original coordinates."""
        return unicodedata.normalize("NFKC", self.text).upper()


@dataclass(frozen=True)
class RetractionResult:
    """Per-group retraction outcome, at source-column granularity."""

    before: dict[str, str]
    after: dict[str, str]
    combined_address_retracted: str
    comment: str
    retracted_entities: tuple[str, ...] = ()
    removed_forms: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> dict[str, Any]:
        return {
            "combined_address_retracted": self.combined_address_retracted,
            "comment": self.comment,
            "actual_column_before_retraction": dict(self.before),
            "actual_column_after_retraction": dict(self.after),
            "retracted_entities": list(self.retracted_entities),
            "removed_forms": list(self.removed_forms),
        }


def token_spans(text: str) -> tuple[TokenSpan, ...]:
    """Split text into alphanumeric tokens, keeping original character offsets.

    Offsets are into the *original* string, so removals never disturb text the
    caller did not ask to remove.
    """
    spans: list[TokenSpan] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isalnum():
            if start is None:
                start = index
        elif start is not None:
            spans.append(TokenSpan(text[start:index], start, index))
            start = None
    if start is not None:
        spans.append(TokenSpan(text[start:], start, len(text)))
    return tuple(spans)


def remove_token_phrases(
    text: str,
    phrases: Iterable[str],
    *,
    restrict_to_trailing_tokens: int | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Remove every standalone token-phrase occurrence of each phrase.

    Returns ``(new_text, removed_forms)``. Matching is case-insensitive and
    aligned to whole tokens, so a phrase occurring only as a substring of a
    longer word is not a match and is not removed. Repeated occurrences are all
    removed (``CITIGROUP CENTRE AUCKLAND AUCKLAND`` → ``CITIGROUP CENTRE``).

    Each removed token span also swallows one adjacent run of separator
    characters — the preceding run when there is one, otherwise the following —
    so a removal does not leave an orphaned comma or a double space behind.
    Whitespace is normalized afterwards; nothing else about the surviving text
    is rewritten.

    ``restrict_to_trailing_tokens`` limits matching to occurrences ending within
    the final N tokens. This is what keeps an ambiguous alpha-2 code such as
    ``IN`` from being stripped out of ordinary prose: only the trailing
    country-position occurrence is eligible.
    """
    original = text or ""
    if not original.strip():
        return original, ()

    spans = token_spans(original)
    if not spans:
        return original, ()

    keys = [span.key for span in spans]
    removals: list[tuple[int, int]] = []
    removed_forms: list[str] = []

    for phrase in phrases:
        needle = [
            unicodedata.normalize("NFKC", token.text).upper()
            for token in token_spans(phrase or "")
        ]
        if not needle or len(needle) > len(keys):
            continue

        earliest_end = (
            len(keys) - restrict_to_trailing_tokens
            if restrict_to_trailing_tokens is not None
            else 0
        )

        matched = False
        index = 0
        while index <= len(keys) - len(needle):
            end_index = index + len(needle) - 1
            if keys[index : index + len(needle)] == needle and end_index >= earliest_end:
                first, last = spans[index], spans[end_index]
                removals.append(
                    _expanded_span(original, first.start, last.end, index == 0)
                )
                matched = True
                index += len(needle)
            else:
                index += 1
        if matched:
            removed_forms.append(phrase)

    if not removals:
        return original, ()

    kept: list[str] = []
    cursor = 0
    for start, end in sorted(removals):
        if start > cursor:
            kept.append(original[cursor:start])
        cursor = max(cursor, end)
    kept.append(original[cursor:])

    result = normalize_whitespace("".join(kept))
    # Only separators orphaned by the removal are stripped from the ends; the
    # interior of the surviving text is untouched.
    result = normalize_whitespace(result.strip(_SEPARATORS + " "))
    return result, tuple(removed_forms)


def _expanded_span(
    text: str, start: int, end: int, is_first_token: bool
) -> tuple[int, int]:
    """Grow a removal to absorb one adjacent separator/whitespace run."""
    if not is_first_token:
        cursor = start
        while cursor > 0 and (text[cursor - 1].isspace() or text[cursor - 1] in _SEPARATORS):
            cursor -= 1
        if cursor < start:
            return cursor, end
    cursor = end
    while cursor < len(text) and (text[cursor].isspace() or text[cursor] in _SEPARATORS):
        cursor += 1
    return start, cursor


def retract_group(
    source_values: Mapping[str, Any],
    source_fields: Sequence[str],
    *,
    town: str,
    country_value: str,
    town_exists: bool,
    country_exists: bool,
    iso_provider: Any = None,
    zero_is_missing: bool = True,
) -> RetractionResult:
    """Retract verified Town/Country evidence from one group's source columns.

    ``source_values`` is read only — the caller's dataframe is never mutated.
    The retracted combined address is rebuilt from the after-values with
    :func:`swift_address.grouping.build_combined_address`, so it follows exactly
    the same joining and missing-field conventions as Pass 1.
    """
    before = {
        field_name: trim_field(source_values.get(field_name))
        for field_name in source_fields
    }

    retract_town = bool(town_exists and town not in {"", NO_TOWN})
    country_codes = (
        [code for code in country_value.split(",") if code]
        if country_exists and country_value not in {"", NO_COUNTRY}
        else []
    )
    # A comma-separated candidate set is unresolved by definition, and
    # `country_exists` is only ever True for a single resolved code — but guard
    # explicitly rather than relying on that invariant holding forever.
    retract_country = bool(country_codes) and len(country_codes) == 1

    if not retract_town and not retract_country:
        combined = build_combined_address(
            [before[name] for name in source_fields], zero_is_missing=zero_is_missing
        )
        return RetractionResult(
            before=before,
            after=dict(before),
            combined_address_retracted=clean_address(combined),
            comment=_comment(False, False, town, "", ()),
            retracted_entities=(),
        )

    code = country_codes[0] if retract_country else ""

    # Which textual country forms are eligible is decided ONCE, against the
    # combined address — the same text that produced `country_exists`. Deciding
    # it per field would be wrong: a token sitting mid-address can be in the
    # trailing window of its own short field, which is how "SUITE 5 IN TOWER"
    # would otherwise lose its preposition.
    combined_before = build_combined_address(
        [before[name] for name in source_fields], zero_is_missing=zero_is_missing
    )
    if retract_country and iso_provider is not None:
        eligible_forms = list(iso_provider.matched_presence_forms(combined_before, code))
        code_is_ambiguous = iso_provider.is_ambiguous_alpha2(code)
        trailing_window = iso_provider.trailing_country_token_window
    else:
        eligible_forms = [code] if retract_country else []
        code_is_ambiguous = False
        trailing_window = 0

    # Unrestricted forms: full country names, and non-colliding codes.
    open_forms = [
        form for form in eligible_forms
        if not (code_is_ambiguous and form.upper() == code.upper())
    ]
    # A colliding code is only removable in trailing country position, and only
    # from the field that actually carries the address tail.
    restricted_code = (
        code if (code_is_ambiguous and code in eligible_forms) else ""
    )
    tail_field = _last_non_empty_field(before, source_fields)

    after: dict[str, str] = {}
    removed_forms: list[str] = []
    town_removed = False
    country_removed = False

    def _note(form: str) -> None:
        nonlocal town_removed, country_removed
        if retract_town and form == town:
            town_removed = True
        else:
            country_removed = True
        if form not in removed_forms:
            removed_forms.append(form)

    for field_name in source_fields:
        value = before[field_name]
        if not value:
            after[field_name] = value
            continue

        phrases: list[str] = ([town] if retract_town else []) + open_forms
        updated, removed = remove_token_phrases(value, phrases)
        for form in removed:
            _note(form)

        if restricted_code and field_name == tail_field:
            updated, removed = remove_token_phrases(
                updated, [restricted_code],
                restrict_to_trailing_tokens=trailing_window,
            )
            for form in removed:
                _note(form)

        after[field_name] = updated

    combined = build_combined_address(
        [after[name] for name in source_fields], zero_is_missing=zero_is_missing
    )

    entities: list[str] = []
    if town_removed:
        entities.append("town")
    if country_removed:
        entities.append("country")

    return RetractionResult(
        before=before,
        after=after,
        combined_address_retracted=clean_address(combined),
        comment=_comment(town_removed, country_removed, town, code, tuple(removed_forms)),
        retracted_entities=tuple(entities),
        removed_forms=tuple(removed_forms),
    )


def _last_non_empty_field(
    values: Mapping[str, str], source_fields: Sequence[str]
) -> str:
    """The configured field carrying the address tail, or "" when all are empty."""
    for field_name in reversed(list(source_fields)):
        if values.get(field_name):
            return field_name
    return ""


def _comment(
    town_removed: bool,
    country_removed: bool,
    town: str,
    country_code: str,
    removed_forms: tuple[str, ...],
) -> str:
    """One deterministic line (occasionally two). Never model-written."""
    if town_removed and country_removed:
        return (
            f"Retracted Town={town} and Country={country_code} from verified "
            "explicit address evidence."
        )
    if town_removed:
        return (
            f"Retracted Town={town}. Country was not explicitly verified in the "
            "input, so it was retained only as a prediction."
        )
    if country_removed:
        return (
            f"Retracted Country={country_code}. Town was not explicitly verified "
            "in the input, so it was retained only as a prediction."
        )
    if removed_forms:  # pragma: no cover - defensive
        return "Retracted verified evidence: " + ", ".join(removed_forms) + "."
    return (
        "No retraction: neither predicted Town nor Country was explicitly "
        "verified in the source address."
    )


def null_retraction(source_fields: Sequence[str]) -> RetractionResult:
    """Retraction record for a null-skipped group: nothing present, nothing removed."""
    empty = {name: "" for name in source_fields}
    return RetractionResult(
        before=empty,
        after=dict(empty),
        combined_address_retracted="",
        comment="",
        retracted_entities=(),
    )
