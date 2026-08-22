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
    "token_phrase_matches",
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
    #: How many standalone Town occurrences the group's source fields carried,
    #: and how many were actually removed. At most one is ever removed, so a
    #: count above one is the audit record of a repeated Town that was
    #: deliberately left partly in place. Audit only — never a CSV column.
    town_occurrences_found: int = 0
    town_occurrences_removed: int = 0

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
            "town_occurrences_found": self.town_occurrences_found,
            "town_occurrences_removed": self.town_occurrences_removed,
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


def token_phrase_matches(
    text: str,
    phrase: str,
    *,
    restrict_to_trailing_tokens: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Every standalone token-phrase occurrence of ``phrase``, in text order.

    Returns ``(first_token_index, last_token_index)`` pairs over the token
    sequence of ``text``. Matching is exactly the matching
    :func:`remove_token_phrases` performs — whole tokens, case-insensitive, so a
    phrase occurring only inside a longer word is not a match. Occurrences never
    overlap: a match consumes its tokens before the scan continues.

    Exposed so a caller can decide *which* occurrence to act on before removing
    anything — the group-level Town rule needs to count occurrences across
    several fields before touching any of them.
    """
    spans = token_spans(text or "")
    if not spans:
        return ()
    keys = [span.key for span in spans]
    needle = [
        unicodedata.normalize("NFKC", token.text).upper()
        for token in token_spans(phrase or "")
    ]
    return _matches(keys, needle, restrict_to_trailing_tokens)


def _matches(
    keys: Sequence[str],
    needle: Sequence[str],
    restrict_to_trailing_tokens: int | None,
) -> tuple[tuple[int, int], ...]:
    """Non-overlapping token-index matches of ``needle`` within ``keys``."""
    if not needle or len(needle) > len(keys):
        return ()
    earliest_end = (
        len(keys) - restrict_to_trailing_tokens
        if restrict_to_trailing_tokens is not None
        else 0
    )
    found: list[tuple[int, int]] = []
    index = 0
    while index <= len(keys) - len(needle):
        end_index = index + len(needle) - 1
        if list(keys[index : index + len(needle)]) == list(needle) and end_index >= earliest_end:
            found.append((index, end_index))
            index += len(needle)
        else:
            index += 1
    return tuple(found)


def remove_token_phrases(
    text: str,
    phrases: Iterable[str],
    *,
    restrict_to_trailing_tokens: int | None = None,
    max_occurrences: int | None = None,
    prefer_last: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Remove standalone token-phrase occurrences of each phrase.

    Returns ``(new_text, removed_forms)``. Matching is case-insensitive and
    aligned to whole tokens, so a phrase occurring only as a substring of a
    longer word is not a match and is not removed.

    Each removed token span also swallows one adjacent run of separator
    characters — the preceding run when there is one, otherwise the following —
    so a removal does not leave an orphaned comma or a double space behind.
    Whitespace is normalized afterwards; nothing else about the surviving text
    is rewritten.

    ``restrict_to_trailing_tokens`` limits matching to occurrences ending within
    the final N tokens. This is what keeps an ambiguous alpha-2 code such as
    ``IN`` from being stripped out of ordinary prose: only the trailing
    country-position occurrence is eligible.

    ``max_occurrences`` caps how many occurrences of *each* phrase are removed.
    The default ``None`` removes every eligible occurrence, which is what
    Country retraction wants — a country code repeated in the text is the same
    single piece of evidence stated twice. ``max_occurrences=1`` with
    ``prefer_last=True`` removes only the right-most occurrence, which is what
    the Town rule wants: an earlier occurrence can belong to an institution or
    building name, so only the later, locality-position one is taken out.

    Note the cap is per phrase *per call*. A rule that limits occurrences across
    several fields must decide which field to act on first — see
    :func:`token_phrase_matches`.
    """
    original = text or ""
    if not original.strip():
        return original, ()
    if max_occurrences is not None and max_occurrences <= 0:
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
        found = _matches(keys, needle, restrict_to_trailing_tokens)
        if not found:
            continue
        if max_occurrences is not None:
            found = (
                found[-max_occurrences:] if prefer_last else found[:max_occurrences]
            )
        for start_index, end_index in found:
            first, last = spans[start_index], spans[end_index]
            removals.append(
                _expanded_span(original, first.start, last.end, start_index == 0)
            )
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
    :func:`models.swft_tc.src.grouping.build_combined_address`, so it follows exactly
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

    # --- Town: at most ONE occurrence per GROUP, the right-most one ---------
    # A Town can legitimately appear more than once in one address — once inside
    # an institution, building or branch name, and once as the locality itself
    # ("CITIGROUP CENTRE AUCKLAND AUCKLAND"). Removing every occurrence deletes
    # part of the organisation's name along with the location. Only one
    # occurrence is evidence of the locality, so only one is retracted.
    #
    # Which one is a deterministic positional choice, not a semantic guess: the
    # right-most standalone occurrence across the configured source fields in
    # configuration order, because the locality sits later in an address than a
    # descriptive prefix does. The scan spans the whole group, so a Town in the
    # final line wins over an earlier one in a previous line.
    town_occurrences_found = 0
    town_target_field = ""
    if retract_town:
        for field_name in source_fields:
            value = before[field_name]
            if not value:
                continue
            occurrences = len(token_phrase_matches(value, town))
            if occurrences:
                town_occurrences_found += occurrences
                town_target_field = field_name

    after: dict[str, str] = {}
    removed_forms: list[str] = []
    town_removed = False
    country_removed = False
    town_occurrences_removed = 0

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

        updated = value

        # Town first, so the single occurrence is chosen against the field text
        # exactly as it was written rather than against a country-stripped
        # remnant. Only the one field carrying the right-most occurrence is
        # touched; every earlier occurrence, in this field or any other, stays.
        if retract_town and field_name == town_target_field:
            updated, removed = remove_token_phrases(
                updated, [town], max_occurrences=1, prefer_last=True
            )
            for form in removed:
                _note(form)
                town_occurrences_removed += 1

        # Country keeps its existing rule: every verified occurrence goes, since
        # a country code repeated in the text is one piece of evidence stated
        # twice, not two separate facts.
        if open_forms:
            updated, removed = remove_token_phrases(updated, open_forms)
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
        town_occurrences_found=town_occurrences_found,
        town_occurrences_removed=town_occurrences_removed,
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
