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
    "BASELINE_POLICY",
    "BASELINE_POLICY_NAME",
    "RetractionPolicy",
    "RetractionResult",
    "SKIP_NOT_VERIFIED",
    "SKIP_NO_ELIGIBLE_OCCURRENCE",
    "SKIP_PROBABILITY",
    "SKIP_PROTECTED_ENTITY_SPAN_ONLY",
    "SKIP_PROTECTED_ONLY",
    "TokenSpan",
    "null_retraction",
    "remove_token_phrases",
    "retract_group",
    "token_phrase_matches",
    "token_spans",
]

#: Characters treated as separators when a removal leaves an orphaned delimiter.
_SEPARATORS = ",;:/|-–—"

#: The production baseline policy name (EXP-01): right-most single Town
#: occurrence, all verified Country occurrences, nothing protected, no gate.
BASELINE_POLICY_NAME = "rightmost_town_allmatch_country"

#: Why an entity that was *predicted* was nevertheless not retracted. Audit
#: values only — never CSV columns.
SKIP_NOT_VERIFIED = "not_explicitly_verified"
SKIP_PROBABILITY = "probability_not_above_threshold"
SKIP_PROTECTED_ONLY = "protected_source_only"
SKIP_NO_ELIGIBLE_OCCURRENCE = "no_eligible_occurrence"
SKIP_PROTECTED_ENTITY_SPAN_ONLY = "protected_entity_span_only"

#: Character spans, per source field, that retraction may not modify:
#: ``{"address_line_1": ((0, 23),)}``. Produced by verified entity detection.
ProtectedSpans = Mapping[str, Sequence[tuple[int, int]]]


@dataclass(frozen=True)
class RetractionPolicy:
    """What may be retracted, and from where. Pure data; no behaviour of its own.

    ``protected_source_positions`` are 1-based ordinals into a group's
    configured source fields — ``(1,)`` protects the first configured line of
    every group, whatever it is called. A protected field still contributes to
    the combined address and to every upstream judgement; it is only never
    modified by retraction, and an occurrence inside it never counts as
    eligible.

    ``*_probability_threshold`` is a strict gate: an entity is eligible only
    when its own model probability is **greater than** the threshold. ``None``
    means no gate. The default instance is the production baseline.
    """

    name: str = BASELINE_POLICY_NAME
    protected_source_positions: tuple[int, ...] = ()
    town_probability_threshold: float | None = None
    country_probability_threshold: float | None = None

    @classmethod
    def from_config(cls, config: Any) -> "RetractionPolicy":
        """Build from a ``RetractionConfig`` (or ``None`` → the baseline)."""
        if config is None:
            return BASELINE_POLICY
        return cls(
            name=str(getattr(config, "policy_name", BASELINE_POLICY_NAME)),
            protected_source_positions=tuple(
                int(p) for p in getattr(config, "protected_source_positions", ()) or ()
            ),
            town_probability_threshold=getattr(config, "town_probability_threshold", None),
            country_probability_threshold=getattr(config, "country_probability_threshold", None),
        )

    @property
    def is_baseline(self) -> bool:
        return (
            not self.protected_source_positions
            and self.town_probability_threshold is None
            and self.country_probability_threshold is None
        )

    def protected_fields(self, source_fields: Sequence[str]) -> tuple[str, ...]:
        """The configured field names the policy protects, in field order."""
        fields = list(source_fields)
        return tuple(
            fields[pos - 1]
            for pos in sorted(set(self.protected_source_positions))
            if 1 <= pos <= len(fields)
        )

    @staticmethod
    def passes(probability: float | None, threshold: float | None) -> bool:
        """Strict gate: ``probability > threshold``. No threshold → always passes.

        A missing probability cannot clear a configured gate — the policy asks
        for evidence of confidence, and none was supplied.
        """
        if threshold is None:
            return True
        if probability is None:
            return False
        return float(probability) > float(threshold)


BASELINE_POLICY = RetractionPolicy()


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
    #: Policy audit. Under the baseline policy these carry their defaults; a
    #: challenger policy makes every eligibility decision visible here.
    policy_name: str = BASELINE_POLICY_NAME
    protected_source_fields: tuple[str, ...] = ()
    town_probability_gate: float | None = None
    country_probability_gate: float | None = None
    town_probability_gate_passed: bool = True
    country_probability_gate_passed: bool = True
    town_retraction_eligible: bool = False
    country_retraction_eligible: bool = False
    town_skip_reason: str | None = None
    country_skip_reason: str | None = None
    #: EXP-03: verified entity spans that were protected, and how many Town /
    #: Country occurrences they shielded. Empty/zero under earlier policies.
    protected_entity_spans: tuple[dict[str, Any], ...] = ()
    town_occurrences_protected_by_entity: int = 0
    country_occurrences_protected_by_entity: int = 0

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
            "policy_name": self.policy_name,
            "protected_source_fields": list(self.protected_source_fields),
            "town_probability_gate": self.town_probability_gate,
            "country_probability_gate": self.country_probability_gate,
            "town_probability_gate_passed": self.town_probability_gate_passed,
            "country_probability_gate_passed": self.country_probability_gate_passed,
            "town_retraction_eligible": self.town_retraction_eligible,
            "country_retraction_eligible": self.country_retraction_eligible,
            "town_skip_reason": self.town_skip_reason,
            "country_skip_reason": self.country_skip_reason,
            "protected_entity_spans": [dict(s) for s in self.protected_entity_spans],
            "town_occurrences_protected_by_entity": self.town_occurrences_protected_by_entity,
            "country_occurrences_protected_by_entity": self.country_occurrences_protected_by_entity,
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


def _excluded_token_indices(
    spans: Sequence[TokenSpan], protected: Sequence[tuple[int, int]] | None
) -> frozenset[int]:
    """Token indices whose characters overlap any protected span."""
    if not protected:
        return frozenset()
    excluded = set()
    for index, span in enumerate(spans):
        for start, end in protected:
            if span.start < end and span.end > start:
                excluded.add(index)
                break
    return frozenset(excluded)


def token_phrase_matches(
    text: str,
    phrase: str,
    *,
    restrict_to_trailing_tokens: int | None = None,
    protected_spans: Sequence[tuple[int, int]] | None = None,
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

    ``protected_spans`` are character ranges of ``text`` that may not be
    touched: a match that overlaps one is not a match. This is how a verified
    entity span keeps its town or country word while the same word elsewhere
    in the field stays eligible.
    """
    spans = token_spans(text or "")
    if not spans:
        return ()
    keys = [span.key for span in spans]
    needle = [
        unicodedata.normalize("NFKC", token.text).upper()
        for token in token_spans(phrase or "")
    ]
    return _matches(keys, needle, restrict_to_trailing_tokens,
                    excluded=_excluded_token_indices(spans, protected_spans))


def _matches(
    keys: Sequence[str],
    needle: Sequence[str],
    restrict_to_trailing_tokens: int | None,
    excluded: frozenset[int] = frozenset(),
) -> tuple[tuple[int, int], ...]:
    """Non-overlapping token-index matches of ``needle`` within ``keys``.

    A candidate match touching any ``excluded`` token index is skipped.
    """
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
        if (
            list(keys[index : index + len(needle)]) == list(needle)
            and end_index >= earliest_end
            and not any(i in excluded for i in range(index, end_index + 1))
        ):
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
    protected_spans: Sequence[tuple[int, int]] | None = None,
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
    excluded = _excluded_token_indices(spans, protected_spans)
    removals: list[tuple[int, int]] = []
    removed_forms: list[str] = []

    for phrase in phrases:
        needle = [
            unicodedata.normalize("NFKC", token.text).upper()
            for token in token_spans(phrase or "")
        ]
        found = _matches(keys, needle, restrict_to_trailing_tokens, excluded=excluded)
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
    town_probability: float | None = None,
    country_probability: float | None = None,
    policy: RetractionPolicy = BASELINE_POLICY,
    protected_spans: ProtectedSpans | None = None,
) -> RetractionResult:
    """Retract verified Town/Country evidence from one group's source columns.

    ``protected_spans`` (EXP-03) are verified entity character spans per field
    that retraction may not touch; an occurrence overlapping one is simply not
    eligible, while the same word elsewhere stays eligible.

    ``source_values`` is read only — the caller's dataframe is never mutated.
    The retracted combined address is rebuilt from the after-values with
    :func:`models.swft_tc.src.grouping.build_combined_address`, so it follows exactly
    the same joining and missing-field conventions as Pass 1.

    ``policy`` decides *eligibility*; it never changes what counts as verified.
    With the default :data:`BASELINE_POLICY` (no protected field, no gate) the
    outcome — values and comment — is byte-for-byte the production baseline.
    """
    before = {
        field_name: trim_field(source_values.get(field_name))
        for field_name in source_fields
    }
    protected = policy.protected_fields(source_fields)
    eligible_fields = [name for name in source_fields if name not in protected]
    span_map: dict[str, tuple[tuple[int, int], ...]] = {
        name: tuple((int(a), int(b)) for a, b in spans)
        for name, spans in (protected_spans or {}).items()
        if name in source_fields and spans
    }
    entity_span_records = tuple(
        {"source_field": name, "start": a, "end": b, "text": before[name][a:b]}
        for name, spans in span_map.items() for a, b in spans
    )

    town_verified = bool(town_exists and town not in {"", NO_TOWN})
    country_codes = (
        [code for code in country_value.split(",") if code]
        if country_exists and country_value not in {"", NO_COUNTRY}
        else []
    )
    # A comma-separated candidate set is unresolved by definition, and
    # `country_exists` is only ever True for a single resolved code — but guard
    # explicitly rather than relying on that invariant holding forever.
    country_verified = bool(country_codes) and len(country_codes) == 1

    town_gate_passed = policy.passes(town_probability, policy.town_probability_threshold)
    country_gate_passed = policy.passes(
        country_probability, policy.country_probability_threshold
    )

    town_skip: str | None = None
    if not town_verified:
        town_skip = SKIP_NOT_VERIFIED
    elif not town_gate_passed:
        town_skip = SKIP_PROBABILITY
    country_skip: str | None = None
    if not country_verified:
        country_skip = SKIP_NOT_VERIFIED
    elif not country_gate_passed:
        country_skip = SKIP_PROBABILITY

    audit = dict(
        policy_name=policy.name,
        protected_source_fields=protected,
        town_probability_gate=policy.town_probability_threshold,
        country_probability_gate=policy.country_probability_threshold,
        town_probability_gate_passed=town_gate_passed,
        country_probability_gate_passed=country_gate_passed,
        protected_entity_spans=entity_span_records,
    )

    # --- Town: at most ONE occurrence per GROUP, the right-most ELIGIBLE one ---
    # A Town can legitimately appear more than once in one address — once inside
    # an institution, building or branch name, and once as the locality itself
    # ("CITIGROUP CENTRE AUCKLAND AUCKLAND"). Removing every occurrence deletes
    # part of the organisation's name along with the location. Only one
    # occurrence is evidence of the locality, so only one is retracted.
    #
    # Which one is a deterministic positional choice, not a semantic guess: the
    # right-most standalone occurrence across the configured source fields in
    # configuration order, because the locality sits later in an address than a
    # descriptive prefix does. Protected fields never qualify, so under a policy
    # that protects line 1 the choice is made among the later lines only — and
    # if the Town occurs nowhere else, nothing is removed.
    town_occurrences_found = 0
    town_occurrences_protected_by_entity = 0
    town_target_field = ""
    if town_verified:
        for field_name in source_fields:
            value = before[field_name]
            if not value:
                continue
            occurrences = len(token_phrase_matches(value, town))
            if occurrences:
                town_occurrences_found += occurrences
                if field_name in eligible_fields:
                    unprotected = len(
                        token_phrase_matches(value, town, protected_spans=span_map.get(field_name))
                    )
                    town_occurrences_protected_by_entity += occurrences - unprotected
                    if unprotected:
                        town_target_field = field_name
    if town_skip is None:
        if town_occurrences_found == 0:
            town_skip = SKIP_NO_ELIGIBLE_OCCURRENCE
        elif not town_target_field:
            town_skip = (
                SKIP_PROTECTED_ENTITY_SPAN_ONLY
                if town_occurrences_protected_by_entity else SKIP_PROTECTED_ONLY
            )
    retract_town = town_skip is None

    # --- Country forms ------------------------------------------------------
    # Which textual country forms are eligible is decided ONCE, against the
    # combined address — the same text that produced `country_exists`. Deciding
    # it per field would be wrong: a token sitting mid-address can be in the
    # trailing window of its own short field, which is how "SUITE 5 IN TOWER"
    # would otherwise lose its preposition.
    combined_before = build_combined_address(
        [before[name] for name in source_fields], zero_is_missing=zero_is_missing
    )
    code = country_codes[0] if country_verified else ""
    if country_verified and iso_provider is not None:
        eligible_forms = list(iso_provider.matched_presence_forms(combined_before, code))
        code_is_ambiguous = iso_provider.is_ambiguous_alpha2(code)
        trailing_window = iso_provider.trailing_country_token_window
    else:
        eligible_forms = [code] if country_verified else []
        code_is_ambiguous = False
        trailing_window = 0

    # Unrestricted forms: full country names, and non-colliding codes.
    open_forms = [
        form for form in eligible_forms
        if not (code_is_ambiguous and form.upper() == code.upper())
    ]
    # A colliding code is only removable in trailing country position, and only
    # from the field that actually carries the address tail — and then only if
    # that field may be modified at all.
    restricted_code = (
        code if (code_is_ambiguous and code in eligible_forms) else ""
    )
    tail_field = _last_non_empty_field(before, source_fields)
    restricted_field = tail_field if tail_field in eligible_fields else ""

    def _country_occurrences(fields: Sequence[str], *, honour_spans: bool) -> int:
        count = 0
        for field_name in fields:
            value = before[field_name]
            if not value:
                continue
            spans = span_map.get(field_name) if honour_spans else None
            for form in open_forms:
                count += len(token_phrase_matches(value, form, protected_spans=spans))
            if restricted_code and field_name == tail_field:
                count += len(
                    token_phrase_matches(
                        value, restricted_code, restrict_to_trailing_tokens=trailing_window,
                        protected_spans=spans,
                    )
                )
        return count

    country_occurrences_protected_by_entity = 0
    if country_skip is None:
        found_anywhere = _country_occurrences(list(source_fields), honour_spans=False)
        found_eligible_fields = _country_occurrences(eligible_fields, honour_spans=False)
        found_eligible = _country_occurrences(eligible_fields, honour_spans=True)
        country_occurrences_protected_by_entity = found_eligible_fields - found_eligible
        if found_anywhere == 0:
            country_skip = SKIP_NO_ELIGIBLE_OCCURRENCE
        elif found_eligible == 0:
            country_skip = (
                SKIP_PROTECTED_ENTITY_SPAN_ONLY
                if country_occurrences_protected_by_entity else SKIP_PROTECTED_ONLY
            )
    retract_country = country_skip is None

    if not retract_town and not retract_country:
        result = RetractionResult(
            before=before,
            after=dict(before),
            combined_address_retracted=clean_address(combined_before),
            comment="",
            retracted_entities=(),
            town_occurrences_found=town_occurrences_found,
            town_retraction_eligible=False,
            country_retraction_eligible=False,
            town_skip_reason=town_skip,
            country_skip_reason=country_skip,
            town_occurrences_protected_by_entity=town_occurrences_protected_by_entity,
            country_occurrences_protected_by_entity=country_occurrences_protected_by_entity,
            **audit,
        )
        return _with_comment(result, policy, town, code, town_removed=False,
                             country_removed=False, removed_forms=())

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
        if not value or field_name not in eligible_fields:
            # Empty, or protected: copied through untouched.
            after[field_name] = value
            continue

        updated = value

        # Town first, so the single occurrence is chosen against the field text
        # exactly as it was written rather than against a country-stripped
        # remnant. Only the one field carrying the right-most eligible
        # occurrence is touched; every earlier occurrence stays.
        field_spans = span_map.get(field_name)
        if retract_town and field_name == town_target_field:
            updated, removed = remove_token_phrases(
                updated, [town], max_occurrences=1, prefer_last=True,
                protected_spans=field_spans,
            )
            for form in removed:
                _note(form)
                town_occurrences_removed += 1

        # Country keeps its rule: every verified occurrence goes, since a
        # country code repeated in the text is one piece of evidence stated
        # twice, not two separate facts.
        # Protected spans are character offsets into the ORIGINAL field value.
        # Once the Town has been cut out the offsets no longer line up, so when a
        # field carries both a protected span and a Town removal the Country
        # pass re-derives the spans by text: the protected phrases are located
        # again in the updated string.
        country_spans = _relocate_spans(updated, before[field_name], field_spans)
        if retract_country and open_forms:
            updated, removed = remove_token_phrases(
                updated, open_forms, protected_spans=country_spans
            )
            for form in removed:
                _note(form)

        if retract_country and restricted_code and field_name == restricted_field:
            country_spans = _relocate_spans(updated, before[field_name], field_spans)
            updated, removed = remove_token_phrases(
                updated, [restricted_code],
                restrict_to_trailing_tokens=trailing_window,
                protected_spans=country_spans,
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

    # An entity judged eligible whose text was then consumed by the other
    # entity's removal (overlapping spans) was not retracted on its own merits.
    if retract_country and not country_removed and country_skip is None:
        country_skip = SKIP_NO_ELIGIBLE_OCCURRENCE
    if retract_town and not town_removed and town_skip is None:
        town_skip = SKIP_NO_ELIGIBLE_OCCURRENCE

    result = RetractionResult(
        before=before,
        after=after,
        combined_address_retracted=clean_address(combined),
        comment="",
        retracted_entities=tuple(entities),
        removed_forms=tuple(removed_forms),
        town_occurrences_found=town_occurrences_found,
        town_occurrences_removed=town_occurrences_removed,
        town_retraction_eligible=retract_town,
        country_retraction_eligible=retract_country,
        town_skip_reason=town_skip,
        country_skip_reason=country_skip,
        town_occurrences_protected_by_entity=town_occurrences_protected_by_entity,
        country_occurrences_protected_by_entity=country_occurrences_protected_by_entity,
        **audit,
    )
    return _with_comment(result, policy, town, code, town_removed=town_removed,
                         country_removed=country_removed,
                         removed_forms=tuple(removed_forms))


def _relocate_spans(
    current: str, original: str, spans: Sequence[tuple[int, int]] | None
) -> tuple[tuple[int, int], ...] | None:
    """Re-find protected phrases in a field whose text has already been edited.

    Spans are recorded against the original field text. If nothing was removed
    the offsets still apply; otherwise each protected phrase is located again
    by token-safe matching in the current text (first occurrence). A phrase
    that can no longer be found protects nothing — it was never eligible text.
    """
    if not spans:
        return None
    if current == original:
        return tuple(spans)
    relocated: list[tuple[int, int]] = []
    tokens = token_spans(current)
    for start, end in spans:
        phrase = original[start:end]
        matches = token_phrase_matches(current, phrase)
        if matches:
            first, last = matches[0]
            relocated.append((tokens[first].start, tokens[last].end))
    return tuple(relocated) or None


def _with_comment(
    result: RetractionResult,
    policy: RetractionPolicy,
    town: str,
    code: str,
    *,
    town_removed: bool,
    country_removed: bool,
    removed_forms: tuple[str, ...],
) -> RetractionResult:
    """Attach the deterministic comment. Baseline wording is byte-identical."""
    from dataclasses import replace

    if policy.is_baseline and not result.protected_entity_spans:
        text = _comment(town_removed, country_removed, town, code, removed_forms)
    else:
        text = _policy_comment(result, policy, town, code, town_removed, country_removed)
    return replace(result, comment=text)


def _policy_comment(
    result: RetractionResult,
    policy: RetractionPolicy,
    town: str,
    code: str,
    town_removed: bool,
    country_removed: bool,
) -> str:
    """Comment for a non-baseline policy: says what happened and why, per entity.

    Never claims "not explicitly verified" when the real reason was a protected
    field, a probability gate, or an overlap with the other entity's removal.
    """
    protected = ", ".join(result.protected_source_fields) or "none"
    entity_texts = ", ".join(
        f"{s['source_field']}:'{s['text']}'" for s in result.protected_entity_spans
    ) or "none"
    parts: list[str] = []

    def _entity(label: str, value: str, removed: bool, skip: str | None,
                probability_gate: float | None, gate_passed: bool) -> str:
        shielded = (result.town_occurrences_protected_by_entity if label == "Town"
                    else result.country_occurrences_protected_by_entity)
        if removed:
            # EXP-02 wording is preserved byte-for-byte when no entity span is
            # involved; the entity sentence is added only when a span shielded text.
            if result.protected_source_fields and not shielded:
                return (f"Retracted {label}={value} from eligible address lines; "
                        f"occurrences in protected line(s) [{protected}] were preserved.")
            if shielded:
                extra = (f" occurrences in protected line(s) [{protected}] and"
                         if result.protected_source_fields else "")
                return (f"Retracted {label}={value} from eligible text;{extra} {shielded} "
                        f"occurrence(s) inside the verified entity span [{entity_texts}] were preserved.")
            return f"Retracted {label}={value}."
        if skip == SKIP_PROTECTED_ENTITY_SPAN_ONLY:
            return (f"{label} occurred only inside the verified entity span [{entity_texts}]; "
                    f"{label} was not retracted.")
        if skip == SKIP_NOT_VERIFIED:
            return (f"{label} was not explicitly verified in the input, so it was "
                    "retained only as a prediction.")
        if skip == SKIP_PROBABILITY:
            return (f"{label} was explicitly verified but its probability did not "
                    f"exceed the retraction threshold {probability_gate:.2f}; "
                    f"{label} was not retracted.")
        if skip == SKIP_PROTECTED_ONLY:
            return (f"{label} occurred only in protected line(s) [{protected}]; "
                    f"{label} was not retracted.")
        if skip == SKIP_NO_ELIGIBLE_OCCURRENCE:
            return (f"{label} was explicitly verified but no separately retractable "
                    f"occurrence remained; {label} was not retracted.")
        return f"{label} was not retracted."  # pragma: no cover - defensive

    parts.append(_entity("Town", town, town_removed, result.town_skip_reason,
                         result.town_probability_gate, result.town_probability_gate_passed))
    parts.append(_entity("Country", code or "", country_removed, result.country_skip_reason,
                         result.country_probability_gate, result.country_probability_gate_passed))
    return f"[{policy.name}] " + " ".join(parts)


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
