"""Deterministic verification, scenario selection, and the Composite Weighted Score.

This module is the business decision layer, and it deliberately sits *outside*
the LLM. Gemini contributes extraction, evidence spans, explicitness claims and
confidence estimates. Everything that decides an outcome — whether evidence
actually supports a claim, which policy scenario applies, which reliability
weights are used, what the routing score is — happens here, in Python, from
YAML-configured numbers. A prompt change cannot move a business decision.

Two rules are worth stating plainly because they are easy to get subtly wrong:

* **Presence beats assertion.** ``*_exists`` is computed from the address text
  on token boundaries. If the model claims explicit support that the text does
  not carry, the answer is ``False`` and the disagreement is recorded.
* **Ambiguity zeroes the country side.** More than one surviving candidate
  forces country probability and country weight to ``0.0``, which forces the
  composite to ``0.0`` and mandatory HITL — no candidate is ever picked
  arbitrarily to make the value scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .cleaning import contains_token_phrase
from .reference_data import Iso3166Provider
from .schemas import NO_COUNTRY, NO_TOWN, ExtractionResponse

__all__ = [
    "AMBIGUOUS_TOWN_INFERRED_SCENARIO",
    "FORCED_REVIEW_STATES",
    "HITL_AMBIGUOUS_COUNTRY",
    "HITL_AUTO_ACCEPT_CANDIDATE",
    "HITL_LOW_SCORE",
    "HITL_MANUAL_OVERRIDE",
    "HITL_PROCESSING_ERROR",
    "HITL_REFERENCE_CONFLICT",
    "HITL_STATE_NOT_EVALUATED",
    "HITL_STATE_PRECEDENCE",
    "HitlDecision",
    "NULL_SKIP_SCENARIO",
    "REASON_BELOW_THRESHOLD",
    "REASON_COUNTRY_AMBIGUOUS",
    "REASON_MANUAL_OVERRIDE",
    "REASON_PROCESSING_ERROR",
    "REASON_REFERENCE_CONFLICT",
    "REFERENCE_CONFLICT",
    "REFERENCE_CONSISTENT",
    "REFERENCE_MULTI_ANNOTATED",
    "REFERENCE_MULTI_ESCALATED",
    "REFERENCE_NOT_CHECKED",
    "REFERENCE_NOT_FOUND",
    "REFERENCE_NO_TOWN",
    "REFERENCE_SUPPLIED",
    "REQUIRED_SCENARIOS",
    "ScoreResult",
    "VerifiedExtraction",
    "determine_hitl_decision",
    "error_result",
    "evaluate",
    "null_hitl_decision",
    "null_result",
    "score",
    "select_scenario",
    "verify_extraction",
]

#: The six scenarios defined by SCORING_SPEC.md. `settings.ScoringConfig`
#: refuses to load a configuration that omits any of them.
REQUIRED_SCENARIOS: tuple[str, ...] = (
    "both_explicit",
    "country_explicit_town_inferred",
    "town_explicit_country_inferred",
    "town_explicit_country_ambiguous",
    "neither_explicit_both_inferred",
    "no_defensible_prediction",
)

#: Documented extension. The source matrix has no row for "country ambiguous
#: AND town not explicitly supported". Reusing `town_explicit_country_ambiguous`
#: there would assert a verified explicitness that does not exist, so this
#: scenario is configured separately in config.yaml. The composite score is 0.0
#: either way; the distinct name keeps the audit trail truthful. Falls back to
#: `no_defensible_prediction` if a configuration omits it.
AMBIGUOUS_TOWN_INFERRED_SCENARIO = "town_inferred_country_ambiguous"

#: Audit-only label for rows short-circuited before any model call. It has no
#: weights entry: the null path never consults the weight matrix.
NULL_SKIP_SCENARIO = "null_skip"


#: Outcomes of the Town/Country reference check. Audit/metrics only — none of
#: these become production CSV columns.
REFERENCE_NOT_CHECKED = "not_checked"
REFERENCE_NO_TOWN = "no_town_predicted"
REFERENCE_NOT_FOUND = "reference_not_found"
REFERENCE_CONSISTENT = "consistent"
REFERENCE_SUPPLIED = "supplied_by_reference"
REFERENCE_MULTI_ESCALATED = "multi_country_escalated"
REFERENCE_MULTI_ANNOTATED = "multi_country_annotated"
REFERENCE_CONFLICT = "conflict"


@dataclass(frozen=True)
class VerifiedExtraction:
    """A model response after Python-side verification and normalization.

    The fields here are what the CSV and the scoring engine consume. The
    original model claims survive in :attr:`notes` and in the cache/audit
    payload, never silently overwritten.
    """

    town: str
    country_candidates: tuple[str, ...]
    country_value: str
    town_exists: bool
    country_exists: bool
    country_ambiguous: bool
    town_probability: float
    country_probability: float
    country_name_value: str = NO_COUNTRY
    rationale_town: str = ""
    rationale_country: str = ""
    reference_status: str = REFERENCE_NOT_CHECKED
    reference_codes: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_town(self) -> bool:
        return self.town not in {"", NO_TOWN}

    @property
    def has_country(self) -> bool:
        return bool(self.country_candidates)

    @property
    def reference_conflict(self) -> bool:
        return self.reference_status == REFERENCE_CONFLICT


@dataclass(frozen=True)
class ScoreResult:
    """The deterministic routing decision for one row/group instance."""

    scenario: str
    town_weight: float
    country_weight: float
    town_probability: float
    country_probability: float
    adjusted_town_score: float
    adjusted_country_score: float
    composite_weighted_score: float
    needs_hitl: bool

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "town_weight": self.town_weight,
            "country_weight": self.country_weight,
            "town_probability": self.town_probability,
            "country_probability": self.country_probability,
            "adjusted_town_score": self.adjusted_town_score,
            "adjusted_country_score": self.adjusted_country_score,
            "composite_weighted_score": self.composite_weighted_score,
            "needs_hitl": self.needs_hitl,
        }


def verify_extraction(
    response: ExtractionResponse,
    cleaned_address: str,
    *,
    iso_provider: Iso3166Provider | None = None,
    town_country_provider: Any = None,
    separator: str = ",",
    candidate_sort: str = "alphabetical",
    ambiguous_country_probability_override: float = 0.0,
    town_country_ambiguity_policy: str = "escalate",
    town_country_max_candidates: int = 0,
) -> VerifiedExtraction:
    """Verify a model response against the address text it was derived from.

    Steps, in order:

    1. drop country candidates that are not valid ISO 3166-1 alpha-2 codes;
    2. verify the predicted town occurs in the address on token boundaries;
    3. verify each candidate country via ISO code or approved name alias;
    4. if several candidates remain but exactly one is explicitly present in
       the text, the text has resolved the ambiguity — collapse to it;
    5. cross-check the predicted town against the Town/Country reference;
    6. order the survivors deterministically, expand the country names from the
       ISO layer, and derive the final probabilities.

    Step 5 is a *validation* signal, never a silent rewrite. Explicit country
    evidence in the address always wins; the reference can only confirm it,
    flag a conflict, fill a gap the model left empty, or — when the address
    carries no explicit country at all and the town genuinely spans several
    countries — escalate the result to unresolved ambiguity.
    """
    notes: list[str] = []

    # 1. Candidate validation -------------------------------------------------
    candidates = list(response.country_candidates)
    if iso_provider is not None:
        invalid = iso_provider.invalid_codes(candidates)
        if invalid:
            notes.append(f"dropped_invalid_iso_codes:{'|'.join(invalid)}")
            candidates = [c for c in candidates if iso_provider.is_valid_alpha2(c)]

    # 2. Town presence --------------------------------------------------------
    town = response.town if response.has_town else NO_TOWN
    town_exists = bool(town != NO_TOWN and contains_token_phrase(cleaned_address, town))
    if response.town_is_explicit and not town_exists:
        # The substring trap lands here: a model asserting RONA for AERONAUTICA
        # claims explicit support the token-boundary check refuses to grant.
        notes.append("town_explicit_claim_unverified")
    elif town_exists and not response.town_is_explicit:
        notes.append("town_present_though_model_marked_inferred")

    # 3. Country presence -----------------------------------------------------
    present = tuple(
        code for code in candidates if _country_present(code, cleaned_address, iso_provider)
    )
    if response.country_is_explicit and not present:
        notes.append("country_explicit_claim_unverified")

    # 4. Text-resolved ambiguity ---------------------------------------------
    if len(candidates) > 1 and len(present) == 1:
        notes.append(f"ambiguity_resolved_by_explicit_text_evidence:{present[0]}")
        candidates = list(present)

    # 5. Town/Country reference cross-check -----------------------------------
    reference_status = REFERENCE_NOT_CHECKED
    reference_codes: tuple[str, ...] = ()
    if town_country_provider is not None:
        candidates, reference_status, reference_codes = _apply_town_country_reference(
            town=town,
            candidates=candidates,
            present=present,
            provider=town_country_provider,
            policy=town_country_ambiguity_policy,
            notes=notes,
        )

    # 6. Ordering, truncation, and finalization -------------------------------
    ordered = _order_candidates(candidates, candidate_sort)
    if town_country_max_candidates and len(ordered) > town_country_max_candidates:
        notes.append(
            f"country_candidates_truncated:{len(ordered)}->{town_country_max_candidates}"
        )
        ordered = ordered[:town_country_max_candidates]

    ambiguous = len(ordered) > 1
    country_value = separator.join(ordered) if ordered else NO_COUNTRY
    country_name_value = _expand_country_names(ordered, iso_provider, separator)

    country_exists = bool(len(ordered) == 1 and ordered[0] in present)

    town_probability = float(response.town_model_confidence) if town != NO_TOWN else 0.0
    if not ordered:
        country_probability = 0.0
    elif ambiguous:
        # Mandatory override: an unresolved candidate set carries no usable
        # country confidence, whatever the model estimated.
        country_probability = float(ambiguous_country_probability_override)
        notes.append("country_probability_overridden_for_ambiguity")
    else:
        country_probability = float(response.country_model_confidence)

    return VerifiedExtraction(
        town=town,
        country_candidates=ordered,
        country_value=country_value,
        country_name_value=country_name_value,
        town_exists=town_exists,
        country_exists=country_exists,
        country_ambiguous=ambiguous,
        town_probability=town_probability,
        country_probability=country_probability,
        rationale_town=response.town_rationale,
        rationale_country=response.country_rationale,
        reference_status=reference_status,
        reference_codes=reference_codes,
        notes=tuple(notes),
    )


def _apply_town_country_reference(
    *,
    town: str,
    candidates: list[str],
    present: tuple[str, ...],
    provider: Any,
    policy: str,
    notes: list[str],
) -> tuple[list[str], str, tuple[str, ...]]:
    """Cross-check the predicted town against the Town/Country reference.

    Returns ``(candidates, status, reference_codes)``. The precedence order is
    deliberate: explicit address evidence first, then a gap fill, then
    escalation. A town missing from the reference is a reference miss, not an
    extraction failure.
    """
    if town == NO_TOWN:
        return candidates, REFERENCE_NO_TOWN, ()

    reference_codes = tuple(provider.lookup_country_codes(town))

    if not reference_codes:
        notes.append("reference_not_found")
        return candidates, REFERENCE_NOT_FOUND, ()

    if present:
        # The address states a country outright. That beats the reference; the
        # reference only gets to agree or raise a flag.
        unsupported = sorted(code for code in present if code not in reference_codes)
        if unsupported:
            notes.append("reference_conflict:" + "|".join(unsupported))
            return candidates, REFERENCE_CONFLICT, reference_codes
        return candidates, REFERENCE_CONSISTENT, reference_codes

    if len(reference_codes) == 1:
        single = reference_codes[0]
        if not candidates:
            # The model found no defensible country; the reference fills the gap.
            # Country probability still comes from the model, so this surfaces a
            # suggestion for review rather than manufacturing confidence.
            notes.append(f"country_supplied_by_reference:{single}")
            return [single], REFERENCE_SUPPLIED, reference_codes
        if set(candidates) == {single}:
            return candidates, REFERENCE_CONSISTENT, reference_codes
        notes.append(
            "reference_conflict:" + "|".join(sorted(set(candidates) - {single}))
        )
        return candidates, REFERENCE_CONFLICT, reference_codes

    # Several reference countries and no explicit country evidence in the text.
    if policy == "escalate":
        merged = sorted(set(candidates) | set(reference_codes))
        notes.append("reference_multi_country_escalated:" + "|".join(reference_codes))
        return merged, REFERENCE_MULTI_ESCALATED, reference_codes

    notes.append("reference_multi_country:" + "|".join(reference_codes))
    return candidates, REFERENCE_MULTI_ANNOTATED, reference_codes


def _expand_country_names(
    codes: Sequence[str], iso_provider: Iso3166Provider | None, separator: str
) -> str:
    """Deterministically expand ISO codes to country names, one for one.

    The name string always has exactly as many elements as the code string, in
    the same order, so downstream consumers can split both on the separator and
    zip them. Any separator character occurring *inside* a country name is
    folded to a space first — several ISO short names are inverted forms such as
    "Taiwan, Province of China", and one of those would otherwise silently add a
    phantom element and break the alignment contract. Without an ISO provider
    the codes stand in for their own names rather than the column going blank.
    """
    if not codes:
        return NO_COUNTRY
    if iso_provider is None:
        return separator.join(codes)
    return separator.join(
        _strip_separator(name, separator) for name in iso_provider.country_names(codes)
    )


def _strip_separator(name: str, separator: str) -> str:
    if separator and separator in name:
        return " ".join(name.replace(separator, " ").split())
    return name


def select_scenario(
    verified: VerifiedExtraction, *, available_scenarios: Iterable[str] = ()
) -> str:
    """Choose the policy scenario from *verified* explicitness and ambiguity.

    Never from the model's own scenario interpretation.
    """
    known = set(available_scenarios)

    if not verified.has_town and not verified.has_country:
        return "no_defensible_prediction"

    if verified.country_ambiguous:
        if verified.town_exists:
            return "town_explicit_country_ambiguous"
        if AMBIGUOUS_TOWN_INFERRED_SCENARIO in known:
            return AMBIGUOUS_TOWN_INFERRED_SCENARIO
        return "no_defensible_prediction"

    if not verified.has_town or not verified.has_country:
        # One side is missing, so the composite is 0.0 regardless of weights.
        # There is no partial-credit scenario in the policy matrix.
        return "no_defensible_prediction"

    if verified.town_exists and verified.country_exists:
        return "both_explicit"
    if verified.country_exists:
        return "country_explicit_town_inferred"
    if verified.town_exists:
        return "town_explicit_country_inferred"
    return "neither_explicit_both_inferred"


def score(
    verified: VerifiedExtraction,
    scoring_config: Any,
) -> ScoreResult:
    """Apply configured reliability weights and compute the composite score.

    ``composite = (town_probability x town_weight) x (country_probability x country_weight)``
    """
    scenario = select_scenario(verified, available_scenarios=scoring_config.rules.keys())
    weights = scoring_config.weights_for(scenario)

    town_weight = float(weights.town_weight)
    country_weight = float(weights.country_weight)

    if verified.country_ambiguous:
        # Belt and braces: the ambiguous scenarios already configure 0.00, but
        # the rule is mandatory policy, not a property of a YAML value someone
        # might edit.
        country_weight = 0.0

    adjusted_town = verified.town_probability * town_weight
    adjusted_country = verified.country_probability * country_weight
    composite = adjusted_town * adjusted_country

    needs_hitl = composite < float(scoring_config.hitl_threshold)
    if verified.country_ambiguous and scoring_config.force_ambiguous_country_to_hitl:
        needs_hitl = True
    if verified.reference_conflict:
        # The model and the deterministic reference disagree. Neither is
        # overwritten; a human decides.
        needs_hitl = True

    return ScoreResult(
        scenario=scenario,
        town_weight=town_weight,
        country_weight=country_weight,
        town_probability=verified.town_probability,
        country_probability=verified.country_probability,
        adjusted_town_score=adjusted_town,
        adjusted_country_score=adjusted_country,
        composite_weighted_score=composite,
        needs_hitl=needs_hitl,
    )


def evaluate(
    response: ExtractionResponse,
    cleaned_address: str,
    scoring_config: Any,
    *,
    iso_provider: Iso3166Provider | None = None,
    town_country_provider: Any = None,
    separator: str = ",",
    candidate_sort: str = "alphabetical",
    town_country_ambiguity_policy: str = "escalate",
    town_country_max_candidates: int = 0,
) -> tuple[VerifiedExtraction, ScoreResult]:
    """Verify then score a single model response."""
    verified = verify_extraction(
        response,
        cleaned_address,
        iso_provider=iso_provider,
        town_country_provider=town_country_provider,
        separator=separator,
        candidate_sort=candidate_sort,
        ambiguous_country_probability_override=float(
            scoring_config.ambiguous_country_probability_override
        ),
        town_country_ambiguity_policy=town_country_ambiguity_policy,
        town_country_max_candidates=town_country_max_candidates,
    )
    return verified, score(verified, scoring_config)


def null_result() -> tuple[VerifiedExtraction, ScoreResult]:
    """The fixed result for an empty combined address. No model call is involved.

    Uses the audit-only ``null_skip`` scenario and hard zeros rather than a
    weight-matrix lookup: this row was never a model conclusion.
    """
    verified = VerifiedExtraction(
        town=NO_TOWN,
        country_candidates=(),
        country_value=NO_COUNTRY,
        country_name_value=NO_COUNTRY,
        town_exists=False,
        country_exists=False,
        country_ambiguous=False,
        town_probability=0.0,
        country_probability=0.0,
        rationale_town="",
        rationale_country="",
        notes=("empty_combined_address_no_model_call",),
    )
    result = ScoreResult(
        scenario=NULL_SKIP_SCENARIO,
        town_weight=0.0,
        country_weight=0.0,
        town_probability=0.0,
        country_probability=0.0,
        adjusted_town_score=0.0,
        adjusted_country_score=0.0,
        composite_weighted_score=0.0,
        needs_hitl=False,
    )
    return verified, result


def error_result(reason: str) -> tuple[VerifiedExtraction, ScoreResult]:
    """Safe neutral values for a row whose model call failed after retries.

    Deliberately *not* presented as a model conclusion: the failure is recorded
    in ``processing_errors.csv`` and in run metrics, the row is routed to HITL,
    and the reason travels in the audit notes.
    """
    verified = VerifiedExtraction(
        town=NO_TOWN,
        country_candidates=(),
        country_value=NO_COUNTRY,
        country_name_value=NO_COUNTRY,
        town_exists=False,
        country_exists=False,
        country_ambiguous=False,
        town_probability=0.0,
        country_probability=0.0,
        rationale_town="",
        rationale_country="",
        notes=(f"extraction_failed:{reason}",),
    )
    result = ScoreResult(
        scenario="extraction_error",
        town_weight=0.0,
        country_weight=0.0,
        town_probability=0.0,
        country_probability=0.0,
        adjusted_town_score=0.0,
        adjusted_country_score=0.0,
        composite_weighted_score=0.0,
        needs_hitl=True,
    )
    return verified, result


# ---------------------------------------------------------------------------
# HITL routing decision
# ---------------------------------------------------------------------------

#: The only values `HITL_state_group_<id>` may take. Gemini never chooses one:
#: this is deterministic Python policy end to end.
HITL_AUTO_ACCEPT_CANDIDATE = "AUTO_ACCEPT_CANDIDATE"
HITL_LOW_SCORE = "HITL_LOW_SCORE"
HITL_AMBIGUOUS_COUNTRY = "HITL_AMBIGUOUS_COUNTRY"
HITL_REFERENCE_CONFLICT = "HITL_REFERENCE_CONFLICT"
HITL_PROCESSING_ERROR = "HITL_PROCESSING_ERROR"
HITL_MANUAL_OVERRIDE = "HITL_MANUAL_OVERRIDE"

#: Precedence, strongest control first. Several conditions can hold at once —
#: an ambiguous country always scores 0.0 and is therefore *also* below any
#: sane threshold — so the primary state names the root cause rather than the
#: symptom. Every applicable reason still travels in `contributing_reasons`.
HITL_STATE_PRECEDENCE: tuple[str, ...] = (
    HITL_PROCESSING_ERROR,
    HITL_MANUAL_OVERRIDE,
    HITL_AMBIGUOUS_COUNTRY,
    HITL_REFERENCE_CONFLICT,
    HITL_LOW_SCORE,
    HITL_AUTO_ACCEPT_CANDIDATE,
)

#: States reached because a control overrode (or independently required) review,
#: rather than because the score fell short. Makes it auditable *why* a case is
#: with a human.
FORCED_REVIEW_STATES: frozenset[str] = frozenset(
    {
        HITL_PROCESSING_ERROR,
        HITL_MANUAL_OVERRIDE,
        HITL_AMBIGUOUS_COUNTRY,
        HITL_REFERENCE_CONFLICT,
    }
)

#: Machine-readable contributing-reason tokens, in precedence order.
REASON_PROCESSING_ERROR = "processing_error"
REASON_MANUAL_OVERRIDE = "manual_override"
REASON_COUNTRY_AMBIGUOUS = "country_ambiguous"
REASON_REFERENCE_CONFLICT = "reference_conflict"
REASON_BELOW_THRESHOLD = "below_threshold"

#: The null-skip state: a group short-circuited before any model call was never
#: evaluated, so it is neither auto-accepted nor routed for review. Blank, not
#: `AUTO_ACCEPT_CANDIDATE` — calling it a candidate would imply a judgement that
#: was never made.
HITL_STATE_NOT_EVALUATED = ""


@dataclass(frozen=True)
class HitlDecision:
    """The final, deterministic human-review decision for one group instance."""

    required: bool
    state: str
    reason: str
    threshold: float
    composite_weighted_score: float
    forced_review: bool
    contributing_reasons: tuple[str, ...] = ()
    manual_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "state": self.state,
            "reason": self.reason,
            "configured_threshold": self.threshold,
            "composite_weighted_score": self.composite_weighted_score,
            "forced_review": self.forced_review,
            "contributing_reasons": list(self.contributing_reasons),
            "manual_override": self.manual_override,
        }


def determine_hitl_decision(
    verified: VerifiedExtraction,
    score_result: ScoreResult,
    scoring_config: Any,
    *,
    extraction_error: bool = False,
    manual_override: bool = False,
    manual_override_reason: str = "",
) -> HitlDecision:
    """Decide whether a group instance needs a human, and say why.

    The threshold is **not** the only rule. Review is required when the
    composite score falls below ``scoring.hitl_threshold`` *or* when any
    forced-review control applies. A case with a score comfortably above the
    threshold still goes to a human if deterministic reference data disagrees
    with the prediction — the number never overrules the control.

    Precedence is :data:`HITL_STATE_PRECEDENCE`. Comparison against the
    threshold uses full precision; rounding happens only when composing the
    human-readable reason.

    No model call. Gemini chooses neither the state nor the wording.
    """
    threshold = float(scoring_config.hitl_threshold)
    composite = float(score_result.composite_weighted_score)
    below_threshold = composite < threshold

    ambiguous = bool(verified.country_ambiguous) and bool(
        getattr(scoring_config, "force_ambiguous_country_to_hitl", True)
    )
    conflict = bool(verified.reference_conflict)

    # Every applicable reason, in precedence order, regardless of which one
    # becomes the primary state.
    reasons: list[str] = []
    if extraction_error:
        reasons.append(REASON_PROCESSING_ERROR)
    if manual_override:
        reasons.append(REASON_MANUAL_OVERRIDE)
    if ambiguous:
        reasons.append(REASON_COUNTRY_AMBIGUOUS)
    if conflict:
        reasons.append(REASON_REFERENCE_CONFLICT)
    if below_threshold:
        reasons.append(REASON_BELOW_THRESHOLD)

    if extraction_error:
        state = HITL_PROCESSING_ERROR
    elif manual_override:
        state = HITL_MANUAL_OVERRIDE
    elif ambiguous:
        state = HITL_AMBIGUOUS_COUNTRY
    elif conflict:
        state = HITL_REFERENCE_CONFLICT
    elif below_threshold:
        state = HITL_LOW_SCORE
    else:
        state = HITL_AUTO_ACCEPT_CANDIDATE

    return HitlDecision(
        required=state != HITL_AUTO_ACCEPT_CANDIDATE,
        state=state,
        reason=_hitl_reason(
            state, composite, threshold, below_threshold, manual_override_reason
        ),
        threshold=threshold,
        composite_weighted_score=composite,
        forced_review=state in FORCED_REVIEW_STATES,
        contributing_reasons=tuple(reasons),
        manual_override=bool(manual_override),
    )


def null_hitl_decision(scoring_config: Any) -> HitlDecision:
    """The decision for a null-skipped group: no judgement was ever made.

    Blank state and reason, `required=False`. Deliberately *not*
    `AUTO_ACCEPT_CANDIDATE` — the model never saw this group, so calling it a
    candidate would assert a conclusion nobody reached.
    """
    return HitlDecision(
        required=False,
        state=HITL_STATE_NOT_EVALUATED,
        reason="",
        threshold=float(scoring_config.hitl_threshold),
        composite_weighted_score=0.0,
        forced_review=False,
        contributing_reasons=(),
        manual_override=False,
    )


def _hitl_reason(
    state: str,
    composite: float,
    threshold: float,
    below_threshold: bool,
    manual_override_reason: str,
) -> str:
    """One short deterministic sentence. Never model-written."""
    score_text = _format_score(composite)
    threshold_text = _format_score(threshold)

    if state == HITL_PROCESSING_ERROR:
        return (
            "Extraction failed after configured retries; no valid model result "
            "was produced."
        )
    if state == HITL_MANUAL_OVERRIDE:
        return manual_override_reason or (
            "Manual business override requires human review."
        )
    if state == HITL_AMBIGUOUS_COUNTRY:
        return (
            "Multiple Country candidates remain unresolved; mandatory human "
            "review is required."
        )
    if state == HITL_REFERENCE_CONFLICT:
        if below_threshold:
            return (
                "Predicted Country conflicts with deterministic reference data; "
                "human review is required."
            )
        # Worth spelling out: the score passed and the control still won.
        return (
            "Predicted Country conflicts with deterministic reference data; human "
            f"review is required despite Composite Weighted Score {score_text} "
            f"meeting configured threshold {threshold_text}."
        )
    if state == HITL_LOW_SCORE:
        return (
            f"Composite Weighted Score {score_text} is below configured HITL "
            f"threshold {threshold_text}."
        )
    return (
        f"Composite Weighted Score {score_text} meets configured HITL threshold "
        f"{threshold_text} and no forced-review condition is present."
    )


def _format_score(value: float) -> str:
    """Render a score for display at 4 decimals, trimmed to at least 2.

    ``0.80`` reads as "0.80" rather than "0.8", while ``0.7999`` keeps its
    precision instead of rounding to "0.80" and making the sentence
    "0.80 is below configured HITL threshold 0.80" look self-contradictory.
    """
    text = f"{float(value):.4f}"
    whole, _, fraction = text.partition(".")
    fraction = fraction.rstrip("0")
    fraction = fraction.ljust(2, "0")
    return f"{whole}.{fraction}"


def _country_present(
    code: str, address: str, iso_provider: Iso3166Provider | None
) -> bool:
    if iso_provider is not None:
        return iso_provider.country_is_present(address, code)
    # Without an ISO dataset only the bare code can be checked, and only on
    # token boundaries.
    return contains_token_phrase(address, code)


def _order_candidates(candidates: Sequence[str], sort_mode: str) -> tuple[str, ...]:
    deduped = tuple(dict.fromkeys(code.upper() for code in candidates))
    if sort_mode == "model_order":
        return deduped
    return tuple(sorted(deduped))
