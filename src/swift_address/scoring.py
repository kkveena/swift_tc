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
    "NULL_SKIP_SCENARIO",
    "REQUIRED_SCENARIOS",
    "ScoreResult",
    "VerifiedExtraction",
    "error_result",
    "evaluate",
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
    rationale_town: str = ""
    rationale_country: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_town(self) -> bool:
        return self.town not in {"", NO_TOWN}

    @property
    def has_country(self) -> bool:
        return bool(self.country_candidates)


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
    separator: str = ",",
    candidate_sort: str = "alphabetical",
    ambiguous_country_probability_override: float = 0.0,
) -> VerifiedExtraction:
    """Verify a model response against the address text it was derived from.

    Steps, in order:

    1. drop country candidates that are not valid ISO 3166-1 alpha-2 codes;
    2. verify the predicted town occurs in the address on token boundaries;
    3. verify each candidate country via ISO code or approved name alias;
    4. if several candidates remain but exactly one is explicitly present in
       the text, the text has resolved the ambiguity — collapse to it;
    5. order the survivors deterministically and derive the final probabilities.
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

    # 5. Ordering and finalization -------------------------------------------
    ordered = _order_candidates(candidates, candidate_sort)
    ambiguous = len(ordered) > 1
    country_value = separator.join(ordered) if ordered else NO_COUNTRY

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
        town_exists=town_exists,
        country_exists=country_exists,
        country_ambiguous=ambiguous,
        town_probability=town_probability,
        country_probability=country_probability,
        rationale_town=response.town_rationale,
        rationale_country=response.country_rationale,
        notes=tuple(notes),
    )


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
    separator: str = ",",
    candidate_sort: str = "alphabetical",
) -> tuple[VerifiedExtraction, ScoreResult]:
    """Verify then score a single model response."""
    verified = verify_extraction(
        response,
        cleaned_address,
        iso_provider=iso_provider,
        separator=separator,
        candidate_sort=candidate_sort,
        ambiguous_country_probability_override=float(
            scoring_config.ambiguous_country_probability_override
        ),
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
