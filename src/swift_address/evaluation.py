"""Ground-truth correctness labels and cross-entropy evaluation.

Two ideas live here, and keeping them apart is the whole point of the module.

``predicted_town_exists`` / ``predicted_country_exists`` (produced in
:mod:`swift_address.scoring`) answer **"is the predicted value explicitly
present in the input address text?"**. Their meaning is unchanged.

``town_exists_ok`` / ``country_exists_ok`` (produced here) answer a different
question: **"when independent deterministic evidence is available, was the
prediction correct?"** They are *nullable* booleans:

===========  ===============================================================
``True``     evidence is available and supports the prediction
``False``    evidence is available and contradicts the prediction
``None``     evidence is insufficient, unavailable, unresolved, or ambiguous
===========  ===============================================================

"Not found in the reference" is ``None``, never ``False``. Unknown is not the
same thing as incorrect, and collapsing the two would silently manufacture
model errors out of reference-coverage gaps — and then feed them into the loss.

Cross-entropy scores how well the model's *confidence* matched those labels. It
is an evaluation metric, not a routing metric:

* Composite Weighted Score — operational HITL routing — **higher is better**.
* Cross-entropy — confidence calibration against ground truth — **lower is
  better**.

Gemini never computes any of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .schemas import NO_COUNTRY, NO_TOWN
from .scoring import (
    REFERENCE_CONFLICT,
    REFERENCE_MULTI_ANNOTATED,
    REFERENCE_MULTI_ESCALATED,
    REFERENCE_NOT_CHECKED,
    REFERENCE_NOT_FOUND,
    VerifiedExtraction,
)

__all__ = [
    "CROSS_ENTROPY_EPSILON",
    "CrossEntropyResult",
    "GroundTruth",
    "STATUS_AMBIGUOUS",
    "STATUS_BOTH_GROUNDED",
    "STATUS_COUNTRY_ONLY",
    "STATUS_NOT_AVAILABLE",
    "STATUS_REFERENCE_NOT_FOUND",
    "STATUS_TOWN_ONLY",
    "binary_cross_entropy",
    "compute_cross_entropy",
    "evaluate_ground_truth",
    "null_cross_entropy",
    "null_ground_truth",
]

#: Clipping bound for probabilities. Without it a confident-and-correct p=1.0
#: gives log(0) = -inf on the wrong side of the loss.
CROSS_ENTROPY_EPSILON = 1e-6

STATUS_BOTH_GROUNDED = "both_grounded"
STATUS_TOWN_ONLY = "town_only"
STATUS_COUNTRY_ONLY = "country_only"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_AMBIGUOUS = "ambiguous_ground_truth"
STATUS_REFERENCE_NOT_FOUND = "reference_not_found"

#: Statuses that mean "only one component was grounded".
PARTIAL_STATUSES = frozenset({STATUS_TOWN_ONLY, STATUS_COUNTRY_ONLY})


@dataclass(frozen=True)
class GroundTruth:
    """Nullable correctness labels plus why each one came out that way."""

    town_exists_ok: bool | None = None
    country_exists_ok: bool | None = None
    town_basis: str = "no_evidence"
    country_basis: str = "no_evidence"

    @property
    def town_available(self) -> bool:
        return self.town_exists_ok is not None

    @property
    def country_available(self) -> bool:
        return self.country_exists_ok is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "town_exists_ok": self.town_exists_ok,
            "country_exists_ok": self.country_exists_ok,
            "town_basis": self.town_basis,
            "country_basis": self.country_basis,
        }


@dataclass(frozen=True)
class CrossEntropyResult:
    """Per-component and combined binary cross-entropy."""

    town_ground_truth_available: bool = False
    town_correct: bool | None = None
    town_probability: float | None = None
    town_cross_entropy: float | None = None
    country_ground_truth_available: bool = False
    country_correct: bool | None = None
    country_probability: float | None = None
    country_cross_entropy: float | None = None
    group_cross_entropy: float | None = None
    status: str = STATUS_NOT_AVAILABLE

    @property
    def is_partial(self) -> bool:
        return self.status in PARTIAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "town_ground_truth_available": self.town_ground_truth_available,
            "town_correct": self.town_correct,
            "town_probability": _round(self.town_probability),
            "town_cross_entropy": _round(self.town_cross_entropy),
            "country_ground_truth_available": self.country_ground_truth_available,
            "country_correct": self.country_correct,
            "country_probability": _round(self.country_probability),
            "country_cross_entropy": _round(self.country_cross_entropy),
            "group_cross_entropy": _round(self.group_cross_entropy),
            "status": self.status,
        }


def binary_cross_entropy(
    correct: bool, probability: float, *, epsilon: float = CROSS_ENTROPY_EPSILON
) -> float:
    """Binary cross-entropy (log loss) of a confidence against a correctness label.

    ``BCE(y, p) = -(y * log(p) + (1 - y) * log(1 - p))`` with ``p`` clipped into
    ``[epsilon, 1 - epsilon]``.

    Confident and right is cheap; confident and wrong is expensive::

        binary_cross_entropy(True,  0.95)  ->  0.051293
        binary_cross_entropy(False, 0.95)  ->  2.995732
    """
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    if correct:
        return -math.log(probability)
    return -math.log(1.0 - probability)


def evaluate_ground_truth(
    verified: VerifiedExtraction,
    *,
    town_country_provider: Any = None,
    extraction_failed: bool = False,
) -> GroundTruth:
    """Derive nullable correctness labels from deterministic evidence only.

    **Town.** ``True`` needs two independent things to agree: the predicted town
    is explicitly present in the address on token boundaries (evidence that
    comes from the *input*, not from the model), and the normalized town exists
    in the configured Town/Country reference. Neither alone is enough — looking
    the model's own answer up in a gazetteer would be circular, and text
    presence alone does not prove the span is a town. ``False`` requires
    positive contradiction: the model asserted an explicit town that
    token-boundary verification proves is not there. Everything else is
    ``None``.

    **Country.** ``True`` when the predicted code is explicitly supported in the
    address, or when a reference-known town resolves to exactly one country and
    the prediction is that country. ``False`` when deterministic reference truth
    exists and the prediction contradicts it. A town spanning several countries
    with no explicit resolution is ``None`` — unresolved, not wrong.
    """
    if extraction_failed:
        return GroundTruth(
            town_basis="extraction_error", country_basis="extraction_error"
        )

    town_ok, town_basis = _town_label(verified, town_country_provider)
    country_ok, country_basis = _country_label(verified, town_country_provider)
    return GroundTruth(
        town_exists_ok=town_ok,
        country_exists_ok=country_ok,
        town_basis=town_basis,
        country_basis=country_basis,
    )


def _town_label(
    verified: VerifiedExtraction, provider: Any
) -> tuple[bool | None, str]:
    if verified.town in {"", NO_TOWN}:
        return None, "no_town_predicted"

    # Positive contradiction: the model claimed explicit support the text does
    # not carry. This is the AERONAUTICA -> RONA case.
    if "town_explicit_claim_unverified" in verified.notes:
        return False, "explicit_claim_refuted_by_text"

    if not verified.town_exists:
        # Inferred town. Nothing independent confirms or refutes it.
        return None, "town_inferred_no_independent_truth"

    if provider is None:
        return None, "no_town_country_reference"

    if not provider.knows(verified.town):
        # Reference coverage gap. Not a model error.
        return None, "town_absent_from_reference"

    return True, "explicit_in_text_and_known_to_reference"


def _country_label(
    verified: VerifiedExtraction, provider: Any
) -> tuple[bool | None, str]:
    if verified.country_value in {"", NO_COUNTRY} or not verified.country_candidates:
        return None, "no_country_predicted"

    if verified.country_ambiguous:
        return None, "country_candidates_unresolved"

    predicted = verified.country_candidates[0]

    if verified.reference_status == REFERENCE_CONFLICT:
        # Deterministic reference truth exists and disagrees.
        return False, "contradicted_by_reference"

    if verified.country_exists:
        return True, "explicit_in_address_text"

    reference_codes = verified.reference_codes
    if not reference_codes:
        if verified.reference_status in {REFERENCE_NOT_FOUND, REFERENCE_NOT_CHECKED}:
            return None, (
                "reference_not_found"
                if verified.reference_status == REFERENCE_NOT_FOUND
                else "no_town_country_reference"
            )
        return None, "no_reference_evidence"

    if verified.reference_status in {
        REFERENCE_MULTI_ANNOTATED,
        REFERENCE_MULTI_ESCALATED,
    } or len(reference_codes) > 1:
        # The town spans several countries and the address did not resolve it.
        # Unresolved is not incorrect.
        return None, "reference_multi_country_unresolved"

    if predicted == reference_codes[0]:
        return True, "single_country_reference_match"
    return False, "contradicted_by_reference"


def compute_cross_entropy(
    verified: VerifiedExtraction,
    ground_truth: GroundTruth,
    *,
    epsilon: float = CROSS_ENTROPY_EPSILON,
) -> CrossEntropyResult:
    """Binary cross-entropy per component, combined into a group loss.

    Both labels present → the group loss is the mean of the two component
    losses. One label → that component's loss, with a ``town_only`` /
    ``country_only`` status. Neither → ``None``.

    Missing reference coverage never becomes an artificially high loss: an
    ungrounded observation is excluded from the metric rather than counted as a
    model failure.
    """
    town_ce: float | None = None
    country_ce: float | None = None

    town_probability = float(verified.town_probability)
    country_probability = float(verified.country_probability)

    if ground_truth.town_available:
        town_ce = binary_cross_entropy(
            bool(ground_truth.town_exists_ok), town_probability, epsilon=epsilon
        )
    if ground_truth.country_available:
        country_ce = binary_cross_entropy(
            bool(ground_truth.country_exists_ok), country_probability, epsilon=epsilon
        )

    available = [value for value in (town_ce, country_ce) if value is not None]
    group_ce = sum(available) / len(available) if available else None

    return CrossEntropyResult(
        town_ground_truth_available=ground_truth.town_available,
        town_correct=ground_truth.town_exists_ok,
        town_probability=town_probability if ground_truth.town_available else None,
        town_cross_entropy=town_ce,
        country_ground_truth_available=ground_truth.country_available,
        country_correct=ground_truth.country_exists_ok,
        country_probability=(
            country_probability if ground_truth.country_available else None
        ),
        country_cross_entropy=country_ce,
        group_cross_entropy=group_ce,
        status=_status(verified, ground_truth),
    )


def _status(verified: VerifiedExtraction, ground_truth: GroundTruth) -> str:
    town = ground_truth.town_available
    country = ground_truth.country_available

    if town and country:
        return STATUS_BOTH_GROUNDED
    if town:
        return STATUS_TOWN_ONLY
    if country:
        return STATUS_COUNTRY_ONLY

    # Neither component grounded — say why, so a coverage gap is separable from
    # genuine ambiguity in the reports.
    if verified.reference_status == REFERENCE_NOT_FOUND:
        return STATUS_REFERENCE_NOT_FOUND
    if verified.country_ambiguous or verified.reference_status in {
        REFERENCE_MULTI_ANNOTATED,
        REFERENCE_MULTI_ESCALATED,
    }:
        return STATUS_AMBIGUOUS
    return STATUS_NOT_AVAILABLE


def null_ground_truth() -> GroundTruth:
    """Labels for a null-skipped group: unknown, and never a model error."""
    return GroundTruth(town_basis="null_skip", country_basis="null_skip")


def null_cross_entropy() -> CrossEntropyResult:
    """Cross-entropy for a null-skipped or ungrounded group: nothing to score."""
    return CrossEntropyResult(status=STATUS_NOT_AVAILABLE)


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)
