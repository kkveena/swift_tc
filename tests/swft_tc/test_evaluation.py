"""Ground-truth correctness labels and binary cross-entropy.

The distinction under test throughout: ``predicted_*_exists`` is "explicitly
present in the text"; ``*_exists_ok`` is "correct, when independent evidence
exists". ``*_exists_ok`` is a plain boolean — unknown collapses to ``False`` —
but ``*_basis`` / ``*_available`` still separate a genuine judgement from a
collapsed unknown, which is what keeps cross-entropy and the reporting
correctness rate from counting a coverage gap as a model error.
"""

from __future__ import annotations

import math

import pytest

from models.swft_tc.src.evaluation import (
    CROSS_ENTROPY_EPSILON,
    STATUS_AMBIGUOUS,
    STATUS_BOTH_GROUNDED,
    STATUS_COUNTRY_ONLY,
    STATUS_NOT_AVAILABLE,
    STATUS_REFERENCE_NOT_FOUND,
    STATUS_TOWN_ONLY,
    binary_cross_entropy,
    compute_cross_entropy,
    evaluate_ground_truth,
    null_cross_entropy,
    null_ground_truth,
)
from models.swft_tc.src.schemas import parse_extraction_response
from models.swft_tc.src.scoring import evaluate, verify_extraction

from test_pipeline import BOSTON_RESPONSE


def make_response(**overrides):
    payload = dict(BOSTON_RESPONSE)
    payload.update(overrides)
    return parse_extraction_response(payload)


def verify(response, address, iso_provider, town_country_provider=None, **kwargs):
    return verify_extraction(
        response,
        address,
        iso_provider=iso_provider,
        town_country_provider=town_country_provider,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Meaning of the existing vs the new fields
# ---------------------------------------------------------------------------


class TestFieldMeaningsAreDistinct:
    def test_predicted_exists_still_means_explicit_text_presence(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider, town_country_provider,
        )
        assert verified.town_exists is True
        assert verified.country_exists is True

    def test_predicted_exists_is_false_when_value_absent_from_text(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town="TAIPEI", country_candidates=["TW"],
                          country_is_explicit=False),
            "TAIPEI HEAD OFFICE", iso_provider, town_country_provider,
        )
        assert verified.town_exists is True
        assert verified.country_exists is False       # TW is not in the text

    def test_exists_ok_can_be_true_where_exists_is_false(
        self, iso_provider, town_country_provider
    ):
        """The two fields answer different questions and may legitimately differ.

        TAIPEI's country is not written in the address (`country_exists` False),
        but the reference resolves TAIPEI to exactly one country, which the
        prediction matches — so the correctness label is True.
        """
        verified = verify(
            make_response(town="TAIPEI", country_candidates=["TW"],
                          country_is_explicit=False),
            "TAIPEI HEAD OFFICE", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert verified.country_exists is False
        assert truth.country_exists_ok is True
        assert truth.country_basis == "single_country_reference_match"


# ---------------------------------------------------------------------------
# Town label
# ---------------------------------------------------------------------------


class TestTownGroundTruth:
    def test_true_when_explicit_in_text_and_known_to_reference(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.town_exists_ok is True

    def test_false_only_on_positive_contradiction(
        self, iso_provider, town_country_provider
    ):
        """AERONAUTICA -> RONA: the model asserted explicit support that is absent."""
        verified = verify(
            make_response(town="RONA", town_evidence="RONA", town_is_explicit=True,
                          country_candidates=[], country_is_explicit=False),
            "AERONAUTICA", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.town_exists_ok is False
        assert truth.town_basis == "explicit_claim_refuted_by_text"

    def test_false_when_town_merely_inferred(self, iso_provider, town_country_provider):
        verified = verify(
            make_response(town="BOSTON", town_is_explicit=False),
            "PO BOX 1234 US", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert verified.town_exists is False
        assert truth.town_exists_ok is False

    def test_false_when_town_absent_from_reference(
        self, iso_provider, town_country_provider
    ):
        """A coverage gap collapses to False, but town_basis still says why."""
        verified = verify(
            make_response(town="NOWHERESVILLE", country_candidates=[],
                          country_is_explicit=False),
            "1 MAIN STREET NOWHERESVILLE", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert verified.town_exists is True          # it IS in the text
        assert truth.town_exists_ok is False         # but nothing corroborates it
        assert truth.town_basis == "town_absent_from_reference"

    def test_false_without_any_reference(self, iso_provider):
        verified = verify(make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
                          iso_provider, None)
        truth = evaluate_ground_truth(verified, town_country_provider=None)
        assert truth.town_exists_ok is False
        assert truth.town_basis == "no_town_country_reference"

    def test_false_when_no_town_predicted(self, iso_provider, town_country_provider):
        from models.swft_tc.src.schemas import NO_TOWN

        verified = verify(
            make_response(town=NO_TOWN, country_candidates=[]),
            "AERONAUTICA", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.town_exists_ok is False

    def test_false_on_extraction_error(self):
        from models.swft_tc.src.scoring import error_result

        verified, _ = error_result("timeout")
        truth = evaluate_ground_truth(verified, extraction_failed=True)
        assert truth.town_exists_ok is False
        assert truth.country_exists_ok is False
        assert truth.town_basis == "extraction_error"


# ---------------------------------------------------------------------------
# Country label
# ---------------------------------------------------------------------------


class TestCountryGroundTruth:
    def test_true_when_explicit_in_address(self, iso_provider, town_country_provider):
        verified = verify(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is True
        assert truth.country_basis == "explicit_in_address_text"

    def test_true_on_single_country_reference_match(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town="AUCKLAND", country_candidates=["NZ"],
                          country_is_explicit=False),
            "23 CUSTOMS STREET AUCKLAND", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is True

    def test_false_when_contradicted_by_reference(
        self, iso_provider, town_country_provider
    ):
        """The reference resolves ZURICH to CH; the model said FR."""
        verified = verify(
            make_response(town="ZURICH", country_candidates=["FR"],
                          country_is_explicit=False),
            "BAHNHOFSTRASSE 1 ZURICH", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is False
        assert truth.country_basis == "contradicted_by_reference"

    def test_false_for_multi_country_town_without_resolution(
        self, iso_provider, town_country_provider
    ):
        """Unresolved collapses to False, but country_basis still says why."""
        verified = verify(
            make_response(town="LIMA", country_candidates=["PE"],
                          country_is_explicit=False),
            "441-445 JIRON SANTA ROSA LIMA 15001",
            iso_provider, town_country_provider,
            town_country_ambiguity_policy="annotate",
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is False
        assert truth.country_basis == "reference_multi_country_unresolved"

    def test_false_for_escalated_ambiguous_candidate_set(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town="LIMA", country_candidates=["PE"],
                          country_is_explicit=False),
            "441-445 JIRON SANTA ROSA LIMA 15001",
            iso_provider, town_country_provider,
            town_country_ambiguity_policy="escalate",
        )
        assert verified.country_ambiguous is True
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is False

    def test_false_when_town_not_found_in_reference(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town="NOWHERESVILLE", country_candidates=["US"],
                          country_is_explicit=False),
            "1 MAIN STREET NOWHERESVILLE", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is False

    def test_false_when_no_country_predicted(self, iso_provider, town_country_provider):
        verified = verify(
            make_response(town="BOSTON", country_candidates=[],
                          country_is_explicit=False),
            "1 LINCOLN STREET BOSTON", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        assert truth.country_exists_ok is False


# ---------------------------------------------------------------------------
# Binary cross-entropy
# ---------------------------------------------------------------------------


class TestBinaryCrossEntropy:
    def test_confident_and_correct_is_cheap(self):
        assert binary_cross_entropy(True, 0.95) == pytest.approx(0.051293, abs=1e-6)

    def test_confident_and_wrong_is_expensive(self):
        assert binary_cross_entropy(False, 0.95) == pytest.approx(2.995732, abs=1e-6)

    def test_matches_the_negative_log_definition(self):
        assert binary_cross_entropy(True, 0.98) == pytest.approx(-math.log(0.98))
        assert binary_cross_entropy(False, 0.2) == pytest.approx(-math.log(0.8))

    @pytest.mark.parametrize("probability", [0.0, 1.0])
    def test_extremes_are_clipped_and_finite(self, probability):
        for correct in (True, False):
            value = binary_cross_entropy(correct, probability)
            assert math.isfinite(value)
            assert value >= 0.0

    def test_clipping_uses_the_configured_epsilon(self):
        assert binary_cross_entropy(True, 0.0) == pytest.approx(
            -math.log(CROSS_ENTROPY_EPSILON)
        )

    def test_uncertainty_costs_the_same_either_way(self):
        assert binary_cross_entropy(True, 0.5) == pytest.approx(
            binary_cross_entropy(False, 0.5)
        )

    def test_loss_increases_as_confidence_in_a_wrong_answer_grows(self):
        losses = [binary_cross_entropy(False, p) for p in (0.6, 0.8, 0.95, 0.99)]
        assert losses == sorted(losses)


class TestGroupCrossEntropy:
    def _verified(self, iso_provider, town_country_provider, **overrides):
        return verify(
            make_response(**overrides), overrides.pop("_address", "X"),
            iso_provider, town_country_provider,
        )

    def test_both_available_averages_the_components(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town_model_confidence=0.98, country_model_confidence=0.99),
            "1 LINCOLN STREET BOSTON MA 02111 US", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)

        assert result.status == STATUS_BOTH_GROUNDED
        assert result.town_cross_entropy == pytest.approx(-math.log(0.98), abs=1e-6)
        assert result.country_cross_entropy == pytest.approx(-math.log(0.99), abs=1e-6)
        assert result.group_cross_entropy == pytest.approx(
            (result.town_cross_entropy + result.country_cross_entropy) / 2
        )

    def test_town_only_is_partial_and_uses_the_available_component(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town="LIMA", town_model_confidence=0.98,
                          country_candidates=["PE"], country_is_explicit=False),
            "441-445 JIRON SANTA ROSA LIMA 15001",
            iso_provider, town_country_provider,
            town_country_ambiguity_policy="annotate",
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)

        assert result.status == STATUS_TOWN_ONLY
        assert result.is_partial is True
        assert result.country_cross_entropy is None
        assert result.group_cross_entropy == pytest.approx(result.town_cross_entropy)

    def test_country_only_is_partial(self, iso_provider, town_country_provider):
        verified = verify(
            make_response(town="NOWHERESVILLE", country_candidates=["US"]),
            "PO BOX 9 NOWHERESVILLE US", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)

        assert result.status == STATUS_COUNTRY_ONLY
        assert result.town_cross_entropy is None
        assert result.group_cross_entropy == pytest.approx(result.country_cross_entropy)

    def test_neither_available_yields_null_not_a_high_loss(
        self, iso_provider, town_country_provider
    ):
        """Missing reference coverage is not a model error."""
        verified = verify(
            make_response(town="NOWHERESVILLE", country_candidates=[],
                          country_is_explicit=False),
            "1 MAIN STREET NOWHERESVILLE", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)

        assert result.group_cross_entropy is None
        assert result.status == STATUS_REFERENCE_NOT_FOUND

    def test_ambiguous_ground_truth_status(self, iso_provider, town_country_provider):
        verified = verify(
            make_response(town="HAMILTON", country_candidates=["CA"],
                          country_is_explicit=False),
            "1 FRONT ST HAMILTON", iso_provider, town_country_provider,
            town_country_ambiguity_policy="escalate",
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)
        # Town is grounded here, so the status names the grounded component.
        assert result.status in {STATUS_TOWN_ONLY, STATUS_AMBIGUOUS}

    def test_wrong_and_confident_produces_a_large_group_loss(
        self, iso_provider, town_country_provider
    ):
        verified = verify(
            make_response(town="ZURICH", country_candidates=["FR"],
                          country_is_explicit=False, country_model_confidence=0.95),
            "BAHNHOFSTRASSE 1 ZURICH", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)

        assert result.country_correct is False
        assert result.country_cross_entropy == pytest.approx(2.995732, abs=1e-6)

    def test_null_helpers_are_ungrounded(self):
        assert null_ground_truth().town_exists_ok is False
        assert null_ground_truth().town_available is False
        assert null_cross_entropy().group_cross_entropy is None
        assert null_cross_entropy().status == STATUS_NOT_AVAILABLE

    def test_ungrounded_components_report_none_not_a_collapsed_false(
        self, iso_provider, town_country_provider
    ):
        """Inside the audit payload, ungrounded means None — never False.

        The collapse of unknown into False is scoped to the `*_exists_ok` CSV
        columns. The audit trail keeps "contradicted" and "no evidence" apart,
        so `*_correct` is gated on availability exactly like `*_probability`
        and `*_cross_entropy` beside it.
        """
        verified = verify(
            make_response(town="NOWHERESVILLE", country_candidates=[],
                          country_is_explicit=False),
            "1 MAIN STREET NOWHERESVILLE", iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        result = compute_cross_entropy(verified, truth)

        assert truth.town_available is False
        assert truth.town_exists_ok is False        # the collapsed CSV value
        assert result.town_correct is None          # but the audit says "unknown"
        assert result.town_probability is None
        assert result.town_cross_entropy is None

    def test_null_skip_and_ungrounded_extraction_agree(
        self, iso_provider, town_country_provider
    ):
        """Two ungrounded states must report the same *values*.

        A null-skipped group and an extracted-but-ungrounded one are both
        `available=False`. They previously disagreed on `*_correct`: null-skip
        reported None, the extracted one a collapsed False.

        `status` is excluded on purpose — it exists precisely to say *why* a
        component is ungrounded, so `reference_not_found` being more specific
        than `not_available` is the field working as intended.
        """
        verified = verify(
            make_response(town="NOWHERESVILLE", country_candidates=[],
                          country_is_explicit=False),
            "1 MAIN STREET NOWHERESVILLE", iso_provider, town_country_provider,
        )
        ungrounded = compute_cross_entropy(
            verified,
            evaluate_ground_truth(
                verified, town_country_provider=town_country_provider
            ),
        ).to_dict()
        null_skip = null_cross_entropy().to_dict()

        for key in (
            "town_ground_truth_available", "town_correct", "town_probability",
            "town_cross_entropy", "country_ground_truth_available",
            "country_correct", "country_probability", "country_cross_entropy",
            "group_cross_entropy",
        ):
            assert ungrounded[key] == null_skip[key], key

        # Both are ungrounded statuses; only the reason differs.
        assert ungrounded["status"] == STATUS_REFERENCE_NOT_FOUND
        assert null_skip["status"] == STATUS_NOT_AVAILABLE

    def test_grounded_components_still_report_their_label(
        self, iso_provider, town_country_provider
    ):
        """Gating on availability must not blank out a real judgement."""
        verified = verify(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider, town_country_provider,
        )
        result = compute_cross_entropy(
            verified,
            evaluate_ground_truth(
                verified, town_country_provider=town_country_provider
            ),
        )
        assert result.town_correct is True
        assert result.country_correct is True

    def test_a_refuted_claim_still_reports_false_not_none(
        self, iso_provider, town_country_provider
    ):
        """AERONAUTICA -> RONA is grounded: a real contradiction, not a gap."""
        verified = verify(
            make_response(town="RONA", town_evidence="RONA", town_is_explicit=True,
                          country_candidates=[], country_is_explicit=False),
            "AERONAUTICA", iso_provider, town_country_provider,
        )
        result = compute_cross_entropy(
            verified,
            evaluate_ground_truth(
                verified, town_country_provider=town_country_provider
            ),
        )
        assert result.town_ground_truth_available is True
        assert result.town_correct is False
        assert result.town_cross_entropy is not None

    def test_json_payload_shape(self, iso_provider, town_country_provider):
        verified = verify(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider, town_country_provider,
        )
        truth = evaluate_ground_truth(
            verified, town_country_provider=town_country_provider
        )
        payload = compute_cross_entropy(verified, truth).to_dict()

        assert set(payload) == {
            "town_ground_truth_available", "town_correct", "town_probability",
            "town_cross_entropy", "country_ground_truth_available",
            "country_correct", "country_probability", "country_cross_entropy",
            "group_cross_entropy", "status",
        }


class TestScoringIsUnchanged:
    """Cross-entropy must not disturb the operational routing score."""

    def test_composite_formula_still_holds(self, config, iso_provider):
        _, result = evaluate(
            make_response(town_model_confidence=0.99, country_model_confidence=0.98),
            "1 LINCOLN STREET BOSTON MA 02111 US",
            config.scoring, iso_provider=iso_provider,
        )
        assert result.composite_weighted_score == pytest.approx(0.9702)

    def test_weight_matrix_is_untouched(self, config):
        expected = {
            "both_explicit": (1.00, 1.00),
            "country_explicit_town_inferred": (0.50, 1.00),
            "town_explicit_country_inferred": (0.75, 0.50),
            "town_explicit_country_ambiguous": (0.50, 0.00),
            "neither_explicit_both_inferred": (0.20, 0.20),
            "no_defensible_prediction": (0.00, 0.00),
            "town_inferred_country_ambiguous": (0.20, 0.00),
        }
        for scenario, (town_weight, country_weight) in expected.items():
            weights = config.scoring.weights_for(scenario)
            assert (weights.town_weight, weights.country_weight) == (
                pytest.approx(town_weight), pytest.approx(country_weight)
            )
