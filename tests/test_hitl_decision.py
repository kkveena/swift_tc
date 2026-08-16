"""The explicit HITL routing decision: states, precedence, and forced review.

The rule under test throughout: **the threshold is not the only decision rule.**
A case can require human review with a score comfortably above
`scoring.hitl_threshold`, because a forced-review control overrode the number.
"""

from __future__ import annotations

import pytest

from swift_address.scoring import (
    FORCED_REVIEW_STATES,
    HITL_AMBIGUOUS_COUNTRY,
    HITL_AUTO_ACCEPT_CANDIDATE,
    HITL_LOW_SCORE,
    HITL_MANUAL_OVERRIDE,
    HITL_PROCESSING_ERROR,
    HITL_REFERENCE_CONFLICT,
    HITL_STATE_NOT_EVALUATED,
    HITL_STATE_PRECEDENCE,
    REFERENCE_CONFLICT,
    REFERENCE_CONSISTENT,
    ScoreResult,
    VerifiedExtraction,
    determine_hitl_decision,
    null_hitl_decision,
)


def verified(*, ambiguous: bool = False, conflict: bool = False) -> VerifiedExtraction:
    return VerifiedExtraction(
        town="BOSTON",
        country_candidates=("US",),
        country_value="US",
        country_name_value="United States",
        town_exists=True,
        country_exists=True,
        country_ambiguous=ambiguous,
        town_probability=0.99,
        country_probability=0.99,
        reference_status=REFERENCE_CONFLICT if conflict else REFERENCE_CONSISTENT,
    )


def scored(composite: float, *, threshold: float = 0.80) -> ScoreResult:
    return ScoreResult(
        scenario="both_explicit",
        town_weight=1.0,
        country_weight=1.0,
        town_probability=0.99,
        country_probability=0.99,
        adjusted_town_score=0.99,
        adjusted_country_score=0.99,
        composite_weighted_score=composite,
        needs_hitl=composite < threshold,
    )


@pytest.fixture
def scoring(config):
    """The real configured policy — threshold 0.80, ambiguity forced to HITL."""
    return config.scoring


class TestConfiguredThreshold:
    def test_operational_threshold_is_080(self, scoring):
        """`scoring.hitl_threshold` is the routing cutoff and stays 0.80."""
        assert scoring.hitl_threshold == 0.80

    def test_recommended_threshold_is_a_separate_analytical_setting(self, config):
        """0.90 is a reporting recommendation; it must not become the cutoff."""
        assert config.reporting.recommended_threshold == 0.90
        assert config.scoring.hitl_threshold != config.reporting.recommended_threshold

    def test_decision_reports_the_configured_threshold(self, scoring):
        decision = determine_hitl_decision(verified(), scored(0.95), scoring)
        assert decision.threshold == 0.80


class TestThresholdBehaviour:
    def test_just_below_threshold_is_low_score(self, scoring):
        decision = determine_hitl_decision(verified(), scored(0.79), scoring)
        assert decision.state == HITL_LOW_SCORE
        assert decision.required is True

    def test_exactly_at_threshold_is_auto_accept(self, scoring):
        """The boundary belongs to auto-accept: `>=`, not `>`."""
        decision = determine_hitl_decision(verified(), scored(0.80), scoring)
        assert decision.state == HITL_AUTO_ACCEPT_CANDIDATE
        assert decision.required is False

    def test_just_above_threshold_is_auto_accept(self, scoring):
        decision = determine_hitl_decision(verified(), scored(0.81), scoring)
        assert decision.state == HITL_AUTO_ACCEPT_CANDIDATE
        assert decision.required is False

    def test_comparison_uses_full_precision_not_the_displayed_value(self, scoring):
        """0.7999 must not round to "0.80" and slip through as auto-accept."""
        decision = determine_hitl_decision(verified(), scored(0.7999), scoring)
        assert decision.state == HITL_LOW_SCORE
        assert "0.7999" in decision.reason      # displayed without misleading rounding


class TestAmbiguityPrecedence:
    def test_ambiguous_with_zero_score_names_ambiguity_not_low_score(self, scoring):
        """Ambiguity always scores 0.0, so it is *also* below threshold.

        The primary state must name the root cause, not the symptom.
        """
        decision = determine_hitl_decision(
            verified(ambiguous=True), scored(0.00), scoring
        )
        assert decision.state == HITL_AMBIGUOUS_COUNTRY
        assert decision.state != HITL_LOW_SCORE
        assert "below_threshold" in decision.contributing_reasons

    def test_ambiguous_above_threshold_still_routes_to_hitl(self, scoring):
        decision = determine_hitl_decision(
            verified(ambiguous=True), scored(0.95), scoring
        )
        assert decision.state == HITL_AMBIGUOUS_COUNTRY
        assert decision.required is True
        assert decision.forced_review is True

    def test_ambiguity_respects_the_configuration_switch(self, scoring):
        """If the policy stops forcing ambiguity, the state follows the score."""
        relaxed = scoring.model_copy(update={"force_ambiguous_country_to_hitl": False})
        decision = determine_hitl_decision(
            verified(ambiguous=True), scored(0.95), relaxed
        )
        assert decision.state == HITL_AUTO_ACCEPT_CANDIDATE


class TestReferenceConflictPrecedence:
    def test_conflict_below_threshold_names_the_conflict(self, scoring):
        decision = determine_hitl_decision(
            verified(conflict=True), scored(0.60), scoring
        )
        assert decision.state == HITL_REFERENCE_CONFLICT
        assert decision.state != HITL_LOW_SCORE
        assert decision.contributing_reasons == (
            "reference_conflict", "below_threshold",
        )

    def test_conflict_above_threshold_still_requires_review(self, scoring):
        """The mandatory case: the score passes and the control still wins."""
        decision = determine_hitl_decision(
            verified(conflict=True), scored(0.91), scoring
        )
        assert decision.required is True
        assert decision.state == HITL_REFERENCE_CONFLICT
        assert decision.forced_review is True
        assert decision.contributing_reasons == ("reference_conflict",)

    def test_conflict_above_threshold_reason_says_despite(self, scoring):
        """The wording must make the override explicit to a reviewer."""
        decision = determine_hitl_decision(
            verified(conflict=True), scored(0.91), scoring
        )
        assert decision.reason == (
            "Predicted Country conflicts with deterministic reference data; human "
            "review is required despite Composite Weighted Score 0.91 meeting "
            "configured threshold 0.80."
        )

    def test_ambiguity_outranks_conflict(self, scoring):
        decision = determine_hitl_decision(
            verified(ambiguous=True, conflict=True), scored(0.00), scoring
        )
        assert decision.state == HITL_AMBIGUOUS_COUNTRY
        assert "reference_conflict" in decision.contributing_reasons


class TestProcessingError:
    def test_extraction_failure_is_a_processing_error(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.00), scoring, extraction_error=True
        )
        assert decision.state == HITL_PROCESSING_ERROR
        assert decision.required is True
        assert decision.forced_review is True

    def test_processing_error_outranks_low_score(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.35), scoring, extraction_error=True
        )
        assert decision.state == HITL_PROCESSING_ERROR
        assert "below_threshold" in decision.contributing_reasons

    def test_processing_error_outranks_every_other_control(self, scoring):
        decision = determine_hitl_decision(
            verified(ambiguous=True, conflict=True), scored(0.00), scoring,
            extraction_error=True, manual_override=True,
        )
        assert decision.state == HITL_PROCESSING_ERROR
        assert set(decision.contributing_reasons) == {
            "processing_error", "manual_override", "country_ambiguous",
            "reference_conflict", "below_threshold",
        }

    def test_reason_does_not_claim_a_model_conclusion(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.00), scoring, extraction_error=True
        )
        assert decision.reason == (
            "Extraction failed after configured retries; no valid model result "
            "was produced."
        )


class TestManualOverride:
    def test_override_forces_review_even_at_a_high_score(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.95), scoring, manual_override=True
        )
        assert decision.state == HITL_MANUAL_OVERRIDE
        assert decision.required is True
        assert decision.manual_override is True

    def test_processing_error_outranks_manual_override(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.95), scoring,
            extraction_error=True, manual_override=True,
        )
        assert decision.state == HITL_PROCESSING_ERROR

    def test_supplied_reason_is_used_verbatim(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.95), scoring, manual_override=True,
            manual_override_reason="Sanctions review requested by Compliance.",
        )
        assert decision.reason == "Sanctions review requested by Compliance."

    def test_default_when_no_reason_supplied(self, scoring):
        decision = determine_hitl_decision(
            verified(), scored(0.95), scoring, manual_override=True
        )
        assert decision.reason == "Manual business override requires human review."

    def test_phase_1_never_sets_an_override(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        """The seam exists for Phase 2; Phase 1 must not fabricate overrides."""
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        for decision in result.decisions_by_address.values():
            assert decision.hitl.manual_override is False
            assert decision.hitl.state != HITL_MANUAL_OVERRIDE


class TestAutoAcceptCandidate:
    def test_requires_every_condition_to_be_clear(self, scoring):
        decision = determine_hitl_decision(verified(), scored(0.95), scoring)
        assert decision.state == HITL_AUTO_ACCEPT_CANDIDATE
        assert decision.required is False
        assert decision.forced_review is False
        assert decision.contributing_reasons == ()

    @pytest.mark.parametrize(
        "kwargs,verified_kwargs",
        [
            ({"extraction_error": True}, {}),
            ({"manual_override": True}, {}),
            ({}, {"ambiguous": True}),
            ({}, {"conflict": True}),
        ],
    )
    def test_any_forced_condition_disqualifies_it(
        self, scoring, kwargs, verified_kwargs
    ):
        decision = determine_hitl_decision(
            verified(**verified_kwargs), scored(0.95), scoring, **kwargs
        )
        assert decision.state != HITL_AUTO_ACCEPT_CANDIDATE
        assert decision.required is True

    def test_reason_states_the_threshold_was_met(self, scoring):
        decision = determine_hitl_decision(verified(), scored(0.95), scoring)
        assert decision.reason == (
            "Composite Weighted Score 0.95 meets configured HITL threshold 0.80 "
            "and no forced-review condition is present."
        )


class TestForcedReviewFlag:
    @pytest.mark.parametrize(
        "state", [
            HITL_PROCESSING_ERROR, HITL_MANUAL_OVERRIDE,
            HITL_AMBIGUOUS_COUNTRY, HITL_REFERENCE_CONFLICT,
        ],
    )
    def test_control_driven_states_are_forced(self, state):
        assert state in FORCED_REVIEW_STATES

    @pytest.mark.parametrize("state", [HITL_LOW_SCORE, HITL_AUTO_ACCEPT_CANDIDATE])
    def test_score_driven_states_are_not_forced(self, state):
        assert state not in FORCED_REVIEW_STATES

    def test_low_score_only_is_not_forced_review(self, scoring):
        decision = determine_hitl_decision(verified(), scored(0.35), scoring)
        assert decision.required is True
        assert decision.forced_review is False

    def test_reference_conflict_is_forced_review(self, scoring):
        decision = determine_hitl_decision(
            verified(conflict=True), scored(0.60), scoring
        )
        assert decision.forced_review is True


class TestPrecedenceOrder:
    def test_declared_order_matches_the_policy(self):
        assert HITL_STATE_PRECEDENCE == (
            HITL_PROCESSING_ERROR,
            HITL_MANUAL_OVERRIDE,
            HITL_AMBIGUOUS_COUNTRY,
            HITL_REFERENCE_CONFLICT,
            HITL_LOW_SCORE,
            HITL_AUTO_ACCEPT_CANDIDATE,
        )

    def test_only_one_primary_state_is_produced(self, scoring):
        decision = determine_hitl_decision(
            verified(ambiguous=True, conflict=True), scored(0.10), scoring
        )
        assert isinstance(decision.state, str)
        assert decision.state in HITL_STATE_PRECEDENCE

    def test_state_is_never_model_generated(self, scoring):
        """The enum is closed: only the six declared states can appear."""
        for kwargs in ({}, {"extraction_error": True}, {"manual_override": True}):
            for verified_kwargs in ({}, {"ambiguous": True}, {"conflict": True}):
                decision = determine_hitl_decision(
                    verified(**verified_kwargs), scored(0.5), scoring, **kwargs
                )
                assert decision.state in HITL_STATE_PRECEDENCE


class TestNullSkip:
    def test_null_group_is_not_evaluated(self, scoring):
        decision = null_hitl_decision(scoring)
        assert decision.required is False
        assert decision.state == HITL_STATE_NOT_EVALUATED == ""
        assert decision.reason == ""

    def test_null_group_is_never_auto_accept(self, scoring):
        """A group the model never saw is not a candidate for anything."""
        assert null_hitl_decision(scoring).state != HITL_AUTO_ACCEPT_CANDIDATE

    def test_null_group_is_not_forced_review(self, scoring):
        assert null_hitl_decision(scoring).forced_review is False


class TestNeedsHitlInvariant:
    """`ScoreResult.needs_hitl` survives and agrees on Phase 1 paths."""

    @pytest.mark.parametrize(
        "verified_kwargs,composite",
        [
            ({}, 0.95), ({}, 0.80), ({}, 0.79), ({}, 0.35),
            ({"ambiguous": True}, 0.00),
            ({"conflict": True}, 0.60),
            ({"conflict": True}, 0.91),
        ],
    )
    def test_decision_agrees_with_needs_hitl(self, scoring, verified_kwargs, composite):
        from swift_address.scoring import score

        subject = verified(**verified_kwargs)
        score_result = score(subject, scoring)
        decision = determine_hitl_decision(subject, score_result, scoring)
        assert decision.required == score_result.needs_hitl

    def test_manual_override_is_the_documented_exception(self, scoring):
        """A Phase 2 override intentionally diverges from the score-only view."""
        subject = verified()
        score_result = scored(0.95)
        decision = determine_hitl_decision(
            subject, score_result, scoring, manual_override=True
        )
        assert score_result.needs_hitl is False
        assert decision.required is True

    def test_needs_hitl_still_exists_on_score_result(self, scoring):
        from swift_address.scoring import score

        assert isinstance(score(verified(), scoring).needs_hitl, bool)

    def test_pipeline_paths_agree(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        for decision in result.decisions_by_address.values():
            assert decision.hitl.required == decision.score.needs_hitl


class TestNoModelInvolvement:
    def test_the_response_schema_has_no_hitl_field(self):
        """Gemini is never asked to choose the state or write the reason."""
        from swift_address.schemas import REQUIRED_RESPONSE_FIELDS, RESPONSE_JSON_SCHEMA

        assert not any("hitl" in field.lower() for field in REQUIRED_RESPONSE_FIELDS)
        assert not any(
            "hitl" in key.lower() for key in RESPONSE_JSON_SCHEMA["properties"]
        )

    def test_the_prompt_never_mentions_a_hitl_state(self, prompt_contract):
        for state in HITL_STATE_PRECEDENCE:
            assert state not in prompt_contract.text
