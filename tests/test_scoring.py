"""Verification, scenario selection, and the Composite Weighted Score.

These tests are independent of Gemini: they feed constructed responses through
the deterministic decision layer and assert the business outcome.
"""

from __future__ import annotations

import pytest

from swift_address.schemas import NO_COUNTRY, NO_TOWN, parse_extraction_response
from swift_address.scoring import (
    AMBIGUOUS_TOWN_INFERRED_SCENARIO,
    REQUIRED_SCENARIOS,
    error_result,
    evaluate,
    null_result,
    select_scenario,
    verify_extraction,
)


def make_response(**overrides):
    """A schema-valid response with sensible defaults, overridable per test."""
    payload = {
        "town": "BOSTON",
        "country_candidates": ["US"],
        "town_evidence": "BOSTON",
        "country_evidence": "US",
        "town_is_explicit": True,
        "country_is_explicit": True,
        "town_ambiguous": False,
        "country_ambiguous": False,
        "town_model_confidence": 0.99,
        "country_model_confidence": 0.99,
        "town_rationale": "Town appears in the address.",
        "country_rationale": "ISO code appears in the address.",
        "reference_basis": ["input_text"],
    }
    payload.update(overrides)
    return parse_extraction_response(payload)


# ---------------------------------------------------------------------------
# Presence verification
# ---------------------------------------------------------------------------


class TestPresenceVerification:
    def test_town_and_country_present_are_verified(self, iso_provider):
        verified = verify_extraction(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider=iso_provider,
        )
        assert verified.town_exists is True
        assert verified.country_exists is True

    def test_model_explicit_claim_is_overruled_when_text_lacks_support(
        self, iso_provider
    ):
        """AERONAUTICA -> RONA: the model asserts it, the text does not support it."""
        verified = verify_extraction(
            make_response(
                town="RONA",
                town_evidence="RONA",
                town_is_explicit=True,
                country_candidates=[],
                country_is_explicit=False,
            ),
            "AERONAUTICA",
            iso_provider=iso_provider,
        )
        assert verified.town_exists is False
        assert "town_explicit_claim_unverified" in verified.notes

    def test_country_explicit_claim_is_overruled_when_code_absent(self, iso_provider):
        verified = verify_extraction(
            make_response(town="LIMA", country_candidates=["PE"], country_is_explicit=True),
            "441-445 JIRON SANTA ROSA LIMA METRO MUNIC OF LIMA 15001",
            iso_provider=iso_provider,
        )
        assert verified.country_exists is False
        assert "country_explicit_claim_unverified" in verified.notes

    def test_country_name_alias_counts_as_explicit_support(self, iso_provider):
        verified = verify_extraction(
            make_response(town="AUCKLAND", country_candidates=["NZ"]),
            "23 CUSTOMS STREET AUCKLAND NEW ZEALAND",
            iso_provider=iso_provider,
        )
        assert verified.country_exists is True

    def test_presence_overrides_a_model_inferred_claim(self, iso_provider):
        """Deterministic evidence wins in both directions, and is recorded."""
        verified = verify_extraction(
            make_response(town_is_explicit=False, country_is_explicit=False),
            "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider=iso_provider,
        )
        assert verified.town_exists is True
        assert "town_present_though_model_marked_inferred" in verified.notes

    def test_colliding_alpha2_token_is_not_evidence(self, iso_provider):
        """"IN" mid-address must not prove India."""
        verified = verify_extraction(
            make_response(town="MUMBAI", country_candidates=["IN"]),
            "SUITE 5 IN TOWER MUMBAI",
            iso_provider=iso_provider,
        )
        assert verified.country_exists is False

    def test_us_state_abbreviation_does_not_prove_morocco(self, iso_provider):
        verified = verify_extraction(
            make_response(town="BOSTON", country_candidates=["MA"]),
            "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider=iso_provider,
        )
        assert verified.country_exists is False

    def test_invalid_iso_codes_are_dropped_with_a_note(self, iso_provider):
        verified = verify_extraction(
            make_response(country_candidates=["US", "ZZ"]),
            "BOSTON MA 02111 US",
            iso_provider=iso_provider,
        )
        assert verified.country_candidates == ("US",)
        assert any(note.startswith("dropped_invalid_iso_codes") for note in verified.notes)


# ---------------------------------------------------------------------------
# Country candidate handling
# ---------------------------------------------------------------------------


class TestCountryCandidateOutput:
    def test_single_candidate_is_written_as_a_scalar_code(self, iso_provider):
        verified = verify_extraction(
            make_response(), "BOSTON MA 02111 US", iso_provider=iso_provider
        )
        assert verified.country_value == "US"
        assert verified.country_ambiguous is False

    def test_multiple_candidates_become_a_deterministic_comma_list(self, iso_provider):
        verified = verify_extraction(
            make_response(
                town="EXAMPLE_TOWN",
                country_candidates=["US", "CA"],
                country_is_explicit=False,
                country_ambiguous=True,
            ),
            "CONTROLLED MULTI COUNTRY FIXTURE EXAMPLE_TOWN",
            iso_provider=iso_provider,
        )
        assert verified.country_value == "CA,US"      # alphabetical, deterministic
        assert verified.country_candidates == ("CA", "US")
        assert verified.country_ambiguous is True

    def test_candidate_order_is_stable_regardless_of_model_order(self, iso_provider):
        first = verify_extraction(
            make_response(country_candidates=["US", "CA"], town="EXAMPLE_TOWN"),
            "EXAMPLE_TOWN",
            iso_provider=iso_provider,
        )
        second = verify_extraction(
            make_response(country_candidates=["CA", "US"], town="EXAMPLE_TOWN"),
            "EXAMPLE_TOWN",
            iso_provider=iso_provider,
        )
        assert first.country_value == second.country_value == "CA,US"

    def test_no_candidate_ever_becomes_the_sentinel(self, iso_provider):
        verified = verify_extraction(
            make_response(country_candidates=["CA", "US"], town="EXAMPLE_TOWN"),
            "EXAMPLE_TOWN",
            iso_provider=iso_provider,
        )
        assert NO_COUNTRY not in verified.country_candidates
        assert all(len(code) == 2 and code.isupper() for code in verified.country_candidates)

    def test_empty_candidate_set_yields_the_sentinel(self, iso_provider):
        verified = verify_extraction(
            make_response(town=NO_TOWN, country_candidates=[]),
            "AERONAUTICA",
            iso_provider=iso_provider,
        )
        assert verified.country_value == NO_COUNTRY

    def test_explicit_text_evidence_resolves_ambiguity(self, iso_provider):
        """Two candidates, but the text names one: that is resolution, not ambiguity."""
        verified = verify_extraction(
            make_response(
                town="LONDON",
                country_candidates=["CA", "GB"],
                country_ambiguous=True,
            ),
            "25 BANK STREET LONDON GB",
            iso_provider=iso_provider,
        )
        assert verified.country_value == "GB"
        assert verified.country_ambiguous is False
        assert any(
            note.startswith("ambiguity_resolved_by_explicit_text_evidence")
            for note in verified.notes
        )

    def test_ambiguity_survives_when_the_text_names_both_countries(self, iso_provider):
        """"1 CANADA SQUARE LONDON GB" names CA and GB; nothing is resolved.

        Resolution requires *exactly one* candidate to have textual support.
        Two supported candidates is still an unresolved choice, and the
        pipeline must not pick one.
        """
        verified = verify_extraction(
            make_response(
                town="LONDON",
                country_candidates=["CA", "GB"],
                country_ambiguous=True,
            ),
            "1 CANADA SQUARE LONDON GB",
            iso_provider=iso_provider,
        )
        assert verified.country_value == "CA,GB"
        assert verified.country_ambiguous is True


# ---------------------------------------------------------------------------
# Scenario selection — every configured policy scenario
# ---------------------------------------------------------------------------


class TestScenarioSelection:
    """Scenarios come from *verified* explicitness, not the model's opinion."""

    def _verified(self, iso_provider, response, address):
        return verify_extraction(response, address, iso_provider=iso_provider)

    def test_both_explicit(self, iso_provider, config):
        verified = self._verified(
            iso_provider, make_response(), "1 LINCOLN STREET BOSTON MA 02111 US"
        )
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "both_explicit"
        )

    def test_country_explicit_town_inferred(self, iso_provider, config):
        verified = self._verified(
            iso_provider,
            make_response(town="BOSTON", town_is_explicit=False),
            "PO BOX 1234 US",
        )
        assert verified.town_exists is False and verified.country_exists is True
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "country_explicit_town_inferred"
        )

    def test_town_explicit_country_inferred(self, iso_provider, config):
        verified = self._verified(
            iso_provider,
            make_response(town="TAIPEI", country_candidates=["TW"], country_is_explicit=False),
            "TAIPEI HEAD OFFICE",
        )
        assert verified.town_exists is True and verified.country_exists is False
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "town_explicit_country_inferred"
        )

    def test_town_explicit_country_ambiguous(self, iso_provider, config):
        verified = self._verified(
            iso_provider,
            make_response(town="EXAMPLE_TOWN", country_candidates=["CA", "US"]),
            "CONTROLLED MULTI COUNTRY FIXTURE EXAMPLE_TOWN",
        )
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "town_explicit_country_ambiguous"
        )

    def test_neither_explicit_both_inferred(self, iso_provider, config):
        verified = self._verified(
            iso_provider,
            make_response(
                town="PARIS", town_is_explicit=False,
                country_candidates=["FR"], country_is_explicit=False,
            ),
            "HEAD OFFICE BUILDING SEVEN",
        )
        assert verified.town_exists is False and verified.country_exists is False
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "neither_explicit_both_inferred"
        )

    def test_no_defensible_prediction(self, iso_provider, config):
        verified = self._verified(
            iso_provider,
            make_response(town=NO_TOWN, country_candidates=[], town_is_explicit=False,
                          country_is_explicit=False),
            "AERONAUTICA",
        )
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "no_defensible_prediction"
        )

    def test_town_inferred_country_ambiguous_extension(self, iso_provider, config):
        """The documented seventh scenario: ambiguity with no verified town."""
        verified = self._verified(
            iso_provider,
            make_response(town="SPRINGFIELD", country_candidates=["CA", "US"]),
            "MAIN BRANCH OFFICE",
        )
        assert verified.country_ambiguous is True and verified.town_exists is False
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            AMBIGUOUS_TOWN_INFERRED_SCENARIO
        )

    def test_extension_falls_back_when_not_configured(self, iso_provider):
        verified = self._verified(
            iso_provider,
            make_response(town="SPRINGFIELD", country_candidates=["CA", "US"]),
            "MAIN BRANCH OFFICE",
        )
        assert select_scenario(verified, available_scenarios=REQUIRED_SCENARIOS) == (
            "no_defensible_prediction"
        )

    def test_partial_prediction_has_no_partial_credit(self, iso_provider, config):
        """Town found but no defensible country: one factor is missing, so 0.0."""
        verified = self._verified(
            iso_provider,
            make_response(town="TAIPEI", country_candidates=[], country_is_explicit=False),
            "TAIPEI HEAD OFFICE",
        )
        assert select_scenario(verified, available_scenarios=config.scoring.rules) == (
            "no_defensible_prediction"
        )

    def test_every_required_scenario_has_configured_weights(self, config):
        for scenario in REQUIRED_SCENARIOS:
            weights = config.scoring.weights_for(scenario)
            assert 0.0 <= weights.town_weight <= 1.0
            assert 0.0 <= weights.country_weight <= 1.0


# ---------------------------------------------------------------------------
# Composite Weighted Score
# ---------------------------------------------------------------------------


class TestCompositeWeightedScore:
    @pytest.mark.parametrize(
        "scenario,town_p,country_p,expected",
        [
            # From SCORING_SPEC.md's worked examples.
            ("both_explicit", 0.99, 0.98, 0.9702),
            ("country_explicit_town_inferred", 0.92, 0.99, 0.4554),
            ("town_explicit_country_inferred", 0.98, 0.95, 0.349125),
            ("neither_explicit_both_inferred", 0.80, 0.75, 0.0240),
            ("no_defensible_prediction", 0.99, 0.99, 0.0),
        ],
    )
    def test_spec_table_values(self, config, scenario, town_p, country_p, expected):
        weights = config.scoring.weights_for(scenario)
        composite = (town_p * weights.town_weight) * (country_p * weights.country_weight)
        assert composite == pytest.approx(expected)

    def test_formula_is_the_product_of_adjusted_scores(self, iso_provider, config):
        verified, result = evaluate(
            make_response(town_model_confidence=0.99, country_model_confidence=0.98),
            "1 LINCOLN STREET BOSTON MA 02111 US",
            config.scoring,
            iso_provider=iso_provider,
        )
        assert result.scenario == "both_explicit"
        assert result.adjusted_town_score == pytest.approx(0.99)
        assert result.adjusted_country_score == pytest.approx(0.98)
        assert result.composite_weighted_score == pytest.approx(0.9702)

    def test_expected_sample_row_lima(self, iso_provider, config):
        """Matches data/sample_expected_group15.csv for CA0000000694."""
        _, result = evaluate(
            make_response(
                town="LIMA", town_model_confidence=0.98,
                country_candidates=["PE"], country_is_explicit=False,
                country_model_confidence=0.95,
            ),
            "441-445 JIRON SANTA ROSA LIMA METRO MUNIC OF LIMA 15001",
            config.scoring,
            iso_provider=iso_provider,
        )
        assert result.scenario == "town_explicit_country_inferred"
        assert result.composite_weighted_score == pytest.approx(0.349125)

    def test_expected_sample_row_taipei(self, iso_provider, config):
        _, result = evaluate(
            make_response(
                town="TAIPEI", town_model_confidence=0.98,
                country_candidates=["TW"], country_is_explicit=False,
                country_model_confidence=0.95,
            ),
            "TAIPEI HEAD OFFICE",
            config.scoring,
            iso_provider=iso_provider,
        )
        assert result.scenario == "town_explicit_country_inferred"
        assert result.composite_weighted_score == pytest.approx(0.349125)


class TestAmbiguityOverride:
    """The mandatory unresolved-multiple-country rule."""

    @pytest.fixture
    def ambiguous(self, iso_provider, config):
        return evaluate(
            make_response(
                town="EXAMPLE_TOWN",
                town_model_confidence=0.98,
                country_candidates=["US", "CA"],
                country_is_explicit=False,
                country_ambiguous=True,
                country_model_confidence=0.90,   # must be discarded
            ),
            "CONTROLLED MULTI COUNTRY FIXTURE EXAMPLE_TOWN",
            config.scoring,
            iso_provider=iso_provider,
        )

    def test_candidates_are_preserved_not_collapsed(self, ambiguous):
        verified, _ = ambiguous
        assert verified.country_value == "CA,US"

    def test_country_probability_is_forced_to_zero(self, ambiguous):
        verified, result = ambiguous
        assert verified.country_probability == 0.0
        assert result.country_probability == 0.0
        assert "country_probability_overridden_for_ambiguity" in verified.notes

    def test_country_weight_is_zero(self, ambiguous):
        _, result = ambiguous
        assert result.country_weight == 0.0

    def test_composite_is_zero(self, ambiguous):
        _, result = ambiguous
        assert result.composite_weighted_score == 0.0

    def test_routed_to_hitl(self, ambiguous):
        _, result = ambiguous
        assert result.needs_hitl is True

    def test_town_probability_survives(self, ambiguous):
        verified, result = ambiguous
        assert verified.town_probability == pytest.approx(0.98)
        assert result.town_weight == pytest.approx(0.50)

    def test_matches_the_expected_fixture_row(self, ambiguous):
        """AMBIGUOUS_TEST in data/sample_expected_group15.csv."""
        verified, result = ambiguous
        assert (verified.town, verified.country_value) == ("EXAMPLE_TOWN", "CA,US")
        assert verified.town_exists is True
        assert verified.country_exists is False
        assert result.scenario == "town_explicit_country_ambiguous"
        assert result.composite_weighted_score == 0.0

    def test_ambiguity_zeroes_the_weight_even_if_config_says_otherwise(
        self, iso_provider, config
    ):
        """Policy, not a YAML value someone might edit."""
        tampered = config.scoring.model_copy(
            update={
                "rules": {
                    **config.scoring.rules,
                    "town_explicit_country_ambiguous": config.scoring.weights_for(
                        "both_explicit"
                    ),
                }
            }
        )
        _, result = evaluate(
            make_response(town="EXAMPLE_TOWN", country_candidates=["CA", "US"]),
            "EXAMPLE_TOWN BRANCH",
            tampered,
            iso_provider=iso_provider,
        )
        assert result.country_weight == 0.0
        assert result.composite_weighted_score == 0.0


class TestHitlRouting:
    def test_high_confidence_both_explicit_clears_the_threshold(
        self, iso_provider, config
    ):
        _, result = evaluate(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            config.scoring, iso_provider=iso_provider,
        )
        assert result.composite_weighted_score > config.scoring.hitl_threshold
        assert result.needs_hitl is False

    def test_inferred_country_falls_below_the_threshold(self, iso_provider, config):
        _, result = evaluate(
            make_response(town="TAIPEI", country_candidates=["TW"], country_is_explicit=False),
            "TAIPEI HEAD OFFICE", config.scoring, iso_provider=iso_provider,
        )
        assert result.needs_hitl is True

    def test_threshold_comes_from_config(self, config):
        assert config.scoring.hitl_threshold == 0.80


class TestSpecialResults:
    def test_null_result_is_all_zeros_and_not_hitl(self):
        verified, result = null_result()
        assert (verified.town, verified.country_value) == (NO_TOWN, NO_COUNTRY)
        assert result.scenario == "null_skip"
        assert result.composite_weighted_score == 0.0
        assert result.needs_hitl is False

    def test_error_result_is_neutral_and_forced_to_hitl(self):
        """An API failure is not a NO_TOWN conclusion."""
        verified, result = error_result("timeout")
        assert (verified.town, verified.country_value) == (NO_TOWN, NO_COUNTRY)
        assert result.scenario == "extraction_error"
        assert result.needs_hitl is True
        assert verified.notes == ("extraction_failed:timeout",)

    def test_score_rejects_an_unknown_scenario(self, config):
        with pytest.raises(KeyError, match="no reliability weights"):
            config.scoring.weights_for("invented_scenario")


class TestConfigValidation:
    def test_missing_required_scenario_is_rejected(self, repo_root, tmp_path):
        import yaml

        from swift_address.settings import load_config

        source = yaml.safe_load((repo_root / "config" / "config.yaml").read_text())
        del source["scoring"]["rules"]["both_explicit"]
        broken = tmp_path / "broken.yaml"
        broken.write_text(yaml.safe_dump(source), encoding="utf-8")

        with pytest.raises(Exception, match="both_explicit"):
            load_config(broken, base_dir=repo_root)

    def test_weight_outside_unit_interval_is_rejected(self, repo_root, tmp_path):
        import yaml

        from swift_address.settings import load_config

        source = yaml.safe_load((repo_root / "config" / "config.yaml").read_text())
        source["scoring"]["rules"]["both_explicit"]["town_weight"] = 1.5
        broken = tmp_path / "broken.yaml"
        broken.write_text(yaml.safe_dump(source), encoding="utf-8")

        with pytest.raises(Exception, match="town_weight"):
            load_config(broken, base_dir=repo_root)
