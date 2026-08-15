"""The prompt/response contract.

The prompt text lives in exactly one file. These tests keep that file, the JSON
Schema sent to the model, and the pydantic response model from drifting apart,
and prove that malformed output is caught rather than absorbed.
"""

from __future__ import annotations

import json

import pytest

from swift_address.schemas import (
    NO_COUNTRY,
    NO_TOWN,
    REQUIRED_RESPONSE_FIELDS,
    RESPONSE_JSON_SCHEMA,
    ExtractionResponse,
    MalformedExtractionResponse,
    build_user_payload,
    load_prompt_contract,
    parse_extraction_response,
)


def valid_payload(**overrides) -> dict:
    payload = {
        "town": "AUCKLAND",
        "country_candidates": ["NZ"],
        "town_evidence": "AUCKLAND",
        "country_evidence": "NZ",
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
    return payload


class TestPromptFile:
    def test_prompt_loads_from_its_single_source(self, prompt_contract, config):
        assert prompt_contract.source_path.name == "GEMINI_EXTRACTION_PROMPT.md"
        assert prompt_contract.version == config.project.prompt_version
        assert len(prompt_contract.text) > 500

    def test_prompt_documents_every_required_response_field(self, prompt_contract):
        for field in REQUIRED_RESPONSE_FIELDS:
            assert field in prompt_contract.text, f"{field} is missing from the prompt"

    def test_prompt_states_the_anti_substring_rule(self, prompt_contract):
        text = prompt_contract.text
        assert "AERONAUTICA" in text and "RONA" in text

    def test_prompt_forbids_unsupported_reference_claims(self, prompt_contract):
        assert "SWIFTRef" in prompt_contract.text
        assert "reference_context" in prompt_contract.text

    def test_prompt_states_the_sentinels(self, prompt_contract):
        assert NO_TOWN in prompt_contract.text
        assert NO_COUNTRY in prompt_contract.text

    def test_prompt_has_greenwich_new_york_regression_example(self, prompt_contract):
        text = prompt_contract.text
        assert "88 GREENWICH STREET NEW YORK NY 10013-2632 US" in text
        assert '"town":"NEW YORK"' in text
        assert '"country_candidates":["US"]' in text

    def test_prompt_has_lima_postal_code_regression_example(self, prompt_contract):
        text = prompt_contract.text
        assert "441-445 JIRON SANTA ROSA LIMA METRO MUNIC OF LIMA 15001" in text
        assert '"town":"LIMA"' in text
        assert '"country_candidates":["PE"]' in text
        assert '"country_is_explicit":false' in text

    def test_missing_prompt_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompt_contract(tmp_path / "absent.md", "v1")


class TestResponseSchema:
    def test_schema_requires_exactly_the_contract_fields(self):
        assert tuple(RESPONSE_JSON_SCHEMA["required"]) == REQUIRED_RESPONSE_FIELDS
        assert set(RESPONSE_JSON_SCHEMA["properties"]) == set(REQUIRED_RESPONSE_FIELDS)

    def test_schema_and_pydantic_model_agree(self):
        assert set(RESPONSE_JSON_SCHEMA["properties"]) <= set(
            ExtractionResponse.model_fields
        )

    def test_schema_is_json_serializable(self):
        json.dumps(RESPONSE_JSON_SCHEMA)

    def test_country_candidates_is_a_list_not_a_scalar(self):
        """Python collapses candidates; the model must never be asked to choose."""
        assert RESPONSE_JSON_SCHEMA["properties"]["country_candidates"]["type"] == "array"


class TestUserPayload:
    def test_payload_carries_address_and_reference_context(self):
        payload = json.loads(build_user_payload("BOSTON MA US", {"sources": ["iso3166"]}))
        assert payload["address"] == "BOSTON MA US"
        assert payload["reference_context"] == {"sources": ["iso3166"]}

    def test_absent_reference_context_is_an_empty_object(self):
        payload = json.loads(build_user_payload("BOSTON MA US"))
        assert payload["reference_context"] == {}

    def test_payload_is_deterministic(self):
        first = build_user_payload("ACCRA GH", {"b": 1, "a": 2})
        second = build_user_payload("ACCRA GH", {"a": 2, "b": 1})
        assert first == second      # stable cache/audit behaviour


class TestValidResponses:
    def test_happy_path(self):
        response = parse_extraction_response(valid_payload())
        assert response.town == "AUCKLAND"
        assert response.country_candidates == ("NZ",)
        assert response.has_town and response.has_country

    def test_json_string_is_accepted(self):
        response = parse_extraction_response(json.dumps(valid_payload()))
        assert response.town == "AUCKLAND"

    def test_town_is_uppercased_and_whitespace_collapsed(self):
        response = parse_extraction_response(valid_payload(town="  new   york "))
        assert response.town == "NEW YORK"

    def test_lowercase_codes_are_normalized(self):
        response = parse_extraction_response(valid_payload(country_candidates=["nz", "au"]))
        assert response.country_candidates == ("NZ", "AU")

    def test_duplicate_candidates_are_removed(self):
        response = parse_extraction_response(
            valid_payload(country_candidates=["US", "us", "US"])
        )
        assert response.country_candidates == ("US",)

    def test_no_country_sentinel_is_stripped_from_candidates(self):
        """The sentinel is the pipeline's to write, never a candidate."""
        response = parse_extraction_response(valid_payload(country_candidates=["NO_COUNTRY"]))
        assert response.country_candidates == ()
        assert response.has_country is False

    def test_empty_candidate_list_is_valid(self):
        response = parse_extraction_response(
            valid_payload(town=NO_TOWN, country_candidates=[])
        )
        assert response.has_country is False
        assert response.has_town is False

    def test_multiple_candidates_report_ambiguity(self):
        response = parse_extraction_response(valid_payload(country_candidates=["CA", "US"]))
        assert response.is_country_ambiguous is True

    def test_flag_alone_cannot_manufacture_ambiguity(self):
        response = parse_extraction_response(
            valid_payload(country_candidates=["US"], country_ambiguous=True)
        )
        assert response.is_country_ambiguous is False

    def test_extra_unknown_fields_are_ignored(self):
        response = parse_extraction_response(valid_payload(unexpected="ignored"))
        assert response.town == "AUCKLAND"


class TestMalformedResponses:
    def test_invalid_json(self):
        with pytest.raises(MalformedExtractionResponse, match="not valid JSON"):
            parse_extraction_response("{not json")

    def test_empty_response(self):
        with pytest.raises(MalformedExtractionResponse, match="empty response"):
            parse_extraction_response("   ")

    def test_truncated_json(self):
        with pytest.raises(MalformedExtractionResponse):
            parse_extraction_response('{"town": "BOSTON", "country_candidates": ["U')

    def test_non_object_top_level(self):
        with pytest.raises(MalformedExtractionResponse, match="must be a JSON object"):
            parse_extraction_response('["BOSTON"]')

    def test_missing_required_field(self):
        payload = valid_payload()
        del payload["country_candidates"]
        with pytest.raises(MalformedExtractionResponse, match="country_candidates"):
            parse_extraction_response(payload)

    @pytest.mark.parametrize("code", ["USA", "U", "12", "U5", "United States", ""])
    def test_non_alpha2_country_codes_are_rejected(self, code):
        if code == "":
            # An empty entry is dropped rather than being an error.
            assert parse_extraction_response(
                valid_payload(country_candidates=[code])
            ).country_candidates == ()
            return
        with pytest.raises(MalformedExtractionResponse, match="alpha-2"):
            parse_extraction_response(valid_payload(country_candidates=[code]))

    @pytest.mark.parametrize("confidence", [-0.1, 1.5, 42])
    def test_out_of_range_confidence_is_rejected(self, confidence):
        with pytest.raises(MalformedExtractionResponse, match="outside"):
            parse_extraction_response(valid_payload(town_model_confidence=confidence))

    def test_non_numeric_confidence_is_rejected(self):
        with pytest.raises(MalformedExtractionResponse):
            parse_extraction_response(valid_payload(country_model_confidence="high"))

    def test_non_boolean_flag_is_rejected(self):
        with pytest.raises(MalformedExtractionResponse):
            parse_extraction_response(valid_payload(town_is_explicit="maybe"))

    def test_raw_text_is_retained_for_debugging(self):
        with pytest.raises(MalformedExtractionResponse) as excinfo:
            parse_extraction_response("{bad}")
        assert excinfo.value.raw == "{bad}"


class TestAeronauticaContract:
    """The substring trap, end to end through the contract."""

    def test_expected_response_shape_for_aeronautica(self, iso_provider):
        from swift_address.scoring import verify_extraction

        response = parse_extraction_response(
            valid_payload(
                town=NO_TOWN,
                country_candidates=[],
                town_evidence="",
                country_evidence="",
                town_is_explicit=False,
                country_is_explicit=False,
                town_model_confidence=0.0,
                country_model_confidence=0.0,
                town_rationale="No defensible town in the supplied text.",
                country_rationale="No defensible country in the supplied text.",
            )
        )
        verified = verify_extraction(response, "AERONAUTICA", iso_provider=iso_provider)
        assert verified.town == NO_TOWN
        assert verified.country_value == NO_COUNTRY

    def test_a_rona_claim_is_stripped_of_its_explicitness(self, iso_provider):
        from swift_address.scoring import evaluate

        response = parse_extraction_response(
            valid_payload(
                town="RONA", town_evidence="RONA", town_is_explicit=True,
                country_candidates=["CA"], country_is_explicit=False,
                town_model_confidence=0.9, country_model_confidence=0.8,
            )
        )
        verified, result = evaluate(
            response, "AERONAUTICA", _scoring_config(), iso_provider=iso_provider
        )
        assert verified.town_exists is False
        assert verified.country_exists is False
        assert result.scenario == "neither_explicit_both_inferred"
        # Weight 0.20 x 0.20 keeps an unsupported claim far below any threshold.
        assert result.composite_weighted_score == pytest.approx(0.9 * 0.2 * 0.8 * 0.2)
        assert result.needs_hitl is True


def _scoring_config():
    from pathlib import Path

    from swift_address.settings import load_config

    root = Path(__file__).resolve().parents[1]
    return load_config(root / "config" / "config.yaml", base_dir=root).scoring
