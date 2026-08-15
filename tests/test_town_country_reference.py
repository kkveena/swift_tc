"""The Town/Country development reference provider and its validation effects.

Tests use the tiny committed fixture, never the real reference file: that file
is a large, environment-specific external runtime dependency and is not in
version control.
"""

from __future__ import annotations

import pytest

from swift_address.reference_data import (
    TownCountryProvider,
    TownCountryReferenceError,
    build_town_country_provider,
    resolve_town_country_file,
)
from swift_address.schemas import NO_TOWN, parse_extraction_response
from swift_address.scoring import (
    REFERENCE_CONFLICT,
    REFERENCE_CONSISTENT,
    REFERENCE_MULTI_ANNOTATED,
    REFERENCE_MULTI_ESCALATED,
    REFERENCE_NOT_FOUND,
    REFERENCE_NO_TOWN,
    REFERENCE_SUPPLIED,
    evaluate,
    verify_extraction,
)

from test_pipeline import BOSTON_RESPONSE


def make_response(**overrides):
    payload = dict(BOSTON_RESPONSE)
    payload.update(overrides)
    return parse_extraction_response(payload)


class TestLoading:
    def test_loads_the_fixture_once(self, town_country_provider):
        assert town_country_provider is not None
        assert len(town_country_provider) > 0

    def test_provenance_is_reported(self, town_country_provider):
        provenance = town_country_provider.provenance
        assert provenance["provider"] == "town_country"
        assert provenance["rows"] == 17
        assert provenance["source_dataset"] == "test_fixture"
        assert provenance["source_version"] == "test-1"

    def test_development_reference_is_never_production_approved(
        self, town_country_provider
    ):
        assert town_country_provider.provenance["approved_for_production"] is False

    def test_file_claiming_approval_cannot_override_configuration(
        self, tmp_path, town_country_fixture_path
    ):
        """Both the operator and the file must agree before this reads approved."""
        text = town_country_fixture_path.read_text(encoding="utf-8").replace(
            ",false", ",true"
        )
        path = tmp_path / "claims_approval.csv"
        path.write_text(text, encoding="utf-8")

        provider = TownCountryProvider.from_file(
            path, version="v", approved_for_production=False
        )
        assert provider.provenance["approved_for_production"] is False

    def test_disabled_returns_no_provider(self, config):
        disabled = config.reference_data.model_copy(
            update={"town_country_enabled": False}
        )
        assert build_town_country_provider(disabled, base_dir=config.base_dir) is None

    def test_missing_file_fails_fast_with_guidance(self, config, tmp_path):
        """No silent fallback to web search or model geography."""
        missing = config.reference_data.model_copy(
            update={"town_country_path": str(tmp_path / "absent_reference.csv")}
        )
        with pytest.raises(TownCountryReferenceError) as excinfo:
            build_town_country_provider(missing, base_dir=config.base_dir)

        message = str(excinfo.value)
        assert "not found" in message
        assert "town_country_enabled" in message
        assert "build_geonames_town_country_reference.py" in message
        assert "will not substitute web search" in message

    def test_extensionless_path_resolves_to_the_csv(self, tmp_path):
        """The brief configures an extension-less path; the file carries .csv."""
        (tmp_path / "town_country_reference.csv").write_text(
            "town_name,town_name_normalized,country_code\nZurich,ZURICH,CH\n",
            encoding="utf-8",
        )
        resolved = resolve_town_country_file(tmp_path / "town_country_reference")
        assert resolved.name == "town_country_reference.csv"

    def test_tab_separated_file_is_detected(self, tmp_path):
        path = tmp_path / "ref.tsv"
        path.write_text(
            "town_name\ttown_name_normalized\tcountry_code\nZurich\tZURICH\tCH\n",
            encoding="utf-8",
        )
        provider = TownCountryProvider.from_file(path, version="v")
        assert provider.lookup_country_codes("ZURICH") == ("CH",)

    def test_differing_headers_are_mapped_not_rewritten(self, tmp_path):
        path = tmp_path / "vendor.csv"
        original = "city,iso2\nZurich,CH\nHamilton,BM\nHamilton,NZ\n"
        path.write_text(original, encoding="utf-8")

        provider = TownCountryProvider.from_file(
            path, version="v",
            column_map={"city": "town_name", "iso2": "country_code"},
        )
        assert provider.lookup_country_codes("ZURICH") == ("CH",)
        assert provider.lookup_country_codes("HAMILTON") == ("BM", "NZ")
        assert path.read_text(encoding="utf-8") == original   # never mutated

    def test_unmappable_headers_are_reported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("place,nation\nZurich,CH\n", encoding="utf-8")
        with pytest.raises(TownCountryReferenceError, match="town_country_column_map"):
            TownCountryProvider.from_file(path, version="v")


class TestLookup:
    def test_single_country_town(self, town_country_provider):
        assert town_country_provider.lookup_country_codes("AUCKLAND") == ("NZ",)

    def test_multi_country_town_returns_sorted_distinct_codes(self, town_country_provider):
        assert town_country_provider.lookup_country_codes("HAMILTON") == ("BM", "CA", "NZ")

    def test_lookup_is_case_insensitive(self, town_country_provider):
        assert town_country_provider.lookup_country_codes("auckland") == ("NZ",)

    def test_punctuation_folded_form_matches(self, town_country_provider):
        """The file holds SAINT-DENIS; a predicted SAINT DENIS still matches."""
        assert town_country_provider.lookup_country_codes("SAINT DENIS") == ("FR",)
        assert town_country_provider.lookup_country_codes("SAINT-DENIS") == ("FR",)

    def test_unknown_town_returns_empty(self, town_country_provider):
        assert town_country_provider.lookup_country_codes("NOWHERESVILLE") == ()
        assert town_country_provider.knows("NOWHERESVILLE") is False

    def test_empty_town_returns_empty(self, town_country_provider):
        assert town_country_provider.lookup_country_codes("") == ()

    def test_index_is_built_once_not_scanned_per_lookup(self, town_country_provider):
        """Lookups hit a dict; the file is not re-read."""
        assert isinstance(town_country_provider._index, dict)
        town_country_provider._index.clear()
        assert town_country_provider.lookup_country_codes("AUCKLAND") == ()

    def test_provider_supplies_no_prompt_context(self, town_country_provider):
        """Nothing goes into the prompt, so the model cannot cite it."""
        context = town_country_provider.get_context("1 LINCOLN STREET BOSTON MA US")
        assert context.is_empty


class TestValidationEffects:
    def test_single_country_town_validates_as_consistent(
        self, iso_provider, town_country_provider
    ):
        verified = verify_extraction(
            make_response(town="AUCKLAND", country_candidates=["NZ"]),
            "23 CUSTOMS STREET AUCKLAND 1140 NZ",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_CONSISTENT
        assert verified.country_value == "NZ"
        assert verified.country_ambiguous is False

    def test_explicit_country_evidence_beats_reference_multiplicity(
        self, iso_provider, town_country_provider
    ):
        """BOSTON spans US and GB in the reference, but the address says US."""
        verified = verify_extraction(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_CONSISTENT
        assert verified.country_value == "US"
        assert verified.country_ambiguous is False

    def test_multi_country_town_without_explicit_evidence_escalates(
        self, iso_provider, town_country_provider
    ):
        """LIMA spans PE and US; the address states no country."""
        verified = verify_extraction(
            make_response(
                town="LIMA", country_candidates=["PE"], country_is_explicit=False
            ),
            "441-445 JIRON SANTA ROSA LIMA METRO MUNIC OF LIMA 15001",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_MULTI_ESCALATED
        assert verified.country_value == "PE,US"
        assert verified.country_name_value == "Peru,United States"
        assert verified.country_ambiguous is True
        assert verified.country_probability == 0.0

    def test_escalated_ambiguity_scores_zero_and_forces_hitl(
        self, iso_provider, town_country_provider, config
    ):
        _, result = evaluate(
            make_response(
                town="LIMA", country_candidates=["PE"], country_is_explicit=False
            ),
            "441-445 JIRON SANTA ROSA LIMA 15001",
            config.scoring,
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert result.country_weight == 0.0
        assert result.composite_weighted_score == 0.0
        assert result.needs_hitl is True

    def test_annotate_policy_leaves_the_prediction_untouched(
        self, iso_provider, town_country_provider
    ):
        verified = verify_extraction(
            make_response(
                town="LIMA", country_candidates=["PE"], country_is_explicit=False
            ),
            "441-445 JIRON SANTA ROSA LIMA 15001",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
            town_country_ambiguity_policy="annotate",
        )
        assert verified.reference_status == REFERENCE_MULTI_ANNOTATED
        assert verified.country_value == "PE"
        assert verified.country_ambiguous is False
        assert any(n.startswith("reference_multi_country:") for n in verified.notes)

    def test_conflict_does_not_overwrite_the_model(
        self, iso_provider, town_country_provider, config
    ):
        """The reference says Zurich is CH; the model says FR. Neither wins silently."""
        verified, result = evaluate(
            make_response(
                town="ZURICH", country_candidates=["FR"], country_is_explicit=False
            ),
            "BAHNHOFSTRASSE 1 ZURICH",
            config.scoring,
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_CONFLICT
        assert verified.country_value == "FR"          # model output preserved
        assert verified.reference_codes == ("CH",)     # reference finding preserved
        assert result.needs_hitl is True
        assert any(n.startswith("reference_conflict:") for n in verified.notes)

    def test_explicit_evidence_conflicting_with_reference_flags_hitl(
        self, iso_provider, town_country_provider, config
    ):
        verified, result = evaluate(
            make_response(town="AUCKLAND", country_candidates=["FR"]),
            "23 CUSTOMS STREET AUCKLAND FR",
            config.scoring,
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_CONFLICT
        assert verified.country_value == "FR"
        assert result.needs_hitl is True

    def test_missing_town_is_not_an_extraction_error(
        self, iso_provider, town_country_provider, config
    ):
        verified, result = evaluate(
            make_response(
                town="NOWHERESVILLE", country_candidates=["US"],
                country_is_explicit=False,
            ),
            "1 MAIN STREET NOWHERESVILLE",
            config.scoring,
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_NOT_FOUND
        assert result.scenario != "extraction_error"
        assert verified.country_value == "US"       # untouched
        assert "reference_not_found" in verified.notes

    def test_no_predicted_town_skips_the_lookup(
        self, iso_provider, town_country_provider
    ):
        verified = verify_extraction(
            make_response(town=NO_TOWN, country_candidates=[]),
            "AERONAUTICA",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_NO_TOWN

    def test_reference_fills_a_gap_the_model_left_empty(
        self, iso_provider, town_country_provider
    ):
        verified = verify_extraction(
            make_response(
                town="AUCKLAND", country_candidates=[], country_is_explicit=False,
            ),
            "23 CUSTOMS STREET AUCKLAND",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.reference_status == REFERENCE_SUPPLIED
        assert verified.country_value == "NZ"
        assert verified.country_name_value == "New Zealand"
        assert "country_supplied_by_reference:NZ" in verified.notes

    def test_aeronautica_is_unaffected_by_the_reference(
        self, iso_provider, town_country_provider, config
    ):
        """The anti-substring guarantee still holds with the reference enabled."""
        verified, result = evaluate(
            make_response(
                town="RONA", town_evidence="RONA", town_is_explicit=True,
                country_candidates=[], country_is_explicit=False,
            ),
            "AERONAUTICA",
            config.scoring,
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
        )
        assert verified.town_exists is False
        assert "town_explicit_claim_unverified" in verified.notes

    def test_candidate_cap_truncates_only_the_written_value(
        self, iso_provider, town_country_provider
    ):
        verified = verify_extraction(
            make_response(town="HAMILTON", country_candidates=[], country_is_explicit=False),
            "HAMILTON BRANCH OFFICE",
            iso_provider=iso_provider,
            town_country_provider=town_country_provider,
            town_country_max_candidates=2,
        )
        assert verified.country_value == "BM,CA"
        assert verified.country_name_value == "Bermuda,Canada"
        assert verified.reference_codes == ("BM", "CA", "NZ")   # full set in audit
        assert any(n.startswith("country_candidates_truncated:") for n in verified.notes)


class TestPipelineIntegration:
    def test_reference_status_reaches_metrics_and_audit(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider,
            mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        town_country = result.metrics["reference_data"]["town_country"]
        assert town_country["loaded"] is True
        assert town_country["approved_for_production"] is False
        assert town_country["ambiguity_policy"] == "escalate"
        assert result.metrics["outcomes"]["reference_status_counts"]

        entry = next(iter(result.audit.values()))
        assert "reference_status" in entry["verified"]

    def test_reference_columns_are_not_added_to_the_csv(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from swift_address.io import read_input_csv
        from swift_address.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider,
            mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        assert not [c for c in result.frame.columns if "reference_status" in c]
        assert len(result.frame.columns) == 50 + 16 * config.fields_per_group
