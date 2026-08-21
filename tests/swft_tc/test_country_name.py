"""Deterministic Country Name expansion.

The name column is derived from the ISO reference layer, never predicted. Its
one hard contract is element-for-element alignment with the code column.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.swft_tc.src.pipeline import Phase1Pipeline
from models.swft_tc.src.reference_data import NullReferenceDataProvider
from models.swft_tc.src.schemas import NO_COUNTRY, NO_TOWN, parse_extraction_response
from models.swft_tc.src.scoring import error_result, null_result, verify_extraction

from test_pipeline import BOSTON_RESPONSE, two_group_config


def make_response(**overrides):
    payload = dict(BOSTON_RESPONSE)
    payload.update(overrides)
    return parse_extraction_response(payload)


class TestCodeToNameExpansion:
    def test_single_code_expands(self, iso_provider):
        assert iso_provider.country_name("US") == "United States"
        assert iso_provider.country_name("CA") == "Canada"
        assert iso_provider.country_name("NZ") == "New Zealand"

    def test_expansion_is_deterministic(self, iso_provider):
        assert iso_provider.country_name("US") == iso_provider.country_name("us")

    def test_unknown_code_falls_back_to_the_code(self, iso_provider):
        """Never blank: a name column that silently empties loses information."""
        assert iso_provider.country_name("ZZ") == "ZZ"

    def test_list_expansion_preserves_order_and_length(self, iso_provider):
        codes = ("CA", "US", "NZ")
        names = iso_provider.country_names(codes)
        assert len(names) == len(codes)
        assert names == ("Canada", "United States", "New Zealand")


class TestVerifiedExtractionCountryName:
    def test_scalar_country(self, iso_provider):
        verified = verify_extraction(
            make_response(), "1 LINCOLN STREET BOSTON MA 02111 US",
            iso_provider=iso_provider,
        )
        assert verified.country_value == "US"
        assert verified.country_name_value == "United States"

    def test_ambiguous_codes_and_names_stay_aligned(self, iso_provider):
        verified = verify_extraction(
            make_response(town="EXAMPLE_TOWN", country_candidates=["US", "CA"]),
            "CONTROLLED MULTI COUNTRY FIXTURE EXAMPLE_TOWN",
            iso_provider=iso_provider,
        )
        assert verified.country_value == "CA,US"
        assert verified.country_name_value == "Canada,United States"

        codes = verified.country_value.split(",")
        names = verified.country_name_value.split(",")
        assert len(codes) == len(names) == 2
        assert dict(zip(codes, names)) == {"CA": "Canada", "US": "United States"}

    def test_three_way_ambiguity_stays_aligned(self, iso_provider):
        verified = verify_extraction(
            make_response(town="HAMILTON", country_candidates=["NZ", "BM", "CA"]),
            "HAMILTON BRANCH",
            iso_provider=iso_provider,
        )
        assert verified.country_value == "BM,CA,NZ"
        assert verified.country_name_value == "Bermuda,Canada,New Zealand"

    def test_no_country_produces_no_country_name(self, iso_provider):
        verified = verify_extraction(
            make_response(town=NO_TOWN, country_candidates=[]),
            "AERONAUTICA",
            iso_provider=iso_provider,
        )
        assert verified.country_value == NO_COUNTRY
        assert verified.country_name_value == NO_COUNTRY

    def test_null_result_uses_the_sentinel(self):
        verified, _ = null_result()
        assert verified.country_name_value == NO_COUNTRY

    def test_error_result_uses_the_sentinel(self):
        verified, _ = error_result("timeout")
        assert verified.country_name_value == NO_COUNTRY

    def test_codes_stand_in_when_no_iso_provider(self):
        """Alignment survives even without a reference dataset."""
        verified = verify_extraction(
            make_response(town="EXAMPLE_TOWN", country_candidates=["US", "CA"]),
            "EXAMPLE_TOWN",
            iso_provider=None,
        )
        assert verified.country_value == "CA,US"
        assert verified.country_name_value == "CA,US"

    def test_no_iso_name_contains_the_separator(self, iso_provider):
        """A comma inside a name would add a phantom element to the name list."""
        offenders = [
            code for code in ("TW", "KR", "MD", "TZ", "PS", "CD", "BQ", "SH")
            if "," in iso_provider.country_name(code)
        ]
        assert offenders == []

    def test_separator_inside_a_name_is_folded(self, iso_provider):
        """Guard for a replacement dataset that reintroduces inverted names."""
        from models.swft_tc.src.reference_data import CountryRecord, Iso3166Provider

        provider = Iso3166Provider(
            [CountryRecord(alpha2="TW", name="Taiwan, Province of China"),
             CountryRecord(alpha2="CA", name="Canada")],
            version="test",
        )
        verified = verify_extraction(
            make_response(town="HAMILTON", country_candidates=["TW", "CA"]),
            "HAMILTON BRANCH",
            iso_provider=provider,
        )
        assert verified.country_value == "CA,TW"
        assert verified.country_name_value == "Canada,Taiwan Province of China"
        assert len(verified.country_value.split(",")) == (
            len(verified.country_name_value.split(","))
        )

    def test_name_is_never_requested_from_the_model(self):
        """The response schema has no country-name field to fill in."""
        from models.swft_tc.src.schemas import REQUIRED_RESPONSE_FIELDS, RESPONSE_JSON_SCHEMA

        assert not any("name" in field for field in REQUIRED_RESPONSE_FIELDS)
        assert "country_name" not in RESPONSE_JSON_SCHEMA["properties"]


class TestCountryNameInOutput:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {"RECORD_ID": ["R1", "R2"],
             "A1": ["1 LINCOLN STREET", ""],
             "A2": ["BOSTON MA 02111 US", "0"],
             "B1": ["", ""], "B2": ["", ""]}
        )

    def _run(self, config, frame, client, reference_provider):
        return Phase1Pipeline(
            config, two_group_config(), client=client,
            reference_provider=reference_provider, mode="test",
        ).run(frame)

    def test_column_is_present_and_populated(self, config, frame, reference_provider):
        from models.swft_tc.src.gemini_client import ScriptedExtractionClient

        result = self._run(
            config, frame, ScriptedExtractionClient({}, default=BOSTON_RESPONSE),
            reference_provider,
        )
        assert "predicted_country_name_group_1" in result.frame.columns
        assert result.frame.loc[0, "predicted_country_group_1"] == "US"
        assert result.frame.loc[0, "predicted_country_name_group_1"] == "United States"

    def test_empty_group_gets_the_sentinel(self, config, frame, reference_provider):
        from models.swft_tc.src.gemini_client import ScriptedExtractionClient

        result = self._run(
            config, frame, ScriptedExtractionClient({}, default=BOSTON_RESPONSE),
            reference_provider,
        )
        assert result.frame.loc[1, "predicted_country_group_1"] == NO_COUNTRY
        assert result.frame.loc[1, "predicted_country_name_group_1"] == NO_COUNTRY

    def test_column_round_trips_through_csv(
        self, config, frame, tmp_path, reference_provider
    ):
        from models.swft_tc.src.gemini_client import ScriptedExtractionClient
        from models.swft_tc.src.io import read_output_csv, write_output_csv

        result = self._run(
            config, frame, ScriptedExtractionClient({}, default=BOSTON_RESPONSE),
            reference_provider,
        )
        reloaded = read_output_csv(write_output_csv(result.frame, tmp_path / "o.csv"))
        assert reloaded.loc[0, "predicted_country_name_group_1"] == "United States"

    def test_codes_stand_in_without_an_iso_provider(self, config, frame):
        """The documented fallback: never blank, always aligned."""
        from models.swft_tc.src.gemini_client import ScriptedExtractionClient

        result = self._run(
            config, frame, ScriptedExtractionClient({}, default=BOSTON_RESPONSE),
            NullReferenceDataProvider(),
        )
        assert result.frame.loc[0, "predicted_country_name_group_1"] == "US"

    def test_every_row_keeps_code_and_name_aligned(self, config, frame, reference_provider):
        from models.swft_tc.src.gemini_client import ScriptedExtractionClient

        result = self._run(
            config, frame, ScriptedExtractionClient({}, default=BOSTON_RESPONSE),
            reference_provider,
        )
        for group in ("1", "2"):
            codes = result.frame[f"predicted_country_group_{group}"]
            names = result.frame[f"predicted_country_name_group_{group}"]
            for code_value, name_value in zip(codes, names):
                assert len(code_value.split(",")) == len(name_value.split(","))
