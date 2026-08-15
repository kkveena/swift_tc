"""An empty combined address must cost nothing and reach no model.

The fake client here raises on any call, so a single stray request fails the
test loudly rather than showing up as a surprise on a quota bill.
"""

from __future__ import annotations

import pandas as pd
import pytest

from swift_address.gemini_client import ExtractionOutcome
from swift_address.grouping import AddressGroup, GroupConfig
from swift_address.pipeline import Phase1Pipeline
from swift_address.reference_data import NullReferenceDataProvider, ReferenceContext
from swift_address.schemas import NO_COUNTRY, NO_TOWN, parse_extraction_response


class ExplodingClient:
    """Any call at all is a test failure."""

    model = "exploding-test-client"

    def __init__(self) -> None:
        self._call_count = 0

    def extract(self, address: str, reference_context: ReferenceContext):
        self._call_count += 1
        raise AssertionError(
            f"Gemini was called for an address that should have been skipped: {address!r}"
        )

    @property
    def call_count(self) -> int:
        return self._call_count


class CountingClient:
    """Records calls and returns a fixed valid response."""

    model = "counting-test-client"

    def __init__(self) -> None:
        self.addresses: list[str] = []

    def extract(self, address: str, reference_context: ReferenceContext):
        self.addresses.append(address)
        return ExtractionOutcome(
            response=parse_extraction_response(
                {
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
                    "town_rationale": "explicit",
                    "country_rationale": "explicit",
                    "reference_basis": ["input_text"],
                }
            ),
            model=self.model,
        )

    @property
    def call_count(self) -> int:
        return len(self.addresses)


def _two_group_config() -> GroupConfig:
    return GroupConfig(
        groups=(
            AddressGroup(group_id="1", source_fields=("A1", "A2", "A3")),
            AddressGroup(group_id="2", source_fields=("B1", "B2", "B3")),
        )
    )


def _all_empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RECORD_ID": ["CA0000000863", "CA0000000864"],
            "A1": ["", "0"],
            "A2": ["0", "   "],
            "A3": ["   ", ""],
            "B1": ["", "0"],
            "B2": ["", ""],
            "B3": ["0", "0"],
        }
    )


@pytest.fixture
def pipeline_factory(config):
    def build(group_config, client):
        return Phase1Pipeline(
            config,
            group_config,
            client=client,
            reference_provider=NullReferenceDataProvider(),
            mode="test",
        )

    return build


class TestZeroCallsForEmptyAddresses:
    def test_no_model_call_when_every_group_is_empty(self, pipeline_factory):
        client = ExplodingClient()
        result = pipeline_factory(_two_group_config(), client).run(_all_empty_frame())

        assert client.call_count == 0
        assert result.pass1.empty_instances == 4
        assert result.pass1.non_empty_instances == 0
        assert result.pass1.unique_addresses == 0

    def test_null_defaults_are_written_exactly(self, pipeline_factory):
        result = pipeline_factory(_two_group_config(), ExplodingClient()).run(
            _all_empty_frame()
        )
        row = result.frame.iloc[0]

        for group_id in ("1", "2"):
            assert row[f"combined_address_group_{group_id}"] == ""
            assert row[f"combined_address_cleaned_group_{group_id}"] == ""
            assert row[f"predicted_town_group_{group_id}"] == NO_TOWN
            assert row[f"predicted_country_group_{group_id}"] == NO_COUNTRY
            assert row[f"predicted_town_probability_group_{group_id}"] == 0.0
            assert row[f"predicted_country_probability_group_{group_id}"] == 0.0
            assert bool(row[f"predicted_town_exists_group_{group_id}"]) is False
            assert bool(row[f"predicted_country_exists_group_{group_id}"]) is False
            assert row[f"composite_weighted_score_group_{group_id}"] == 0.0
            assert row[f"rationale_town_group_{group_id}"] == ""
            assert row[f"rationale_country_group_{group_id}"] == ""

    def test_null_rows_are_not_routed_to_hitl(self, pipeline_factory):
        result = pipeline_factory(_two_group_config(), ExplodingClient()).run(
            _all_empty_frame()
        )
        # A skipped null address is a known, complete answer, not a review item.
        assert result.metrics["hitl"]["instances_below_threshold"] == 0
        assert result.metrics["outcomes"]["scenario_counts"]["null_skip"] == 4

    def test_only_the_non_empty_group_is_enqueued(self, pipeline_factory):
        client = CountingClient()
        frame = pd.DataFrame(
            {
                "RECORD_ID": ["R1"],
                "A1": ["1 LINCOLN STREET"],
                "A2": ["BOSTON MA 02111 US"],
                "A3": ["0"],
                "B1": ["0"],
                "B2": [""],
                "B3": ["   "],
            }
        )
        result = pipeline_factory(_two_group_config(), client).run(frame)

        assert client.call_count == 1
        assert client.addresses == ["1 LINCOLN STREET BOSTON MA 02111 US"]
        assert result.frame.loc[0, "predicted_town_group_1"] == "BOSTON"
        assert result.frame.loc[0, "predicted_town_group_2"] == NO_TOWN

    def test_zero_only_group_is_treated_as_empty(self, pipeline_factory):
        client = ExplodingClient()
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["0"], "A2": ["0"], "A3": ["0"],
             "B1": ["0"], "B2": ["0"], "B3": ["0"]}
        )
        result = pipeline_factory(_two_group_config(), client).run(frame)

        assert client.call_count == 0
        assert result.frame.loc[0, "combined_address_group_1"] == ""

    def test_sample_input_skips_every_unmapped_group(
        self, config, group_config, sample_input_path
    ):
        """The real 50-column sample: only group 15 carries data."""
        from swift_address.io import read_input_csv

        client = CountingClient()
        pipeline = Phase1Pipeline(
            config,
            group_config,
            client=client,
            reference_provider=NullReferenceDataProvider(),
            mode="test",
        )
        result = pipeline.run(read_input_csv(sample_input_path))

        # 8 rows x 16 groups = 128 instances; only 7 non-empty, all in group 15.
        assert result.pass1.total_instances == 128
        assert result.pass1.empty_instances == 121
        assert result.pass1.non_empty_instances == 7
        assert client.call_count == 7
        assert result.metrics["efficiency"]["calls_avoided_by_null_skip"] == 121
