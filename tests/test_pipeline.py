"""Pipeline behaviour: deduplication, caching, retry, and failure handling."""

from __future__ import annotations

import pandas as pd
import pytest

from swift_address.cache import AddressCache, make_cache_key
from swift_address.gemini_client import (
    GeminiClient,
    PermanentExtractionError,
    ScriptedExtractionClient,
    TransientExtractionError,
    is_transient_error,
)
from swift_address.grouping import AddressGroup, GroupConfig
from swift_address.pipeline import Phase1Pipeline
from swift_address.reference_data import NullReferenceDataProvider
from swift_address.schemas import NO_COUNTRY, NO_TOWN

BOSTON = "1 LINCOLN STREET BOSTON MA 02111 US"

BOSTON_RESPONSE = {
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


def two_group_config() -> GroupConfig:
    return GroupConfig(
        groups=(
            AddressGroup(group_id="1", source_fields=("A1", "A2")),
            AddressGroup(group_id="2", source_fields=("B1", "B2")),
        )
    )


def build_pipeline(config, client, group_config=None, cache=None):
    return Phase1Pipeline(
        config,
        group_config or two_group_config(),
        client=client,
        reference_provider=NullReferenceDataProvider(),
        cache=cache,
        mode="test",
    )


class TestDeduplication:
    def test_repeated_address_costs_one_call_across_rows_and_groups(self, config):
        """The same address in 3 rows x 2 groups is one unique cache miss."""
        frame = pd.DataFrame(
            {
                "RECORD_ID": ["R1", "R2", "R3"],
                "A1": ["1 LINCOLN STREET"] * 3,
                "A2": ["BOSTON MA 02111 US"] * 3,
                "B1": ["1 LINCOLN STREET"] * 3,
                "B2": ["BOSTON MA 02111 US"] * 3,
            }
        )
        client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        result = build_pipeline(config, client).run(frame)

        assert result.pass1.non_empty_instances == 6
        assert result.pass1.unique_addresses == 1
        assert client.call_count == 1
        assert result.metrics["efficiency"]["calls_avoided_by_dedupe"] == 5

    def test_the_single_result_is_mapped_to_every_occurrence(self, config):
        frame = pd.DataFrame(
            {
                "RECORD_ID": ["R1", "R2"],
                "A1": ["1 LINCOLN STREET"] * 2,
                "A2": ["BOSTON MA 02111 US"] * 2,
                "B1": ["1 LINCOLN STREET"] * 2,
                "B2": ["BOSTON MA 02111 US"] * 2,
            }
        )
        client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        result = build_pipeline(config, client).run(frame)

        for group in ("1", "2"):
            column = f"predicted_town_group_{group}"
            assert result.frame[column].tolist() == ["BOSTON", "BOSTON"]
            assert result.frame[f"composite_weighted_score_group_{group}"].tolist() == (
                pytest.approx([0.9801, 0.9801])
            )

    def test_whitespace_variants_collapse_to_one_call(self, config):
        """Cleaning happens before dedupe, so spacing differences are free."""
        frame = pd.DataFrame(
            {
                "RECORD_ID": ["R1", "R2"],
                "A1": ["1 LINCOLN   STREET", "1 LINCOLN STREET"],
                "A2": ["BOSTON MA 02111 US", " BOSTON MA 02111 US "],
                "B1": ["", ""],
                "B2": ["", ""],
            }
        )
        client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        result = build_pipeline(config, client).run(frame)

        assert client.call_count == 1
        assert result.pass1.unique_addresses == 1

    def test_distinct_addresses_are_not_merged(self, config):
        frame = pd.DataFrame(
            {
                "RECORD_ID": ["R1", "R2"],
                "A1": ["1 LINCOLN STREET", "388 GREENWICH STREET"],
                "A2": ["BOSTON MA 02111 US", "NEW YORK NY 10013-2632 US"],
                "B1": ["", ""],
                "B2": ["", ""],
            }
        )
        client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        build_pipeline(config, client).run(frame)
        assert client.call_count == 2


class TestCacheKey:
    def test_key_is_stable_for_the_same_inputs(self):
        args = dict(
            prompt_version="v2", model="gemini-3.5-flash", address=BOSTON,
            reference_context_version="iso-1",
        )
        assert make_cache_key(**args) == make_cache_key(**args)

    @pytest.mark.parametrize(
        "changed",
        [
            {"prompt_version": "v3"},
            {"model": "gemini-other"},
            {"address": "388 GREENWICH STREET"},
            {"reference_context_version": "iso-2"},
        ],
    )
    def test_every_component_changes_the_key(self, changed):
        base = dict(
            prompt_version="v2", model="gemini-3.5-flash", address=BOSTON,
            reference_context_version="iso-1",
        )
        assert make_cache_key(**base) != make_cache_key(**{**base, **changed})

    def test_case_and_spacing_do_not_change_the_key(self):
        base = dict(
            prompt_version="v2", model="m", reference_context_version="v",
        )
        assert make_cache_key(address="boston  ma", **base) == make_cache_key(
            address="BOSTON MA", **base
        )


class TestCachePersistence:
    def test_a_second_run_makes_no_calls(self, config, tmp_path):
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["1 LINCOLN STREET"],
             "A2": ["BOSTON MA 02111 US"], "B1": [""], "B2": [""]}
        )
        cache_path = tmp_path / "cache.jsonl"

        first_client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        build_pipeline(
            config, first_client, cache=AddressCache(cache_path)
        ).run(frame.copy())
        assert first_client.call_count == 1

        second_client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        result = build_pipeline(
            config, second_client, cache=AddressCache(cache_path)
        ).run(frame.copy())

        assert second_client.call_count == 0
        assert result.metrics["efficiency"]["cache_hits"] == 1
        assert result.frame.loc[0, "predicted_town_group_1"] == "BOSTON"

    def test_corrupt_cache_lines_are_skipped(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        path.write_text('{"bad json\n{"key": "k", "response": {}}\n', encoding="utf-8")
        cache = AddressCache(path)
        assert cache.load() == 1


class TestRetryBehaviour:
    """429 then success, with no duplicate rows and no duplicate results."""

    class _FakeSdkClient:
        """Minimal stand-in for the google-genai client object."""

        class _Rate429(Exception):
            status_code = 429

        def __init__(self, failures: int, body: str) -> None:
            self._remaining = failures
            self._body = body
            self.calls = 0
            self.models = self

        def generate_content(self, *, model, contents, config):  # noqa: ARG002
            self.calls += 1
            if self._remaining > 0:
                self._remaining -= 1
                raise self._Rate429("429 RESOURCE_EXHAUSTED: quota exceeded")
            return type("R", (), {"text": self._body, "usage_metadata": None})()

    def _gemini_client(self, prompt_contract, failures: int, body: str) -> GeminiClient:
        return GeminiClient(
            model="gemini-3.5-flash",
            prompt=prompt_contract,
            max_retries=3,
            retry_initial_seconds=0.001,
            retry_max_seconds=0.01,
            retry_jitter_seconds=0.001,
            client=self._FakeSdkClient(failures, body),
        )

    def test_429_then_success_produces_one_clean_row(self, config, prompt_contract):
        import json

        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["1 LINCOLN STREET"],
             "A2": ["BOSTON MA 02111 US"], "B1": [""], "B2": [""]}
        )
        client = self._gemini_client(prompt_contract, 1, json.dumps(BOSTON_RESPONSE))
        result = build_pipeline(config, client).run(frame)

        assert client.call_count == 2                      # one retry, then success
        assert len(result.frame) == 1                      # no duplicated rows
        assert result.errors == []
        assert result.frame.loc[0, "predicted_town_group_1"] == "BOSTON"
        assert result.frame.loc[0, "predicted_country_group_1"] == "US"
        assert result.metrics["outcomes"]["extraction_errors"] == 0

    def test_retries_are_bounded_and_then_recorded_as_an_error(
        self, config, prompt_contract
    ):
        import json

        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["1 LINCOLN STREET"],
             "A2": ["BOSTON MA 02111 US"], "B1": [""], "B2": [""]}
        )
        client = self._gemini_client(prompt_contract, 99, json.dumps(BOSTON_RESPONSE))
        result = build_pipeline(config, client).run(frame)

        assert client.call_count == 4                      # 1 attempt + 3 retries
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "TransientExtractionError"

    def test_malformed_output_is_retried_once_then_reported(self, config, prompt_contract):
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["1 LINCOLN STREET"],
             "A2": ["BOSTON MA 02111 US"], "B1": [""], "B2": [""]}
        )
        client = self._gemini_client(prompt_contract, 0, "{not json at all")
        result = build_pipeline(config, client).run(frame)

        # Two generation attempts, not an unbounded loop.
        assert client.call_count == 2
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "MalformedExtractionResponse"


class TestTransientClassification:
    @pytest.mark.parametrize(
        "exc",
        [
            TransientExtractionError("boom"),
            TimeoutError("timed out"),
            ConnectionError("connection reset"),
            Exception("429 RESOURCE_EXHAUSTED"),
            Exception("503 UNAVAILABLE"),
            Exception("500 internal error"),
        ],
    )
    def test_transient(self, exc):
        assert is_transient_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            PermanentExtractionError("bad key"),
            Exception("400 INVALID_ARGUMENT: bad request"),
            Exception("403 PERMISSION_DENIED"),
            ValueError("schema mismatch"),
        ],
    )
    def test_permanent(self, exc):
        assert is_transient_error(exc) is False

    def test_status_code_attribute_is_honoured(self):
        exc = type("E", (Exception,), {"status_code": 503})("service issue")
        assert is_transient_error(exc) is True


class TestFailureHandling:
    def test_failed_rows_get_neutral_values_and_are_routed_to_hitl(self, config):
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1", "R2"],
             "A1": ["1 LINCOLN STREET", "388 GREENWICH STREET"],
             "A2": ["BOSTON MA 02111 US", "NEW YORK NY 10013-2632 US"],
             "B1": ["", ""], "B2": ["", ""]}
        )
        client = ScriptedExtractionClient(
            {"1 LINCOLN STREET BOSTON MA 02111 US": [BOSTON_RESPONSE]},
            default=PermanentExtractionError("401 UNAUTHENTICATED"),
        )
        result = build_pipeline(config, client).run(frame)

        assert len(result.frame) == 2                      # the row survives
        assert result.frame.loc[1, "predicted_town_group_1"] == NO_TOWN
        assert result.frame.loc[1, "predicted_country_group_1"] == NO_COUNTRY
        assert result.frame.loc[1, "composite_weighted_score_group_1"] == 0.0
        assert result.frame.loc[0, "predicted_town_group_1"] == "BOSTON"

    def test_failure_is_visible_in_metrics_and_the_sidecar(self, config):
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["ADDRESS"], "A2": ["THAT FAILS"],
             "B1": ["ADDRESS"], "B2": ["THAT FAILS"]}
        )
        client = ScriptedExtractionClient(
            {}, default=PermanentExtractionError("401 UNAUTHENTICATED")
        )
        result = build_pipeline(config, client).run(frame)

        assert result.metrics["outcomes"]["extraction_errors"] == 1
        assert result.metrics["outcomes"]["instances_affected_by_errors"] == 2
        error = result.errors[0]
        assert error.occurrences == 2
        assert error.group_ids == ("1", "2")
        assert error.record_ids == ("R1",)

    def test_failure_does_not_masquerade_as_a_model_conclusion(self, config):
        """No rationale text is invented for a call that never returned."""
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["ADDRESS"], "A2": ["THAT FAILS"],
             "B1": [""], "B2": [""]}
        )
        client = ScriptedExtractionClient(
            {}, default=PermanentExtractionError("401 UNAUTHENTICATED")
        )
        result = build_pipeline(config, client).run(frame)
        assert result.frame.loc[0, "rationale_town_group_1"] == ""
        assert result.frame.loc[0, "rationale_country_group_1"] == ""


class TestGovernanceGuards:
    def test_google_search_grounding_is_refused(self, prompt_contract):
        with pytest.raises(PermanentExtractionError, match="data-governance"):
            GeminiClient(
                model="gemini-3.5-flash",
                prompt=prompt_contract,
                enable_google_search_grounding=True,
                client=object(),
            )

    def test_swiftref_provider_refuses_without_entitled_access(self):
        from swift_address.reference_data import (
            SwiftRefNotConfiguredError,
            SwiftRefProvider,
        )

        provider = SwiftRefProvider()
        assert provider.is_configured is False
        with pytest.raises(SwiftRefNotConfiguredError, match="licensed"):
            provider.get_context(BOSTON)

    def test_composite_provider_survives_an_unconfigured_swiftref(self, iso_provider):
        from swift_address.reference_data import (
            CompositeReferenceDataProvider,
            SwiftRefProvider,
        )

        composite = CompositeReferenceDataProvider([iso_provider, SwiftRefProvider()])
        context = composite.get_context(BOSTON)
        assert context.payload["iso3166_codes_present_in_address"] == ["US"]

    def test_empty_reference_context_says_so_explicitly(self):
        context = NullReferenceDataProvider().get_context(BOSTON)
        assert context.is_empty
        assert "do not claim any external lookup" in context.to_prompt_dict()["note"]

    def test_logs_reference_addresses_by_hash(self, config):
        pipeline = build_pipeline(config, ScriptedExtractionClient({}))
        assert pipeline._log_ref(BOSTON).startswith("sha256:")
        assert BOSTON not in pipeline._log_ref(BOSTON)


class TestMetrics:
    def test_metrics_report_reference_provenance(self, config, group_config, mock_client, reference_provider, sample_input_path):
        from swift_address.io import read_input_csv

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        provenance = result.metrics["reference_data"]["provenance"]
        assert provenance["approved_for_production"] is False
        assert provenance["records"] == 249
        assert result.metrics["reference_data"]["google_search_grounding"] is False

    def test_audit_retains_scenario_and_candidates_outside_the_csv(self, config):
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["1 LINCOLN STREET"],
             "A2": ["BOSTON MA 02111 US"], "B1": [""], "B2": [""]}
        )
        client = ScriptedExtractionClient({}, default=BOSTON_RESPONSE)
        result = build_pipeline(config, client).run(frame)

        entry = next(iter(result.audit.values()))
        assert entry["score"]["scenario"] == "both_explicit"
        assert entry["verified"]["country_candidates"] == ["US"]
        assert "address" not in entry            # audit references by hash only
        assert len(entry["address_hash"]) == 64
