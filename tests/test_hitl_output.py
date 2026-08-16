"""HITL fields in the CSV, the detailed JSON, and the state-distribution report."""

from __future__ import annotations

import pandas as pd
import pytest

from swift_address.io import read_input_csv
from swift_address.pipeline import Phase1Pipeline
from swift_address.reporting import build_hitl_state_distribution, write_reports
from swift_address.scoring import (
    HITL_AUTO_ACCEPT_CANDIDATE,
    HITL_STATE_PRECEDENCE,
)
from swift_address.serialization import read_detailed_jsonl, write_detailed_json

GROUP = "15"


@pytest.fixture
def run_result(
    config, group_config, reference_provider, town_country_provider, mock_client,
    sample_input_path,
):
    return Phase1Pipeline(
        config, group_config, client=mock_client,
        reference_provider=reference_provider,
        town_country_provider=town_country_provider, mode="dry_run",
    ).run(read_input_csv(sample_input_path))


@pytest.fixture
def documents(run_result, config, group_config, reference_provider, tmp_path):
    from swift_address.reference_data import find_iso_provider

    path = write_detailed_json(
        run_result.frame, tmp_path / "detail.jsonl",
        config=config, group_config=group_config,
        decisions_by_address=run_result.decisions_by_address,
        iso_provider=find_iso_provider(reference_provider),
    )
    return list(read_detailed_jsonl(path))


class TestCsvColumns:
    def test_three_hitl_columns_exist_per_group(self, run_result, config, group_config):
        for group in group_config.enabled_groups:
            for key in ("hitl_flag", "hitl_state", "hitl_state_reason"):
                assert config.output.column_name(key, group.group_id) in (
                    run_result.frame.columns
                )

    def test_flag_is_a_plain_boolean(self, run_result, config):
        column = config.output.column_name("hitl_flag", GROUP)
        assert pd.api.types.is_bool_dtype(run_result.frame[column])

    def test_state_values_are_from_the_closed_enum(
        self, run_result, config, group_config
    ):
        allowed = set(HITL_STATE_PRECEDENCE) | {""}
        for group in group_config.enabled_groups:
            column = config.output.column_name("hitl_state", group.group_id)
            assert set(run_result.frame[column]) <= allowed

    def test_populated_rows_carry_a_state_and_a_reason(self, run_result, config):
        frame = run_result.frame.set_index("RECORD_ID")
        row = frame.loc["CA0000001855"]        # AUCKLAND / NZ, both explicit
        assert row[config.output.column_name("hitl_state", GROUP)] == (
            HITL_AUTO_ACCEPT_CANDIDATE
        )
        assert bool(row[config.output.column_name("hitl_flag", GROUP)]) is False
        assert "meets configured HITL threshold" in (
            row[config.output.column_name("hitl_state_reason", GROUP)]
        )

    def test_reason_never_mentions_the_analytical_recommendation(
        self, run_result, config
    ):
        """The reason quotes the operational threshold, not the 0.90 advisory."""
        column = config.output.column_name("hitl_state_reason", GROUP)
        for reason in run_result.frame[column]:
            if reason:
                assert "0.80" in reason or "conflicts" in reason or "failed" in reason

    def test_round_trips_through_csv(self, run_result, config, tmp_path):
        from swift_address.io import read_output_csv, write_output_csv

        reloaded = read_output_csv(
            write_output_csv(run_result.frame, tmp_path / "out.csv")
        )
        column = config.output.column_name("hitl_flag", GROUP)
        assert set(reloaded[column]) <= {"True", "False"}


class TestNullGroups:
    def test_null_group_is_blank_not_auto_accept(self, run_result, config):
        """A group short-circuited before any model call was never judged."""
        frame = run_result.frame.set_index("RECORD_ID")
        row = frame.loc["CA0000000863"]        # every group empty on this record
        assert bool(row[config.output.column_name("hitl_flag", GROUP)]) is False
        assert row[config.output.column_name("hitl_state", GROUP)] == ""
        assert row[config.output.column_name("hitl_state_reason", GROUP)] == ""

    def test_unmapped_groups_are_blank_too(self, run_result, config):
        """Groups 1-14 and 16 hold no data in the sample."""
        frame = run_result.frame.set_index("RECORD_ID")
        row = frame.loc["CA0000001855"]
        assert row[config.output.column_name("hitl_state", "1")] == ""

    def test_null_json_block_is_not_evaluated(self, documents):
        hitl = documents[0]["groups"]["1"]["hitl"]
        assert hitl["required"] is False
        assert hitl["state"] == ""
        assert hitl["reason"] == ""
        assert hitl["forced_review"] is False
        assert hitl["contributing_reasons"] == []

    def test_null_skip_status_remains_authoritative(self, documents):
        assert documents[0]["groups"]["1"]["status"] == "null_skip"


class TestJsonBlock:
    def test_every_extracted_group_has_a_hitl_block(self, documents):
        for doc in documents:
            for group in doc["groups"].values():
                assert "hitl" in group

    def test_block_carries_the_full_schema(self, documents):
        group = next(
            g for doc in documents for g in doc["groups"].values()
            if g["status"] == "extracted"
        )
        assert set(group["hitl"]) == {
            "required", "state", "reason", "configured_threshold",
            "composite_weighted_score", "forced_review", "contributing_reasons",
            "manual_override",
        }

    def test_configured_threshold_is_recorded(self, documents, config):
        for doc in documents:
            for group in doc["groups"].values():
                if group["status"] == "extracted":
                    assert group["hitl"]["configured_threshold"] == (
                        config.scoring.hitl_threshold
                    )

    def test_contributing_reasons_are_retained(self, documents):
        for doc in documents:
            for group in doc["groups"].values():
                if group["status"] != "extracted":
                    continue
                hitl = group["hitl"]
                assert isinstance(hitl["contributing_reasons"], list)
                if hitl["state"] == HITL_AUTO_ACCEPT_CANDIDATE:
                    assert hitl["contributing_reasons"] == []
                else:
                    assert hitl["contributing_reasons"]

    def test_json_agrees_with_the_csv(self, documents, run_result, config):
        frame = run_result.frame.set_index("RECORD_ID")
        for doc in documents:
            for group_id, group in doc["groups"].items():
                if group["status"] != "extracted":
                    continue
                row = frame.loc[doc["record_id"]]
                assert group["hitl"]["required"] == bool(
                    row[config.output.column_name("hitl_flag", group_id)]
                )
                assert group["hitl"]["state"] == (
                    row[config.output.column_name("hitl_state", group_id)]
                )
                assert group["hitl"]["reason"] == (
                    row[config.output.column_name("hitl_state_reason", group_id)]
                )

    def test_composite_matches_the_scoring_block(self, documents):
        for doc in documents:
            for group in doc["groups"].values():
                if group["status"] != "extracted":
                    continue
                assert group["hitl"]["composite_weighted_score"] == pytest.approx(
                    group["scoring"]["composite_weighted_score"]
                )

    def test_ambiguous_group_retains_all_country_candidates(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path, tmp_path,
    ):
        """Forcing review must not collapse the candidate set."""
        from swift_address.reference_data import find_iso_provider

        escalating = config.model_copy(
            update={
                "reference_data": config.reference_data.model_copy(
                    update={"town_country_ambiguity_policy": "escalate"}
                )
            }
        )
        result = Phase1Pipeline(
            escalating, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        docs = list(read_detailed_jsonl(write_detailed_json(
            result.frame, tmp_path / "escalated.jsonl",
            config=escalating, group_config=group_config,
            decisions_by_address=result.decisions_by_address,
            iso_provider=find_iso_provider(reference_provider),
        )))
        ambiguous = [
            g for doc in docs for g in doc["groups"].values()
            if g["status"] == "extracted" and "," in g["prediction"]["country"]
        ]
        assert ambiguous, "the escalate policy should produce an ambiguous group"
        for group in ambiguous:
            assert group["hitl"]["state"] == "HITL_AMBIGUOUS_COUNTRY"
            assert len(group["prediction"]["country"].split(",")) > 1
            assert len(group["prediction"]["country_name"].split(",")) > 1


class TestRunMetrics:
    def test_state_counts_are_reported(self, run_result):
        states = run_result.metrics["hitl"]["state_counts"]
        assert states
        assert set(states) <= set(HITL_STATE_PRECEDENCE)
        assert sum(states.values()) == run_result.metrics["pass1"]["non_empty_instances"]

    def test_forced_review_instances_are_counted(self, run_result):
        assert "forced_review_instances" in run_result.metrics["hitl"]

    def test_configured_threshold_is_reported(self, run_result, config):
        assert run_result.metrics["hitl"]["threshold"] == config.scoring.hitl_threshold


class TestStateDistributionReport:
    def test_required_columns(self, run_result):
        frame = build_hitl_state_distribution(run_result.instances)
        assert list(frame.columns) == [
            "HITL_state", "observation_count", "percent_of_non_empty"
        ]

    def test_every_state_appears_even_at_zero(self, run_result):
        frame = build_hitl_state_distribution(run_result.instances)
        assert set(frame["HITL_state"]) >= set(HITL_STATE_PRECEDENCE)

    def test_rows_are_in_precedence_order(self, run_result):
        frame = build_hitl_state_distribution(run_result.instances)
        listed = [s for s in frame["HITL_state"] if s in HITL_STATE_PRECEDENCE]
        assert listed == list(HITL_STATE_PRECEDENCE)

    def test_null_skips_are_excluded_from_the_denominator(self, run_result):
        frame = build_hitl_state_distribution(run_result.instances)
        # 8 rows x 16 groups = 128 instances, of which 121 are empty.
        assert run_result.metrics["pass1"]["empty_instances_skipped"] == 121
        assert frame["observation_count"].sum() == 7
        assert frame["percent_of_non_empty"].sum() == pytest.approx(100.0)

    def test_blank_state_is_not_listed(self, run_result):
        frame = build_hitl_state_distribution(run_result.instances)
        assert "" not in set(frame["HITL_state"])

    def test_empty_input_still_lists_every_state(self):
        frame = build_hitl_state_distribution(pd.DataFrame())
        assert len(frame) == len(HITL_STATE_PRECEDENCE)
        assert frame["observation_count"].sum() == 0

    def test_written_as_a_report_artifact(self, run_result, config):
        report = write_reports(run_result.metrics, run_result.instances, config)
        path = report.paths["hitl_state_distribution"]
        assert path.exists()
        reloaded = pd.read_csv(path)
        assert "HITL_state" in reloaded.columns
        assert set(reloaded["HITL_state"]) >= set(HITL_STATE_PRECEDENCE)

    def test_path_comes_from_configuration(self, config):
        assert config.reporting.hitl_state_distribution_filename == (
            "hitl_state_distribution.csv"
        )
