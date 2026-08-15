"""The nested detailed-JSON output: shape, agreement with the CSV, and streaming."""

from __future__ import annotations

import json
import math

import pytest

from swift_address.io import read_input_csv
from swift_address.pipeline import Phase1Pipeline
from swift_address.serialization import (
    SCHEMA_VERSION,
    iter_record_documents,
    read_detailed_jsonl,
    write_detailed_json,
)


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
    return list(read_detailed_jsonl(path)), path


class TestRecordCoverage:
    def test_every_record_id_appears_exactly_once(self, documents, run_result):
        docs, _ = documents
        ids = [doc["record_id"] for doc in docs]
        assert len(ids) == len(run_result.frame)
        assert ids == run_result.frame["RECORD_ID"].tolist()
        assert len(set(ids)) == len(ids)

    def test_record_id_is_preserved_exactly(self, documents):
        docs, _ = documents
        assert docs[0]["record_id"] == "CA0000000318"      # not coerced to a number

    def test_schema_version_is_stamped(self, documents):
        docs, _ = documents
        assert all(doc["schema_version"] == SCHEMA_VERSION for doc in docs)

    def test_every_enabled_group_is_represented(self, documents, group_config):
        docs, _ = documents
        expected = {group.group_id for group in group_config.enabled_groups}
        for doc in docs:
            assert set(doc["groups"]) == expected

    def test_empty_groups_are_marked_null_skip(self, documents):
        docs, _ = documents
        group_one = docs[0]["groups"]["1"]
        assert group_one["status"] == "null_skip"
        assert group_one["address"]["combined_address_cleaned"] == ""

    def test_empty_groups_can_be_omitted(
        self, run_result, config, group_config, tmp_path
    ):
        path = write_detailed_json(
            run_result.frame, tmp_path / "lean.jsonl",
            config=config, group_config=group_config,
            decisions_by_address=run_result.decisions_by_address,
            include_empty_groups=False,
        )
        docs = list(read_detailed_jsonl(path))
        assert set(docs[0]["groups"]) == {"15"}

    def test_populated_group_carries_every_block(self, documents):
        docs, _ = documents
        group = docs[5]["groups"]["15"]
        assert set(group) >= {
            "status", "source_fields", "address", "prediction", "text_evidence",
            "ground_truth_validation", "scoring", "cross_entropy", "rationale",
            "retraction",
        }
        assert group["source_fields"] == [
            "PRI_PAY_BNF_ADDR_LINE_1",
            "PRI_PAY_BNF_ADDR_LINE_2",
            "PRI_PAY_BNF_ADDR_LINE_3",
        ]

    def test_run_wide_metadata_is_not_duplicated_per_group(self, documents):
        """Model and reference provenance belong in run_metrics, not every group."""
        docs, _ = documents
        group = docs[5]["groups"]["15"]
        assert "model" not in group
        assert "prompt_version" not in group
        assert "reference_data_version" not in group


class TestAgreementWithCsv:
    def test_scalar_values_match_the_csv(self, documents, run_result, config):
        docs, _ = documents
        frame = run_result.frame.set_index("RECORD_ID")

        for doc in docs:
            group = doc["groups"].get("15")
            if group is None or group["status"] == "null_skip":
                continue
            row = frame.loc[doc["record_id"]]
            column = lambda key: config.output.column_name(key, "15")  # noqa: E731

            assert group["prediction"]["town"] == row[column("predicted_town")]
            assert group["prediction"]["country"] == row[column("predicted_country")]
            assert group["prediction"]["country_name"] == (
                row[column("predicted_country_name")]
            )
            assert group["text_evidence"]["predicted_town_exists"] == bool(
                row[column("predicted_town_exists")]
            )
            assert group["scoring"]["composite_weighted_score"] == pytest.approx(
                float(row[column("composite_weighted_score")])
            )

    def test_ground_truth_labels_match_the_csv(self, documents, run_result, config):
        docs, _ = documents
        frame = run_result.frame.set_index("RECORD_ID")

        for doc in docs:
            group = doc["groups"].get("15")
            if group is None or group["status"] == "null_skip":
                continue
            row = frame.loc[doc["record_id"]]
            for json_key, field_key in (
                ("town_exists_ok", "town_exists_ok"),
                ("country_exists_ok", "country_exists_ok"),
            ):
                csv_value = row[config.output.column_name(field_key, "15")]
                json_value = group["ground_truth_validation"][json_key]
                if json_value is None:
                    assert csv_value is None or csv_value is not csv_value or (
                        str(csv_value) in {"<NA>", "nan", ""}
                    )
                else:
                    assert bool(csv_value) == json_value

    def test_cross_entropy_matches_the_csv(self, documents, run_result, config):
        docs, _ = documents
        frame = run_result.frame.set_index("RECORD_ID")
        column = config.output.column_name("cross_entropy", "15")

        for doc in docs:
            group = doc["groups"].get("15")
            if group is None or group["status"] == "null_skip":
                continue
            csv_value = frame.loc[doc["record_id"], column]
            json_value = group["cross_entropy"]["group_cross_entropy"]
            if json_value is None:
                assert math.isnan(float(csv_value))
            else:
                assert float(csv_value) == pytest.approx(json_value)

    def test_retraction_matches_the_csv(self, documents, run_result, config):
        """The two representations come from the same pure function."""
        docs, _ = documents
        frame = run_result.frame.set_index("RECORD_ID")

        for doc in docs:
            group = doc["groups"].get("15")
            if group is None or group["status"] == "null_skip":
                continue
            row = frame.loc[doc["record_id"]]
            assert group["retraction"]["combined_address_retracted"] == (
                row[config.output.column_name("combined_address_retracted", "15")]
            )
            assert group["retraction"]["comment"] == (
                row[config.output.column_name(
                    "combined_address_retracted_comments", "15"
                )]
            )

    def test_before_after_source_columns_are_correct(self, documents, run_result):
        docs, _ = documents
        frame = run_result.frame.set_index("RECORD_ID")
        retraction = docs[5]["groups"]["15"]["retraction"]

        before = retraction["actual_column_before_retraction"]
        after = retraction["actual_column_after_retraction"]
        assert set(before) == set(after)

        # "before" reproduces the untouched source columns.
        row = frame.loc["CA0000001855"]
        for field_name, value in before.items():
            assert value == row[field_name]

        assert before["PRI_PAY_BNF_ADDR_LINE_2"] == "CITIGROUP CENTRE AUCKLAND AUCKLAND"
        assert after["PRI_PAY_BNF_ADDR_LINE_2"] == "CITIGROUP CENTRE"
        assert retraction["retracted_entities"] == ["town", "country"]

    def test_rebuilt_combined_matches_the_after_columns(self, documents):
        from swift_address.cleaning import clean_address
        from swift_address.grouping import build_combined_address

        docs, _ = documents
        for doc in docs:
            group = doc["groups"].get("15")
            if group is None or group["status"] == "null_skip":
                continue
            retraction = group["retraction"]
            after = retraction["actual_column_after_retraction"]
            rebuilt = clean_address(
                build_combined_address(
                    [after[name] for name in group["source_fields"]]
                )
            )
            assert retraction["combined_address_retracted"] == rebuilt


class TestJsonValidity:
    def test_unavailable_numerics_serialize_as_null_never_nan(self, documents):
        docs, raw_path = documents
        text = raw_path.read_text(encoding="utf-8")
        assert "NaN" not in text
        assert "Infinity" not in text

        ungrounded = [
            group
            for doc in docs
            for group in doc["groups"].values()
            if group["cross_entropy"]["group_cross_entropy"] is None
        ]
        assert ungrounded, "the sample should contain at least one ungrounded group"

    def test_nullable_booleans_serialize_as_json_null(self, documents):
        docs, _ = documents
        lima = next(doc for doc in docs if doc["record_id"] == "CA0000000694")
        validation = lima["groups"]["15"]["ground_truth_validation"]
        assert validation["town_exists_ok"] is True
        assert validation["country_exists_ok"] is None      # not False

    def test_every_line_is_independently_parseable(self, documents):
        _, path = documents
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)

    def test_file_is_utf8(self, documents):
        _, path = documents
        path.read_text(encoding="utf-8")

    def test_output_is_deterministic_across_runs(
        self, run_result, config, group_config, tmp_path
    ):
        first = write_detailed_json(
            run_result.frame, tmp_path / "a.jsonl", config=config,
            group_config=group_config,
            decisions_by_address=run_result.decisions_by_address,
        )
        second = write_detailed_json(
            run_result.frame, tmp_path / "b.jsonl", config=config,
            group_config=group_config,
            decisions_by_address=run_result.decisions_by_address,
        )
        assert first.read_bytes() == second.read_bytes()


class TestWriterMechanics:
    def test_missing_output_directory_is_created(
        self, run_result, config, group_config, tmp_path
    ):
        target = tmp_path / "deep" / "nested" / "detail.jsonl"
        assert not target.parent.exists()
        write_detailed_json(
            run_result.frame, target, config=config, group_config=group_config,
            decisions_by_address=run_result.decisions_by_address,
        )
        assert target.exists()

    def test_documents_are_generated_lazily(
        self, run_result, config, group_config
    ):
        """The writer streams: nothing is materialized up front."""
        import types

        stream = iter_record_documents(
            run_result.frame, config=config, group_config=group_config,
            decisions_by_address=run_result.decisions_by_address,
        )
        assert isinstance(stream, types.GeneratorType)
        first = next(stream)
        assert first["record_id"] == "CA0000000318"

    def test_json_array_mode_is_valid_json(
        self, run_result, config, group_config, tmp_path
    ):
        path = write_detailed_json(
            run_result.frame, tmp_path / "detail.json", config=config,
            group_config=group_config,
            decisions_by_address=run_result.decisions_by_address,
            output_format="json",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert len(payload) == len(run_result.frame)

    def test_jsonl_has_one_line_per_record(self, documents, run_result):
        _, path = documents
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == len(run_result.frame)

    def test_written_through_run_phase1(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, prompt_contract, tmp_path,
    ):
        from swift_address.pipeline import run_phase1

        target = tmp_path / "pipeline_detail.jsonl"
        patched = config.model_copy(
            update={
                "processing": config.processing.model_copy(
                    update={"detailed_json_path": str(target)}
                )
            }
        )
        run_phase1(
            "data/sample_input.csv", patched, group_config,
            client=mock_client, reference_provider=reference_provider,
            town_country_provider=town_country_provider,
            prompt=prompt_contract, mode="dry_run",
        )
        assert target.exists()
        assert len(list(read_detailed_jsonl(target))) == 8
