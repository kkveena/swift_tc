"""Output schema: column arithmetic, preservation, dtypes, and round-tripping.

The column counts here are always *calculated* from the loaded configuration.
The literals 176 and 226 appear only as assertions about the supplied sample,
never as inputs to the code.
"""

from __future__ import annotations

import pandas as pd
import pytest

from swift_address.io import (
    ERROR_COLUMNS,
    ProcessingError,
    read_output_csv,
    write_errors_csv,
    write_output_csv,
)
from swift_address.pipeline import Phase1Pipeline
from swift_address.settings import OUTPUT_FIELD_KEYS, load_config


@pytest.fixture
def run_result(config, group_config, reference_provider, mock_client, sample_input_path):
    from swift_address.io import read_input_csv

    pipeline = Phase1Pipeline(
        config,
        group_config,
        client=mock_client,
        reference_provider=reference_provider,
        mode="dry_run",
    )
    return pipeline.run(read_input_csv(sample_input_path))


class TestColumnArithmetic:
    def test_twenty_fields_per_group(self, config):
        """17 prior fields + the explicit HITL flag, state and reason."""
        assert len(OUTPUT_FIELD_KEYS) == 20
        assert config.fields_per_group == 20

    def test_new_fields_are_appended_not_interleaved(self, config):
        """Pre-existing column positions must not shift for flat-file consumers."""
        assert list(OUTPUT_FIELD_KEYS[-8:]) == [
            "town_exists_ok",
            "country_exists_ok",
            "cross_entropy",
            "combined_address_retracted",
            "combined_address_retracted_comments",
            "hitl_flag",
            "hitl_state",
            "hitl_state_reason",
        ]
        names = config.group_column_names("15")
        assert names[-8:] == (
            "town_exists_ok_group_15",
            "country_exists_ok_group_15",
            "cross_entropy_group_15",
            "combined_address_retracted_group_15",
            "combined_address_retracted_group_comments_15",
            "HITL_flag_group_15",
            "HITL_state_group_15",
            "HITL_state_reason_group_15",
        )

    def test_hitl_fields_use_the_canonical_acronym(self, config):
        """HITL = Human-in-the-Loop. No HILT_* transpositions."""
        names = config.group_column_names("15")
        assert "HITL_flag_group_15" in names
        assert not any("HILT" in name for name in names)

    def test_country_name_sits_directly_after_country_code(self, config):
        keys = list(OUTPUT_FIELD_KEYS)
        assert keys.index("predicted_country_name") == keys.index("predicted_country") + 1

        names = list(config.group_column_names("15"))
        assert (
            names.index("predicted_country_name_group_15")
            == names.index("predicted_country_group_15") + 1
        )

    def test_sixteen_groups_append_exactly_320_columns(
        self, config, group_config, run_result, sample_input_path
    ):
        from swift_address.io import read_input_csv

        input_columns = len(read_input_csv(sample_input_path).columns)
        appended = len(run_result.frame.columns) - input_columns
        expected = len(group_config.enabled_groups) * config.fields_per_group

        assert appended == expected
        assert len(group_config.enabled_groups) == 16
        assert appended == 320       # 16 x 20

    def test_fifty_column_input_becomes_370_columns(self, run_result):
        assert run_result.metrics["shape"]["input_columns"] == 50
        assert len(run_result.frame.columns) == 370      # 50 + 16 x 20

    def test_count_is_derived_not_hard_coded(self, config, group_config, run_result):
        shape = run_result.metrics["shape"]
        assert shape["output_columns"] == (
            shape["input_columns"]
            + shape["groups_enabled"] * shape["fields_per_group"]
        )

    def test_a_different_group_count_changes_the_arithmetic(
        self, config, reference_provider, mock_client, tmp_path
    ):
        """Three groups append 3 x 20 columns. Nothing in the code assumes 16."""
        from swift_address.grouping import load_group_config

        group_path = tmp_path / "groups.csv"
        group_path.write_text(
            "group_id,address_line_1,address_line_2,enabled\n"
            "1,A1,A2,True\n2,B1,B2,True\n3,C1,C2,True\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"], "A1": ["BOSTON MA US"], "A2": [""],
             "B1": [""], "B2": [""], "C1": ["0"], "C2": [""]}
        )
        pipeline = Phase1Pipeline(
            config,
            load_group_config(group_path),
            client=mock_client,
            reference_provider=reference_provider,
            mode="dry_run",
        )
        result = pipeline.run(frame)
        assert len(result.frame.columns) == 7 + 3 * config.fields_per_group == 67

    def test_no_accidental_extra_columns(self, config, group_config, run_result, sample_input_path):
        from swift_address.io import read_input_csv

        expected = list(read_input_csv(sample_input_path).columns)
        for group in group_config.enabled_groups:
            expected.extend(config.group_column_names(group.group_id))

        assert list(run_result.frame.columns) == expected

    def test_only_one_naming_set_is_emitted(self, config, run_result):
        """Canonical names present; names unique to the legacy set must be absent.

        The two template sets overlap (e.g. `predicted_town_group_15` is spelled
        the same either way), so only the legacy-*only* names prove the sets
        were not emitted together.
        """
        columns = set(run_result.frame.columns)
        canonical = {t.format(id="15") for t in config.output.templates.values()}
        legacy = {t.format(id="15") for t in config.output.legacy_templates.values()}
        legacy_only = legacy - canonical

        assert canonical <= columns
        assert legacy_only, "legacy templates should differ from canonical ones"
        assert not (legacy_only & columns)
        assert "comined_address_group_15" in legacy_only        # the documented typos
        assert "predicted_countrty_group_15" in legacy_only
        assert "rational_town_group_15" in legacy_only


class TestLegacyNamingStyle:
    def test_legacy_style_emits_only_legacy_names(
        self, repo_root, tmp_path, group_config, reference_provider, mock_client
    ):
        config = load_config(
            repo_root / "config" / "config.yaml",
            base_dir=repo_root,
            overrides={
                "output": {"naming_style": "legacy"},
                "processing": {
                    "cache_path": str(tmp_path / "cache.jsonl"),
                    "output_path": str(tmp_path / "out.csv"),
                },
            },
        )
        frame = pd.DataFrame(
            {"RECORD_ID": ["R1"]}
            | {field: [""] for field in group_config.all_source_fields}
        )
        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider, mode="dry_run",
        ).run(frame)

        columns = set(result.frame.columns)
        assert "comined_address_group_15" in columns
        assert "combined_address_group_15" not in columns
        assert "predicted_countrty_name_group_15" in columns
        assert len(result.frame.columns) == (
            1 + len(group_config.all_source_fields) + 16 * config.fields_per_group
        )


class TestInputPreservation:
    def test_record_id_values_are_preserved_exactly(self, run_result, sample_input_path):
        from swift_address.io import read_input_csv

        original = read_input_csv(sample_input_path)["RECORD_ID"].tolist()
        assert run_result.frame["RECORD_ID"].tolist() == original
        assert original[0] == "CA0000000318"       # not coerced to a number

    def test_all_input_columns_are_preserved_in_order(self, run_result, sample_input_path):
        from swift_address.io import read_input_csv

        original = list(read_input_csv(sample_input_path).columns)
        assert list(run_result.frame.columns)[: len(original)] == original

    def test_source_values_are_untouched(self, run_result, sample_input_path):
        from swift_address.io import read_input_csv

        original = read_input_csv(sample_input_path)
        pd.testing.assert_frame_equal(
            run_result.frame[original.columns], original, check_dtype=False
        )

    def test_row_count_is_unchanged(self, run_result, sample_input_path):
        from swift_address.io import read_input_csv

        assert len(run_result.frame) == len(read_input_csv(sample_input_path)) == 8

    def test_unknown_passthrough_column_survives(self, run_result):
        assert "OTHER" in run_result.frame.columns

    def test_postal_codes_keep_their_leading_zeros(self, run_result):
        assert "02111" in run_result.frame.loc[0, "combined_address_group_15"]


class TestOutputDtypes:
    def test_probability_and_score_columns_are_numeric(self, run_result):
        for column in (
            "predicted_town_probability_group_15",
            "predicted_country_probability_group_15",
            "composite_weighted_score_group_15",
        ):
            assert pd.api.types.is_float_dtype(run_result.frame[column])

    def test_exists_columns_are_boolean(self, run_result):
        for column in (
            "predicted_town_exists_group_15",
            "predicted_country_exists_group_15",
        ):
            assert pd.api.types.is_bool_dtype(run_result.frame[column])

    def test_probabilities_stay_within_the_unit_interval(self, run_result, config, group_config):
        for group in group_config.enabled_groups:
            for key in ("predicted_town_probability", "predicted_country_probability"):
                series = run_result.frame[config.output.column_name(key, group.group_id)]
                assert series.between(0.0, 1.0).all()

    def test_country_values_are_iso_codes_or_the_sentinel(self, run_result, config, group_config):
        for group in group_config.enabled_groups:
            column = config.output.column_name("predicted_country", group.group_id)
            for value in run_result.frame[column]:
                if value == "NO_COUNTRY":
                    continue
                codes = value.split(",")
                assert all(len(code) == 2 and code.isupper() and code.isalpha() for code in codes)
                assert codes == sorted(set(codes))       # deterministic and deduplicated


class TestCsvRoundTrip:
    def test_reload_preserves_rows_and_required_fields(self, run_result, tmp_path):
        path = write_output_csv(run_result.frame, tmp_path / "phase1_output.csv")
        reloaded = read_output_csv(path)

        assert len(reloaded) == len(run_result.frame)
        assert list(reloaded.columns) == list(run_result.frame.columns)
        assert reloaded["RECORD_ID"].tolist() == run_result.frame["RECORD_ID"].tolist()
        assert reloaded.loc[0, "predicted_town_group_15"] == "BOSTON"
        assert reloaded.loc[0, "predicted_town_exists_group_15"] == "True"
        assert float(reloaded.loc[3, "composite_weighted_score_group_15"]) == pytest.approx(
            run_result.frame.loc[3, "composite_weighted_score_group_15"]
        )

    def test_empty_group_round_trips_as_empty_strings(self, run_result, tmp_path):
        path = write_output_csv(run_result.frame, tmp_path / "out.csv")
        reloaded = read_output_csv(path)
        assert reloaded.loc[2, "combined_address_group_15"] == ""
        assert reloaded.loc[2, "predicted_town_group_15"] == "NO_TOWN"
        assert reloaded.loc[2, "rationale_town_group_15"] == ""


class TestErrorSidecar:
    def test_sidecar_is_written_even_with_no_errors(self, tmp_path):
        path = write_errors_csv([], tmp_path / "processing_errors.csv")
        frame = pd.read_csv(path)
        assert list(frame.columns) == list(ERROR_COLUMNS)
        assert len(frame) == 0

    def test_error_row_carries_hash_not_raw_address(self, tmp_path):
        error = ProcessingError(
            address_hash="abc123",
            occurrences=4,
            group_ids=("3", "15"),
            record_ids=("R1", "R2"),
            error_type="TransientExtractionError",
            error_message="429 RESOURCE_EXHAUSTED\nquota",
            model="gemini-3.5-flash",
            prompt_version="v2-composite-weighted",
            attempts=6,
        )
        frame = pd.read_csv(write_errors_csv([error], tmp_path / "errors.csv"))
        row = frame.iloc[0]

        assert row["address_hash"] == "abc123"
        assert row["occurrences"] == 4
        assert row["group_ids"] == "3|15"
        assert "\n" not in row["error_message"]
        assert row["model"] == "gemini-3.5-flash"
        assert "address" not in [c for c in frame.columns if c != "address_hash"]

    def test_record_id_list_is_capped(self, tmp_path):
        error = ProcessingError(
            address_hash="h", occurrences=100,
            group_ids=("1",), record_ids=tuple(f"R{i}" for i in range(100)),
            error_type="E", error_message="m", model="m", prompt_version="v",
        )
        row = error.to_row()
        assert row["record_ids"].endswith("(+75 more)")
        assert row["occurrences"] == 100
