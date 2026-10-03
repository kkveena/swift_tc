"""EXP-02 retraction policy: line-1 protection and strict probability gates.

The policy decides *eligibility* only. Nothing here touches extraction,
verification, scoring or HITL, and the baseline policy (no protected field, no
gate) must reproduce EXP-01 byte for byte — that is tested explicitly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.swft_tc.src.retraction import (
    BASELINE_POLICY,
    SKIP_NOT_VERIFIED,
    SKIP_PROBABILITY,
    SKIP_PROTECTED_ONLY,
    RetractionPolicy,
    retract_group,
)

FIELDS = ("address_line_1", "address_line_2", "address_line_3", "address_line_4")
EXP02 = RetractionPolicy(
    name="line1_protected_confidence_gated",
    protected_source_positions=(1,),
    town_probability_threshold=0.80,
    country_probability_threshold=0.80,
)


def run(values, *, town, country, town_exists=True, country_exists=True,
        town_p=0.99, country_p=0.99, policy=EXP02, iso_provider=None, fields=FIELDS):
    full = {name: values.get(name, "") for name in fields}
    return retract_group(
        full, fields, town=town, country_value=country,
        town_exists=town_exists, country_exists=country_exists,
        iso_provider=iso_provider, town_probability=town_p, country_probability=country_p,
        policy=policy,
    )


class TestPolicyObject:
    def test_default_is_the_baseline(self):
        assert BASELINE_POLICY.is_baseline
        assert BASELINE_POLICY.protected_fields(FIELDS) == ()

    def test_protected_fields_are_positional_not_by_name(self):
        policy = RetractionPolicy(protected_source_positions=(1,))
        assert policy.protected_fields(("L1", "L2")) == ("L1",)
        assert policy.protected_fields(("x", "y", "z")) == ("x",)

    def test_out_of_range_position_is_ignored(self):
        assert RetractionPolicy(protected_source_positions=(9,)).protected_fields(FIELDS) == ()

    @pytest.mark.parametrize("probability,expected", [
        (0.80, False), (0.8000, False), (0.8001, True), (0.81, True), (1.0, True),
        (0.79, False), (0.0, False), (None, False),
    ])
    def test_gate_is_strictly_greater_than(self, probability, expected):
        assert RetractionPolicy.passes(probability, 0.80) is expected

    def test_no_threshold_means_no_gate(self):
        assert RetractionPolicy.passes(0.0, None) is True
        assert RetractionPolicy.passes(None, None) is True

    def test_from_config(self, model_root):
        from models.swft_tc.src.settings import load_config

        config = load_config(
            model_root / "config" / "config_exp02_line1_protected_confidence_gated.yaml",
            base_dir=model_root,
        )
        policy = RetractionPolicy.from_config(config.retraction)
        assert policy == EXP02
        baseline = load_config(
            model_root / "config" / "config_exp01_rightmost_town_allmatch_country.yaml",
            base_dir=model_root,
        )
        assert RetractionPolicy.from_config(baseline.retraction).is_baseline


class TestLineOneProtection:
    def test_line_one_is_never_modified(self, iso_provider):
        result = run(
            {"address_line_1": "CITIBANK LONDON GB", "address_line_2": "33 CANADA SQUARE",
             "address_line_3": "LONDON E14 5LB GB"},
            town="LONDON", country="GB", iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "CITIBANK LONDON GB"
        assert result.protected_source_fields == ("address_line_1",)

    def test_town_in_line_one_and_later_removes_only_the_later(self, iso_provider):
        result = run(
            {"address_line_1": "CITIBANK LONDON", "address_line_2": "33 CANADA SQUARE",
             "address_line_3": "LONDON E14 5LB"},
            town="LONDON", country="GB", country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "CITIBANK LONDON"
        assert result.after["address_line_3"] == "E14 5LB"
        assert result.town_retraction_eligible is True
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_town_only_in_line_one_removes_nothing(self, iso_provider):
        result = run(
            {"address_line_1": "TOWERBANK INTL INC- PANAMA CITY HO",
             "address_line_2": "CALLE 50 Y CALLE ELVIRA MENDEZ",
             "address_line_3": "TOWER FINANCIAL CENTER"},
            town="PANAMA CITY", country="PA", country_exists=False, iso_provider=iso_provider,
        )
        assert result.after == result.before
        assert result.town_retraction_eligible is False
        assert result.town_skip_reason == SKIP_PROTECTED_ONLY
        assert result.town_occurrences_found == 1
        assert "protected" in result.comment and "not retracted" in result.comment

    def test_country_in_line_one_and_later_removes_only_the_later(self, iso_provider):
        result = run(
            {"address_line_1": "NATIONAL BANK OF CANADA",
             "address_line_2": "600 RUE DE LA GAUCHETIERE OUES SUITE 610",
             "address_line_4": "MONTREAL QC H3B 4L3 CANADA"},
            town="MONTREAL", country="CA", iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "NATIONAL BANK OF CANADA"
        assert result.after["address_line_4"] == "QC H3B 4L3"
        assert result.country_retraction_eligible is True
        assert result.retracted_entities == ("town", "country")

    def test_country_only_in_line_one_removes_nothing(self, iso_provider):
        result = run(
            {"address_line_1": "BANCO LAFISE PANAMA SA", "address_line_2": "CALLE 50",
             "address_line_3": "EDIFICIO LAFISE"},
            town="PANAMA", country="PA", town_exists=False, iso_provider=iso_provider,
        )
        assert result.after == result.before
        assert result.country_retraction_eligible is False
        assert result.country_skip_reason == SKIP_PROTECTED_ONLY

    def test_country_in_line_one_plus_several_later_occurrences(self, iso_provider):
        result = run(
            {"address_line_1": "CITIBANK CANADA", "address_line_2": "CANADA SQUARE",
             "address_line_3": "TORONTO ON CANADA", "address_line_4": "CA"},
            town="TORONTO", country="CA", iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "CITIBANK CANADA"
        assert result.after["address_line_2"] == "SQUARE"
        assert result.after["address_line_3"] == "ON"
        assert result.after["address_line_4"] == ""

    def test_qatar_in_line_one_is_protected_and_later_qa_is_removed(self, iso_provider):
        result = run(
            {"address_line_1": "CITIBANK QATAR DOHA", "address_line_2": "WEST BAY 9TH FLOOR",
             "address_line_3": "QFC TOWER 1 DOHA DOHA QA"},
            town="DOHA", country="QA", iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "CITIBANK QATAR DOHA"
        assert result.after["address_line_3"] == "QFC TOWER 1 DOHA"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (3, 1)

    def test_fubon_bank_hong_kong(self, iso_provider):
        result = run(
            {"address_line_1": "FUBON BANK HONG KONG LTD", "address_line_2": "38 DES VOEUX ROAD",
             "address_line_3": "FUBON BANK BUILDING HONG KONG HK"},
            town="HONG KONG", country="HK", iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "FUBON BANK HONG KONG LTD"
        assert result.after["address_line_3"] == "FUBON BANK BUILDING"

    def test_ambiguous_code_in_a_protected_tail_is_not_removed(self, iso_provider):
        """When the only populated line is protected, the trailing-window code stays."""
        result = run(
            {"address_line_1": "PO BOX 5 MA"},
            town="X", country="MA", town_exists=False, iso_provider=iso_provider,
        )
        assert result.after["address_line_1"] == "PO BOX 5 MA"
        assert result.country_skip_reason == SKIP_PROTECTED_ONLY


class TestProbabilityGate:
    BASE = {"address_line_1": "HEAD OFFICE", "address_line_2": "1 MAIN STREET",
            "address_line_3": "AUCKLAND 1140 NZ"}

    def test_town_exactly_point_eight_does_not_retract(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=0.80, country_exists=False,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "AUCKLAND 1140 NZ"
        assert result.town_probability_gate_passed is False
        assert result.town_skip_reason == SKIP_PROBABILITY
        assert "did not exceed the retraction threshold 0.80" in result.comment

    def test_town_just_above_point_eight_may_retract(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=0.8001, country_exists=False,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "1140 NZ"
        assert result.town_probability_gate_passed is True

    def test_country_exactly_point_eight_does_not_retract(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", country_p=0.80, town_exists=False,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "AUCKLAND 1140 NZ"
        assert result.country_skip_reason == SKIP_PROBABILITY

    def test_country_just_above_point_eight_may_retract(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", country_p=0.8001, town_exists=False,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "AUCKLAND 1140"

    def test_town_passes_country_fails(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=0.95, country_p=0.75,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "1140 NZ"
        assert result.retracted_entities == ("town",)
        assert result.country_skip_reason == SKIP_PROBABILITY
        # The comment must not blame verification for a probability decision.
        assert "Country was not explicitly verified" not in result.comment

    def test_country_passes_town_fails(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=0.80, country_p=0.99,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "AUCKLAND 1140"
        assert result.retracted_entities == ("country",)
        assert result.town_skip_reason == SKIP_PROBABILITY

    def test_neither_passes(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=0.5, country_p=0.5,
                     iso_provider=iso_provider)
        assert result.after == result.before
        assert result.retracted_entities == ()

    def test_both_pass(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=0.9, country_p=0.9,
                     iso_provider=iso_provider)
        assert result.after["address_line_3"] == "1140"
        assert result.retracted_entities == ("town", "country")
        assert result.comment.startswith("[line1_protected_confidence_gated]")

    def test_missing_probability_cannot_pass_a_gate(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", town_p=None, country_p=None,
                     iso_provider=iso_provider)
        assert result.after == result.before

    def test_gate_values_are_recorded(self, iso_provider):
        result = run(self.BASE, town="AUCKLAND", country="NZ", iso_provider=iso_provider)
        assert (result.town_probability_gate, result.country_probability_gate) == (0.80, 0.80)
        assert result.policy_name == "line1_protected_confidence_gated"


class TestEligibleFieldRules:
    def test_repeated_town_across_eligible_fields_removes_one_rightmost(self, iso_provider):
        result = run(
            {"address_line_1": "SOME BANK", "address_line_2": "LIMA",
             "address_line_3": "METRO MUNIC OF LIMA 15001"},
            town="LIMA", country="PE", country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["address_line_2"] == "LIMA"
        assert result.after["address_line_3"] == "METRO MUNIC OF 15001"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_country_repeated_across_eligible_fields_removes_all(self, iso_provider):
        result = run(
            {"address_line_1": "SOME BANK", "address_line_2": "PO BOX 1 NZ",
             "address_line_3": "1140 NZ"},
            town="AUCKLAND", country="NZ", town_exists=False, iso_provider=iso_provider,
        )
        assert result.after["address_line_2"] == "PO BOX 1"
        assert result.after["address_line_3"] == "1140"

    @pytest.mark.parametrize("line,town", [
        ("AERONAUTICA MILITARE", "RONA"),
        ("23 CUSTOMS STREET EAST", "US"),
        ("THE BOSTONIAN CLUB", "BOSTON"),
    ])
    def test_substring_traps_remain_protected(self, line, town, iso_provider):
        result = run({"address_line_1": "HQ", "address_line_2": line}, town=town, country="NO_COUNTRY",
                     country_exists=False, iso_provider=iso_provider)
        assert result.after["address_line_2"] == line

    def test_unverified_entities_are_reported_as_such(self, iso_provider):
        result = run(self_base := {"address_line_1": "HQ", "address_line_2": "AUCKLAND NZ"},
                     town="AUCKLAND", country="NZ", town_exists=False, country_exists=False,
                     iso_provider=iso_provider)
        assert result.after == result.before
        assert result.town_skip_reason == SKIP_NOT_VERIFIED
        assert result.country_skip_reason == SKIP_NOT_VERIFIED


class TestBaselineIsUnchanged:
    """EXP-01 / default configuration: byte-for-byte the existing behaviour."""

    CASES = [
        ({"address_line_1": "23 CUSTOMS STREET EAST LEVEL 11",
          "address_line_2": "CITIGROUP CENTRE AUCKLAND AUCKLAND", "address_line_3": "1140 NZ"},
         "AUCKLAND", "NZ", True, True),
        ({"address_line_1": "CITIBANK LONDON", "address_line_2": "33 CANADA SQUARE",
          "address_line_3": "LONDON E14 5LB GB"}, "LONDON", "GB", True, True),
        ({"address_line_1": "TOWERBANK INTL INC- PANAMA CITY HO"}, "PANAMA CITY", "PA", True, False),
        ({"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL QC H3B 4L3 CANADA"},
         "MONTREAL", "CA", True, True),
        ({"address_line_1": "HEAD OFFICE BUILDING"}, "TAIPEI", "TW", False, False),
    ]

    @pytest.mark.parametrize("values,town,country,te,ce", CASES)
    def test_baseline_policy_equals_the_pre_policy_call(self, values, town, country, te, ce, iso_provider):
        full = {name: values.get(name, "") for name in FIELDS}
        legacy = retract_group(full, FIELDS, town=town, country_value=country,
                               town_exists=te, country_exists=ce, iso_provider=iso_provider)
        explicit = retract_group(full, FIELDS, town=town, country_value=country,
                                 town_exists=te, country_exists=ce, iso_provider=iso_provider,
                                 town_probability=0.0, country_probability=0.0, policy=BASELINE_POLICY)
        assert legacy.after == explicit.after
        assert legacy.combined_address_retracted == explicit.combined_address_retracted
        assert legacy.comment == explicit.comment
        assert legacy.retracted_entities == explicit.retracted_entities

    def test_baseline_protects_nothing_and_modifies_line_one(self, iso_provider):
        """Documents the EXP-01 behaviour EXP-02 exists to challenge."""
        result = run({"address_line_1": "TOWERBANK INTL INC- PANAMA CITY HO"},
                     town="PANAMA CITY", country="PA", country_exists=False,
                     policy=BASELINE_POLICY, iso_provider=iso_provider)
        # The hyphen goes too: a removal absorbs one adjacent separator run.
        assert result.after["address_line_1"] == "TOWERBANK INTL INC HO"
        assert result.comment == (
            "Retracted Town=PANAMA CITY. Country was not explicitly verified in the "
            "input, so it was retained only as a prediction."
        )

    def test_baseline_gate_values_are_null(self, iso_provider):
        result = run({"address_line_1": "HQ", "address_line_2": "AUCKLAND NZ"},
                     town="AUCKLAND", country="NZ", policy=BASELINE_POLICY, iso_provider=iso_provider)
        assert result.town_probability_gate is None and result.country_probability_gate is None
        assert result.town_probability_gate_passed and result.country_probability_gate_passed


class TestExperimentInputFixtures:
    """The real EXP-01 input rows the experiment is built around."""

    @pytest.fixture
    def rows(self, model_root):
        frame = pd.read_csv(
            model_root / "data" / "exp01_rightmost_town_allmatch_country_addresses.csv",
            dtype=str, keep_default_na=False, encoding="utf-8-sig",
        ).set_index("RECORD_ID")
        return frame

    @pytest.mark.parametrize("record_id,town,country,line1_fragment", [
        ("1208", "PANAMA CITY", "PA", "TOWERBANK INTL INC- PANAMA CITY HO"),
        ("1212", "PANAMA", "PA", "BANCO LAFISE PANAMA SA"),
        ("2152", "MONTREAL", "CA", "NATIONAL BANK OF CANADA"),
        ("1578", "DOHA", "QA", "CITIBANK QATAR DOHA"),
        ("1450", "HONG KONG", "HK", "FUBON BANK HONG KONG LTD"),
    ])
    def test_line_one_survives_under_exp02(self, rows, record_id, town, country, line1_fragment, iso_provider):
        if record_id not in rows.index:
            pytest.skip(f"{record_id} not in the experiment input")
        row = rows.loc[record_id]
        assert line1_fragment in row["address_line_1"]
        result = run(row[list(FIELDS)].to_dict(), town=town, country=country, iso_provider=iso_provider)
        assert result.after["address_line_1"] == row["address_line_1"]


class TestPipelineIntegration:
    """The policy reaches the CSV and the detailed JSON through the real pipeline."""

    def test_csv_and_json_agree_under_exp02(
        self, model_root, tmp_path, reference_provider, town_country_provider, mock_client,
    ):
        from models.swft_tc.src.grouping import load_group_config
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline
        from models.swft_tc.src.serialization import read_detailed_jsonl, write_detailed_json
        from models.swft_tc.src.settings import load_config

        fixture = model_root.parents[1] / "tests" / "swft_tc" / "fixtures" / "town_country_reference_test.csv"
        config = load_config(
            model_root / "config" / "config_exp02_line1_protected_confidence_gated.yaml",
            base_dir=model_root,
            overrides={
                "processing": {"cache_enabled": False, "cache_path": str(tmp_path / "c.jsonl")},
                "reference_data": {"town_country_path": str(fixture)},
            },
        )
        group_config = load_group_config(config.path(config.project.group_config_path))
        frame = read_input_csv(
            model_root / "data" / "exp01_rightmost_town_allmatch_country_addresses.csv",
            record_id_column=config.project.record_id_column,
        )
        pipeline = Phase1Pipeline(
            config, group_config, client=mock_client, reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        )
        result = pipeline.run(frame)

        path = write_detailed_json(
            result.frame, tmp_path / "detail.jsonl", config=config, group_config=group_config,
            decisions_by_address=result.decisions_by_address, iso_provider=pipeline.iso_provider,
        )
        csv = result.frame.set_index("RECORD_ID")
        for doc in read_detailed_jsonl(path):
            block = doc["groups"]["1"]["retraction"]
            rid = doc["record_id"]
            assert block["policy_name"] == "line1_protected_confidence_gated"
            assert block["protected_source_fields"] == ["address_line_1"]
            # Line 1 is never modified for any record, whatever the stub predicted.
            assert block["actual_column_after_retraction"]["address_line_1"] == csv.loc[rid, "address_line_1"]
            assert block["actual_column_before_retraction"]["address_line_1"] == csv.loc[rid, "address_line_1"]
            assert block["combined_address_retracted"] == csv.loc[rid, "combined_address_retracted_group_1"]
            assert block["comment"] == csv.loc[rid, "combined_address_retracted_group_comments_1"]
