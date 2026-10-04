"""EXP-03 — entity-span protected, confidence-gated retraction.

Entity detection decides WHICH TEXT retraction may not touch, and nothing else.
These tests pin: span verification (never trust the model), the strict
confidence gate, span protection in any source field, the untouched EXP-01 and
EXP-02 behaviour, and the conditional 28-column contract.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from models.swft_tc.src.entity_detection import (
    ENTITY_FIELD_KEYS,
    ENTITY_TYPES,
    NO_ENTITY,
    EntityDetectionResult,
    EntityDetector,
    EntityResponse,
    MockEntityClient,
    build_entity_payload,
    entity_payload_fingerprint,
    make_entity_cache_key,
    parse_entity_response,
    verify_entity_spans,
)
from models.swft_tc.src.cache import AddressCache
from models.swft_tc.src.retraction import (
    BASELINE_POLICY,
    SKIP_PROTECTED_ENTITY_SPAN_ONLY,
    SKIP_PROTECTED_ONLY,
    RetractionPolicy,
    retract_group,
    token_phrase_matches,
)

FIELDS = ("address_line_1", "address_line_2", "address_line_3", "address_line_4")
EXP02 = RetractionPolicy(name="line1_protected_confidence_gated", protected_source_positions=(1,),
                         town_probability_threshold=0.80, country_probability_threshold=0.80)
EXP03 = RetractionPolicy(name="entity_span_protected_confidence_gated", protected_source_positions=(),
                         town_probability_threshold=0.80, country_probability_threshold=0.80)


def response(name, etype, spans, confidence=0.99):
    return parse_entity_response({
        "entity_name": name, "entity_type": etype, "entity_rationale": "test",
        "entity_confidence": confidence,
        "entity_spans": [{"source_field": f, "text": t} for f, t in spans],
    })


def entity(values, name, etype, spans, confidence=0.99, threshold=0.80):
    full = {k: values.get(k, "") for k in FIELDS}
    return EntityDetectionResult.from_response(
        response(name, etype, spans, confidence), source_fields=FIELDS, source_values=full,
        prompt_version="exp03-entity-identification-v1", model="test", threshold=threshold, cache_hit=False)


def run(values, *, town, country, town_exists=True, country_exists=True, town_p=0.99, country_p=0.99,
        policy=EXP03, ent=None, iso_provider=None):
    full = {k: values.get(k, "") for k in FIELDS}
    return retract_group(full, FIELDS, town=town, country_value=country, town_exists=town_exists,
                         country_exists=country_exists, iso_provider=iso_provider,
                         town_probability=town_p, country_probability=country_p, policy=policy,
                         protected_spans=ent.protected_spans_by_field() if ent is not None else None)


# --------------------------------------------------------------------------
class TestResponseParsing:
    def test_taxonomy(self):
        assert "BANK" in ENTITY_TYPES and NO_ENTITY in ENTITY_TYPES and len(ENTITY_TYPES) == 8

    def test_bank_type_is_accepted(self):
        r = response("NATIONAL BANK OF CANADA", "BANK", [("address_line_1", "NATIONAL BANK OF CANADA")])
        assert r.entity_type == "BANK" and r.is_entity

    def test_unknown_type_is_normalised_to_other_with_a_note(self):
        r = response("ACME", "CONGLOMERATE", [("address_line_1", "ACME")])
        assert r.entity_type == "OTHER"
        assert any("outside the taxonomy" in n for n in r.normalisation_notes)

    def test_no_entity_empties_name_and_spans(self):
        r = response("SOMETHING", "NO_ENTITY", [("address_line_1", "SOMETHING")])
        assert (r.entity_name, r.entity_type, r.entity_spans) == (NO_ENTITY, NO_ENTITY, ())

    def test_confidence_is_clamped(self):
        assert response("X", "OTHER", [], confidence=7).entity_confidence == 1.0
        assert response("X", "OTHER", [], confidence="nope").entity_confidence == 0.0

    def test_three_flat_fields(self):
        assert ENTITY_FIELD_KEYS == ("entity_name", "entity_type", "entity_rationale")


class TestSpanVerification:
    VALUES = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_2": "600 RUE",
              "address_line_3": "SUITE 610", "address_line_4": "MONTREAL QC H3B 4L3 CANADA"}

    def test_exact_span_verifies_with_original_offsets(self):
        v, r = verify_entity_spans(response("x", "BANK", [("address_line_1", "national bank of canada")]),
                                   self.VALUES, FIELDS)
        assert r == () and len(v) == 1
        assert (v[0].source_field, v[0].text, v[0].start, v[0].end) == ("address_line_1", "NATIONAL BANK OF CANADA", 0, 23)

    def test_hallucinated_span_is_rejected(self):
        v, r = verify_entity_spans(response("x", "BANK", [("address_line_1", "ROYAL BANK OF CANADA")]), self.VALUES, FIELDS)
        assert v == () and r[0].reason == "text_not_found_on_token_boundaries"

    def test_invalid_source_field_is_rejected(self):
        v, r = verify_entity_spans(response("x", "BANK", [("address_line_9", "NATIONAL BANK OF CANADA")]), self.VALUES, FIELDS)
        assert v == () and r[0].reason == "unknown_source_field"

    def test_subphrase_matching_stays_token_safe(self):
        values = {"address_line_1": "THE BOSTONIAN CLUB"}
        v, r = verify_entity_spans(response("x", "OTHER", [("address_line_1", "BOSTON")]), values, FIELDS)
        assert v == () and r[0].reason == "text_not_found_on_token_boundaries"

    def test_rejected_spans_protect_nothing(self):
        ent = entity(self.VALUES, "ROYAL BANK OF CANADA", "BANK", [("address_line_1", "ROYAL BANK OF CANADA")])
        assert ent.protected_spans_by_field() == {} and len(ent.rejected_spans) == 1

    def test_partial_span_within_a_field(self):
        values = {"address_line_1": "CITIBANK QATAR DOHA"}
        v, _ = verify_entity_spans(response("x", "BANK", [("address_line_1", "CITIBANK QATAR")]), values, FIELDS)
        assert (v[0].start, v[0].end, v[0].text) == (0, 14, "CITIBANK QATAR")


class TestConfidenceGate:
    VALUES = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL CANADA"}

    def test_exactly_point_eight_does_not_protect(self):
        ent = entity(self.VALUES, "NATIONAL BANK OF CANADA", "BANK", [("address_line_1", "NATIONAL BANK OF CANADA")], confidence=0.80)
        assert ent.protection_applies is False and ent.protected_spans_by_field() == {}

    def test_just_above_point_eight_protects(self):
        ent = entity(self.VALUES, "NATIONAL BANK OF CANADA", "BANK", [("address_line_1", "NATIONAL BANK OF CANADA")], confidence=0.8001)
        assert ent.protection_applies is True
        assert ent.protected_spans_by_field() == {"address_line_1": ((0, 23),)}

    def test_no_threshold_means_any_verified_span_protects(self):
        ent = entity(self.VALUES, "NATIONAL BANK OF CANADA", "BANK", [("address_line_1", "NATIONAL BANK OF CANADA")], confidence=0.1, threshold=None)
        assert ent.protection_applies is True

    def test_no_entity_never_protects(self):
        ent = entity(self.VALUES, NO_ENTITY, NO_ENTITY, [], confidence=0.99)
        assert ent.protection_applies is False


class TestSpanProtectedRetraction:
    @pytest.mark.parametrize("field,line", [("address_line_1", 1), ("address_line_2", 2),
                                             ("address_line_3", 3), ("address_line_4", 4)])
    def test_entity_span_in_any_field_protects_its_country_word(self, field, line, iso_provider):
        values = {k: "" for k in FIELDS}
        values[field] = "NATIONAL BANK OF CANADA"
        other = "address_line_4" if field != "address_line_4" else "address_line_3"
        values[other] = "MONTREAL QC H3B 4L3 CANADA"
        ent = entity(values, "NATIONAL BANK OF CANADA", "BANK", [(field, "NATIONAL BANK OF CANADA")])
        result = run(values, town="MONTREAL", country="CA", ent=ent, iso_provider=iso_provider)
        assert result.after[field] == "NATIONAL BANK OF CANADA", f"line {line} entity damaged"
        assert result.after[other] == "QC H3B 4L3"
        assert result.country_occurrences_protected_by_entity == 1

    def test_occurrence_outside_the_span_remains_retractable(self, iso_provider):
        values = {"address_line_1": "FUBON BANK HONG KONG LTD", "address_line_2": "38 DESVOEUX ROAD",
                  "address_line_3": "FUBON BANK BUILDING HONG KONG HK"}
        ent = entity(values, "FUBON BANK HONG KONG LTD", "BANK", [("address_line_1", "FUBON BANK HONG KONG LTD")])
        result = run(values, town="HONG KONG", country="HK", ent=ent, iso_provider=iso_provider)
        assert result.after["address_line_1"] == "FUBON BANK HONG KONG LTD"
        assert result.after["address_line_3"] == "FUBON BANK BUILDING"
        assert (result.town_occurrences_found, result.town_occurrences_removed, result.town_occurrences_protected_by_entity) == (2, 1, 1)

    def test_same_phrase_inside_entity_and_later_in_the_same_field(self, iso_provider):
        values = {"address_line_1": "CITIBANK QATAR DOHA", "address_line_3": "TOWER 1 DOHA DOHA QA"}
        ent = entity(values, "CITIBANK QATAR", "BANK", [("address_line_1", "CITIBANK QATAR")])
        result = run(values, town="DOHA", country="QA", ent=ent, iso_provider=iso_provider)
        # QATAR is inside the span and survives; the line-1 DOHA is outside it and
        # under EXP-03 (no blanket line-1 protection) the right-most DOHA goes.
        assert result.after["address_line_1"] == "CITIBANK QATAR DOHA"
        assert result.after["address_line_3"] == "TOWER 1 DOHA"
        assert result.country_occurrences_protected_by_entity == 1

    def test_town_only_inside_the_span_is_not_retracted(self, iso_provider):
        values = {"address_line_1": "BANCO LAFISE PANAMA SA", "address_line_2": "CALLE 50"}
        ent = entity(values, "BANCO LAFISE PANAMA SA", "BANK", [("address_line_1", "BANCO LAFISE PANAMA SA")])
        result = run(values, town="PANAMA", country="PA", country_exists=False, ent=ent, iso_provider=iso_provider)
        assert result.after == result.before
        assert result.town_skip_reason == SKIP_PROTECTED_ENTITY_SPAN_ONLY
        assert "inside the verified entity span" in result.comment

    def test_exp03_does_not_blanket_protect_line_one(self, iso_provider):
        """No entity span: EXP-03 may change line 1 where EXP-02 could not."""
        values = {"address_line_1": "LIMA OFFICE", "address_line_2": "AV LARCO 1301"}
        result = run(values, town="LIMA", country="PE", country_exists=False, ent=None, iso_provider=iso_provider)
        assert result.after["address_line_1"] == "OFFICE"
        exp02 = run(values, town="LIMA", country="PE", country_exists=False, policy=EXP02, iso_provider=iso_provider)
        assert exp02.after["address_line_1"] == "LIMA OFFICE"
        assert exp02.town_skip_reason == SKIP_PROTECTED_ONLY

    def test_entity_in_line_three_protects_where_exp02_would_not(self, iso_provider):
        values = {"address_line_1": "25 MAIN STREET", "address_line_2": "LEVEL 4",
                  "address_line_3": "NATIONAL BANK OF CANADA", "address_line_4": "TORONTO ON CANADA"}
        ent = entity(values, "NATIONAL BANK OF CANADA", "BANK", [("address_line_3", "NATIONAL BANK OF CANADA")])
        exp03 = run(values, town="TORONTO", country="CA", ent=ent, iso_provider=iso_provider)
        exp02 = run(values, town="TORONTO", country="CA", policy=EXP02, iso_provider=iso_provider)
        assert exp03.after["address_line_3"] == "NATIONAL BANK OF CANADA"
        assert exp02.after["address_line_3"] == "NATIONAL BANK OF"   # EXP-02 only shields line 1
        assert exp03.after["address_line_4"] == exp02.after["address_line_4"] == "ON"

    def test_no_entity_gives_normal_confidence_gated_retraction(self, iso_provider):
        values = {"address_line_1": "HQ", "address_line_3": "AUCKLAND 1140 NZ"}
        ent = entity(values, NO_ENTITY, NO_ENTITY, [], confidence=0.99)
        assert run(values, town="AUCKLAND", country="NZ", ent=ent, iso_provider=iso_provider).after["address_line_3"] == "1140"
        assert run(values, town="AUCKLAND", country="NZ", town_p=0.80, ent=ent, iso_provider=iso_provider).after["address_line_3"] == "AUCKLAND 1140"

    def test_gated_confidence_leaves_the_entity_unprotected(self, iso_provider):
        values = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL CANADA"}
        ent = entity(values, "NATIONAL BANK OF CANADA", "BANK", [("address_line_1", "NATIONAL BANK OF CANADA")], confidence=0.80)
        result = run(values, town="MONTREAL", country="CA", ent=ent, iso_provider=iso_provider)
        assert result.after["address_line_1"] == "NATIONAL BANK OF"   # == 0.80 is not > 0.80
        assert result.protected_entity_spans == ()

    def test_protected_span_is_recorded_in_the_audit(self, iso_provider):
        values = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL CANADA"}
        ent = entity(values, "NATIONAL BANK OF CANADA", "BANK", [("address_line_1", "NATIONAL BANK OF CANADA")])
        result = run(values, town="MONTREAL", country="CA", ent=ent, iso_provider=iso_provider)
        assert result.protected_entity_spans == ({"source_field": "address_line_1", "start": 0, "end": 23,
                                                  "text": "NATIONAL BANK OF CANADA"},)
        assert result.to_dict()["country_occurrences_protected_by_entity"] == 1

    def test_country_inside_span_plus_several_later_occurrences(self, iso_provider):
        values = {"address_line_1": "CITIBANK CANADA", "address_line_2": "CANADA SQUARE",
                  "address_line_3": "TORONTO ON CANADA", "address_line_4": "CA"}
        ent = entity(values, "CITIBANK CANADA", "BANK", [("address_line_1", "CITIBANK CANADA")])
        result = run(values, town="TORONTO", country="CA", ent=ent, iso_provider=iso_provider)
        assert result.after["address_line_1"] == "CITIBANK CANADA"
        assert (result.after["address_line_2"], result.after["address_line_3"], result.after["address_line_4"]) == ("SQUARE", "ON", "")

    def test_town_then_country_in_a_field_with_a_span(self, iso_provider):
        """Offsets are re-derived after the Town is cut, so the span still holds."""
        values = {"address_line_2": "LIMA BANCO DE PERU PE LIMA"}
        ent = entity(values, "BANCO DE PERU", "BANK", [("address_line_2", "BANCO DE PERU")])
        result = run(values, town="LIMA", country="PE", ent=ent, iso_provider=iso_provider)
        assert "BANCO DE PERU" in result.after["address_line_2"]
        assert result.after["address_line_2"] == "LIMA BANCO DE PERU"


class TestTokenProtection:
    def test_protected_span_excludes_overlapping_matches_only(self):
        text = "CANADA SQUARE CANADA"
        assert token_phrase_matches(text, "CANADA") == ((0, 0), (2, 2))
        assert token_phrase_matches(text, "CANADA", protected_spans=[(0, 6)]) == ((2, 2),)


class TestBaselinesUnchanged:
    def test_exp01_ignores_entity_spans_entirely_when_none_supplied(self, iso_provider):
        values = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL QC H3B 4L3 CANADA"}
        result = run(values, town="MONTREAL", country="CA", policy=BASELINE_POLICY, iso_provider=iso_provider)
        assert result.after["address_line_1"] == "NATIONAL BANK OF"
        assert result.comment == "Retracted Town=MONTREAL and Country=CA from verified explicit address evidence."
        assert result.protected_entity_spans == ()

    def test_exp02_behaviour_and_wording_unchanged(self, iso_provider):
        values = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL QC H3B 4L3 CANADA"}
        result = run(values, town="MONTREAL", country="CA", policy=EXP02, iso_provider=iso_provider)
        assert result.after["address_line_1"] == "NATIONAL BANK OF CANADA"
        assert result.after["address_line_4"] == "QC H3B 4L3"
        assert result.comment == (
            "[line1_protected_confidence_gated] Retracted Town=MONTREAL from eligible address lines; "
            "occurrences in protected line(s) [address_line_1] were preserved. Retracted Country=CA from "
            "eligible address lines; occurrences in protected line(s) [address_line_1] were preserved.")

    def test_baseline_configs_stay_at_twenty_fields(self, model_root):
        from models.swft_tc.src.settings import load_config
        for name in ("config.yaml", "config_exp01_rightmost_town_allmatch_country.yaml",
                     "config_exp02_line1_protected_confidence_gated.yaml"):
            c = load_config(model_root / "config" / name, base_dir=model_root)
            assert c.fields_per_group == 20 and c.entity_detection.enabled is False

    def test_exp03_config_has_twenty_three_fields_and_no_positional_protection(self, model_root):
        from models.swft_tc.src.settings import load_config
        c = load_config(model_root / "config" / "config_exp03_entity_span_protected_confidence_gated.yaml", base_dir=model_root)
        assert c.fields_per_group == 23
        assert c.group_column_names("1")[-3:] == ("entity_name_group_1", "entity_type_group_1", "entity_rationale_group_1")
        assert c.retraction.protected_source_positions == ()
        assert c.entity_detection.protection_confidence_threshold == 0.80
        assert c.project.group_config_path == "config/group_config_exp01_rightmost_town_allmatch_country.csv"


class TestCacheAndDetector:
    def test_payload_carries_field_boundaries_and_fingerprint_is_stable(self):
        values = {"address_line_1": "A", "address_line_2": "B"}
        payload = json.loads(build_entity_payload(FIELDS, values))
        assert [f["field_name"] for f in payload["source_fields"]] == list(FIELDS)
        assert entity_payload_fingerprint(FIELDS, values) == entity_payload_fingerprint(FIELDS, dict(values))
        assert entity_payload_fingerprint(FIELDS, values) != entity_payload_fingerprint(FIELDS, {"address_line_1": "B", "address_line_2": "A"})

    def test_cache_key_is_separate_from_the_extraction_key(self):
        from models.swft_tc.src.cache import make_cache_key
        fp = entity_payload_fingerprint(FIELDS, {"address_line_1": "NATIONAL BANK OF CANADA"})
        assert make_entity_cache_key(prompt_version="v", model="m", fingerprint=fp) != make_cache_key(
            prompt_version="v", model="m", address="NATIONAL BANK OF CANADA", reference_context_version="")

    def test_detector_calls_once_then_hits_cache(self, tmp_path):
        client = MockEntityClient()
        det = EntityDetector(client=client, cache=AddressCache(tmp_path / "e.jsonl"), prompt_version="v", protection_confidence_threshold=0.80)
        values = {"address_line_1": "NATIONAL BANK OF CANADA", "address_line_4": "MONTREAL"}
        first = det.detect(FIELDS, values); second = det.detect(FIELDS, values)
        assert client.call_count == 1 and first.cache_hit is False and second.cache_hit is True
        assert first.entity_type == "BANK" and first.verified_spans[0].source_field == "address_line_1"
        assert det.stats == {"entity_cache_hits": 1, "entity_cache_misses": 1, "entity_errors": 0, "entity_backend_calls": 1}


class TestPipelineIntegration:
    def _config(self, model_root, tmp_path):
        from models.swft_tc.src.settings import load_config
        fixture = model_root.parents[1] / "tests" / "swft_tc" / "fixtures" / "town_country_reference_test.csv"
        return load_config(
            model_root / "config" / "config_exp03_entity_span_protected_confidence_gated.yaml", base_dir=model_root,
            overrides={"processing": {"cache_enabled": False, "cache_path": str(tmp_path / "c.jsonl")},
                       "entity_detection": {"cache_path": str(tmp_path / "e.jsonl")},
                       "reference_data": {"town_country_path": str(fixture)}})

    def test_enabled_without_a_client_is_an_explicit_error(self, model_root, tmp_path, reference_provider, mock_client):
        from models.swft_tc.src.grouping import load_group_config
        from models.swft_tc.src.pipeline import Phase1Pipeline
        config = self._config(model_root, tmp_path)
        with pytest.raises(ValueError, match="entity_client"):
            Phase1Pipeline(config, load_group_config(config.path(config.project.group_config_path)),
                           client=mock_client, reference_provider=reference_provider)

    def test_twenty_eight_columns_csv_json_agree_and_spans_never_altered(
        self, model_root, tmp_path, reference_provider, town_country_provider, mock_client,
    ):
        from models.swft_tc.src.grouping import load_group_config
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline
        from models.swft_tc.src.serialization import read_detailed_jsonl, write_detailed_json

        config = self._config(model_root, tmp_path)
        group_config = load_group_config(config.path(config.project.group_config_path))
        frame = read_input_csv(model_root / "data" / "exp01_rightmost_town_allmatch_country_addresses.csv",
                               record_id_column=config.project.record_id_column)
        entity_client = MockEntityClient()
        pipeline = Phase1Pipeline(config, group_config, client=mock_client, reference_provider=reference_provider,
                                  town_country_provider=town_country_provider, mode="dry_run",
                                  entity_client=entity_client, entity_cache=AddressCache(tmp_path / "e.jsonl"))
        result = pipeline.run(frame)
        assert result.frame.shape == (115, 28)
        assert list(result.frame.columns[-3:]) == ["entity_name_group_1", "entity_type_group_1", "entity_rationale_group_1"]
        assert entity_client.call_count == len(result.entity_results) > 0
        assert result.metrics["efficiency"]["entity_backend_calls"] == entity_client.call_count

        path = write_detailed_json(result.frame, tmp_path / "d.jsonl", config=config, group_config=group_config,
                                   decisions_by_address=result.decisions_by_address, iso_provider=pipeline.iso_provider,
                                   entity_results=result.entity_results)
        csv = result.frame.set_index("RECORD_ID")
        for doc in read_detailed_jsonl(path):
            g = doc["groups"]["1"]; rid = doc["record_id"]
            assert "entity_detection" in g and g["entity_detection"]["prompt_version"] == "exp03-entity-identification-v1"
            assert g["entity_detection"]["entity_name"] == csv.loc[rid, "entity_name_group_1"]
            assert g["retraction"]["combined_address_retracted"] == csv.loc[rid, "combined_address_retracted_group_1"]
            assert g["retraction"]["policy_name"] == "entity_span_protected_confidence_gated"
            # The success criterion: no verified, protected span is ever altered.
            after = g["retraction"]["actual_column_after_retraction"]
            for span in g["retraction"]["protected_entity_spans"]:
                assert span["text"] in after[span["source_field"]], (rid, span)
