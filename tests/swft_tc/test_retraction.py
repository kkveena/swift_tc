"""Deterministic retraction of verified Town/Country evidence.

The safety properties here are the point of the module: a retraction that eats
part of an unrelated word, or strips a country code out of ordinary prose, is
worse than no retraction at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.swft_tc.src.retraction import (
    null_retraction,
    remove_token_phrases,
    retract_group,
    token_phrase_matches,
    token_spans,
)

FIELDS = ("L1", "L2", "L3")


def retract(values, *, town, country, town_exists, country_exists, iso_provider=None,
            fields=FIELDS):
    return retract_group(
        values, fields, town=town, country_value=country,
        town_exists=town_exists, country_exists=country_exists,
        iso_provider=iso_provider,
    )


class TestTokenSpans:
    def test_offsets_are_into_the_original_text(self):
        spans = token_spans("23 CUSTOMS ST")
        assert [(s.text, s.start, s.end) for s in spans] == [
            ("23", 0, 2), ("CUSTOMS", 3, 10), ("ST", 11, 13)
        ]

    def test_punctuation_separates_tokens(self):
        assert [s.text for s in token_spans("10013-2632, US")] == ["10013", "2632", "US"]


class TestRemovalSafety:
    """Substring collisions must survive untouched."""

    @pytest.mark.parametrize(
        "text,phrase",
        [
            ("AERONAUTICA", "RONA"),         # the canonical trap
            ("AERONAUTICA", "AERO"),
            ("23 CUSTOMS STREET EAST", "US"),
            ("SANTANDER PLAZA", "SANTA"),
            ("NEWARK", "NEW"),
            ("BOSTONIAN CLUB", "BOSTON"),
        ],
    )
    def test_substrings_are_never_removed(self, text, phrase):
        result, removed = remove_token_phrases(text, [phrase])
        assert result == text
        assert removed == ()

    def test_aeronautica_keeps_rona_through_the_group_api(self, iso_provider):
        result = retract(
            {"L1": "AERONAUTICA", "L2": "", "L3": ""},
            town="RONA", country="NO_COUNTRY",
            town_exists=False, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "AERONAUTICA"
        assert result.combined_address_retracted == "AERONAUTICA"
        assert result.retracted_entities == ()

    def test_customs_keeps_us_while_the_real_us_is_removed(self, iso_provider):
        result = retract(
            {"L1": "23 CUSTOMS STREET", "L2": "BOSTON MA 02111 US", "L3": ""},
            town="BOSTON", country="US",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "23 CUSTOMS STREET"
        assert result.after["L2"] == "MA 02111"

    def test_ambiguous_code_in_prose_is_kept(self, iso_provider):
        """"IN" mid-address is a preposition; only the trailing one is India."""
        result = retract(
            {"L1": "SUITE 5 IN TOWER", "L2": "MUMBAI 400001 IN", "L3": ""},
            town="MUMBAI", country="IN",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "SUITE 5 IN TOWER"   # prose survives
        assert result.after["L2"] == "400001"             # trailing code removed

    def test_ambiguous_code_not_verified_is_never_removed(self, iso_provider):
        result = retract(
            {"L1": "SUITE 5 IN TOWER", "L2": "MUMBAI", "L3": ""},
            town="MUMBAI", country="IN",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "SUITE 5 IN TOWER"


class TestRetractionPolicy:
    def test_both_explicit_removes_both(self, iso_provider):
        result = retract(
            {"L1": "23 CUSTOMS STREET EAST LEVEL 11",
             "L2": "CITIGROUP CENTRE AUCKLAND AUCKLAND", "L3": "1140 NZ"},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        # One AUCKLAND survives: the Town is retracted once per group, not once
        # per occurrence, so the building name keeps its own words.
        assert result.after == {
            "L1": "23 CUSTOMS STREET EAST LEVEL 11",
            "L2": "CITIGROUP CENTRE AUCKLAND",
            "L3": "1140",
        }
        assert result.retracted_entities == ("town", "country")
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_town_only_explicit_leaves_the_inferred_country_alone(self, iso_provider):
        result = retract(
            {"L1": "441-445 JIRON SANTA ROSA", "L2": "LIMA",
             "L3": "METRO MUNIC OF LIMA 15001"},
            town="LIMA", country="PE",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        # Only the right-most LIMA goes; the one standing alone on L2 stays.
        assert result.after["L2"] == "LIMA"
        assert result.after["L3"] == "METRO MUNIC OF 15001"
        assert result.retracted_entities == ("town",)
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)
        assert "PE" not in result.combined_address_retracted

    def test_country_only_explicit_removes_only_country(self, iso_provider):
        result = retract(
            {"L1": "PO BOX 1234", "L2": "US", "L3": ""},
            town="BOSTON", country="US",
            town_exists=False, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after == {"L1": "PO BOX 1234", "L2": "", "L3": ""}
        assert result.retracted_entities == ("country",)

    def test_neither_explicit_leaves_the_address_unchanged(self, iso_provider):
        values = {"L1": "HEAD OFFICE BUILDING", "L2": "", "L3": ""}
        result = retract(
            values, town="TAIPEI", country="TW",
            town_exists=False, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after == result.before == values
        assert result.retracted_entities == ()
        assert result.combined_address_retracted == "HEAD OFFICE BUILDING"

    def test_ambiguous_candidate_set_is_never_retracted(self, iso_provider):
        result = retract(
            {"L1": "MAIN BRANCH", "L2": "CA US", "L3": ""},
            town="EXAMPLE", country="CA,US",
            town_exists=False, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L2"] == "CA US"
        assert result.retracted_entities == ()

    def test_country_name_alias_form_is_removed(self, iso_provider):
        result = retract(
            {"L1": "1 QUEEN ST", "L2": "AUCKLAND NEW ZEALAND", "L3": ""},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L2"] == ""
        assert "New Zealand" in result.removed_forms


class TestOneTownOccurrencePerGroup:
    """A repeated Town is one locality stated twice, not two localities.

    The rule is: retract at most one standalone occurrence of the verified Town
    per *group*, and make it the right-most one. An earlier occurrence can be
    part of an institution, building or branch name; the later one is where the
    locality actually sits. The choice is positional and deterministic — nothing
    here guesses at meaning.
    """

    def test_adjacent_duplicate_leaves_one_behind(self, iso_provider):
        result = retract(
            {"L1": "23 CUSTOMS STREET EAST LEVEL 11",
             "L2": "CITIGROUP CENTRE AUCKLAND AUCKLAND", "L3": "1140 NZ"},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L2"] == "CITIGROUP CENTRE AUCKLAND"
        assert result.combined_address_retracted == (
            "23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE AUCKLAND 1140"
        )
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_repeated_across_address_lines_removes_only_the_last(self, iso_provider):
        """The cap spans the group: L2's LIMA survives because L3 has a later one."""
        result = retract(
            {"L1": "441-445 JIRON SANTA ROSA", "L2": "LIMA",
             "L3": "METRO MUNIC OF LIMA 15001"},
            town="LIMA", country="PE",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after == {
            "L1": "441-445 JIRON SANTA ROSA",
            "L2": "LIMA",
            "L3": "METRO MUNIC OF 15001",
        }
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_three_occurrences_lose_exactly_one(self, iso_provider):
        result = retract(
            {"L1": "AUCKLAND AUCKLAND AUCKLAND", "L2": "", "L3": ""},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "AUCKLAND AUCKLAND"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (3, 1)

    def test_multi_token_town_removes_only_the_last_whole_phrase(self, iso_provider):
        result = retract(
            {"L1": "NEW YORK LIFE BUILDING", "L2": "51 MADISON AVE NEW YORK",
             "L3": ""},
            town="NEW YORK", country="US",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "NEW YORK LIFE BUILDING"
        assert result.after["L2"] == "51 MADISON AVE"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_a_single_occurrence_is_still_removed(self, iso_provider):
        """The cap is a ceiling, not a requirement to leave something behind."""
        result = retract(
            {"L1": "1 QUEEN ST", "L2": "AUCKLAND", "L3": "1140"},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L2"] == ""
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (1, 1)

    def test_the_last_field_wins_even_when_an_earlier_field_repeats(self, iso_provider):
        """Two in L1, one in L3 -> the L3 one goes; both L1 occurrences stay."""
        result = retract(
            {"L1": "LIMA LIMA PLAZA", "L2": "SUITE 4", "L3": "LIMA 15001"},
            town="LIMA", country="PE",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "LIMA LIMA PLAZA"
        assert result.after["L3"] == "15001"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (3, 1)

    def test_country_still_loses_every_verified_occurrence(self, iso_provider):
        """The cap is Town-only. Country retraction is deliberately unchanged."""
        result = retract(
            {"L1": "PO BOX 1 NZ", "L2": "", "L3": "1140 NZ"},
            town="AUCKLAND", country="NZ",
            town_exists=False, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "PO BOX 1"
        assert result.after["L3"] == "1140"
        assert result.town_occurrences_removed == 0

    def test_an_unverified_town_is_never_counted_or_removed(self, iso_provider):
        result = retract(
            {"L1": "AUCKLAND AUCKLAND", "L2": "", "L3": ""},
            town="AUCKLAND", country="NZ",
            town_exists=False, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "AUCKLAND AUCKLAND"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (0, 0)

    def test_a_town_inside_a_region_name_is_still_only_positional(self, iso_provider):
        """A documented edge of the positional rule, pinned so it stays visible.

        "ACCRA GREATER ACCRA" names the city and then the region that contains
        it. The right-most occurrence is the region's second word, so retraction
        leaves "ACCRA GREATER" — a dangling region word. That is what a purely
        positional rule does, and the alternative would be semantic guessing,
        which this module deliberately refuses. The previous remove-all rule left
        a dangling "GREATER" too, so nothing regressed here.
        """
        result = retract(
            {"L1": "25A CASTLE ROAD AMBASSADORIAL AREA",
             "L2": "ACCRA GREATER ACCRA GH", "L3": "0"},
            town="ACCRA", country="GH",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L2"] == "ACCRA GREATER"
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (2, 1)

    def test_a_repeated_town_that_is_only_a_substring_is_not_counted(self, iso_provider):
        """BOSTONIAN is not a second BOSTON, so there is only one occurrence."""
        result = retract(
            {"L1": "THE BOSTONIAN CLUB", "L2": "BOSTON", "L3": ""},
            town="BOSTON", country="US",
            town_exists=True, country_exists=False, iso_provider=iso_provider,
        )
        assert result.after["L1"] == "THE BOSTONIAN CLUB"
        assert result.after["L2"] == ""
        assert (result.town_occurrences_found, result.town_occurrences_removed) == (1, 1)


class TestOccurrenceCappedRemoval:
    """The `max_occurrences` / `prefer_last` extension to the removal utility."""

    def test_default_is_unchanged(self):
        assert remove_token_phrases("A LIMA B LIMA C", ["LIMA"])[0] == "A B C"

    def test_cap_of_one_prefers_the_last(self):
        assert remove_token_phrases(
            "A LIMA B LIMA C", ["LIMA"], max_occurrences=1, prefer_last=True
        )[0] == "A LIMA B C"

    def test_cap_of_one_without_prefer_last_takes_the_first(self):
        assert remove_token_phrases(
            "A LIMA B LIMA C", ["LIMA"], max_occurrences=1
        )[0] == "A B LIMA C"

    def test_cap_above_the_occurrence_count_removes_them_all(self):
        assert remove_token_phrases(
            "A LIMA B LIMA C", ["LIMA"], max_occurrences=5, prefer_last=True
        )[0] == "A B C"

    def test_a_zero_cap_removes_nothing(self):
        text = "A LIMA B"
        assert remove_token_phrases(text, ["LIMA"], max_occurrences=0) == (text, ())

    def test_the_cap_is_per_phrase_not_per_call(self):
        result, _ = remove_token_phrases(
            "LIMA LIMA PE PE", ["LIMA", "PE"], max_occurrences=1, prefer_last=True
        )
        assert result == "LIMA PE"

    def test_capped_removal_keeps_substring_safety(self):
        text = "AERONAUTICA AERONAUTICA"
        assert remove_token_phrases(
            text, ["RONA"], max_occurrences=1, prefer_last=True
        ) == (text, ())


class TestTokenPhraseMatches:
    def test_reports_every_non_overlapping_match(self):
        assert token_phrase_matches("A LIMA B LIMA C", "LIMA") == ((1, 1), (3, 3))

    def test_multi_token_phrase_spans_its_tokens(self):
        assert token_phrase_matches("51 NEW YORK AVE", "NEW YORK") == ((1, 2),)

    def test_substring_occurrences_are_not_matches(self):
        assert token_phrase_matches("AERONAUTICA", "RONA") == ()

    def test_trailing_restriction_is_honoured(self):
        assert token_phrase_matches("IN A B IN", "IN", restrict_to_trailing_tokens=2) == (
            (3, 3),
        )


class TestRepeatedOccurrencesAndArtifacts:
    def test_the_default_removes_every_standalone_occurrence(self):
        """The utility default is unchanged; only the Town *rule* caps removals."""
        result, removed = remove_token_phrases(
            "CITIGROUP CENTRE AUCKLAND AUCKLAND", ["AUCKLAND"]
        )
        assert result == "CITIGROUP CENTRE"
        assert removed == ("AUCKLAND",)

    def test_postal_code_survives_country_removal(self):
        assert remove_token_phrases("1140 NZ", ["NZ"])[0] == "1140"

    def test_orphaned_separators_are_repaired(self):
        assert remove_token_phrases("CITIGROUP CENTRE, AUCKLAND, 1140", ["AUCKLAND"])[0] == (
            "CITIGROUP CENTRE, 1140"
        )

    def test_leading_removal_does_not_leave_a_separator(self):
        assert remove_token_phrases("AUCKLAND, 1140 NZ", ["AUCKLAND"])[0] == "1140 NZ"

    def test_whitespace_is_normalized(self):
        assert remove_token_phrases("A   AUCKLAND   B", ["AUCKLAND"])[0] == "A B"

    def test_surviving_text_keeps_its_original_casing(self):
        result, _ = remove_token_phrases("Level 11 Auckland Tower", ["AUCKLAND"])
        assert result == "Level 11 Tower"

    def test_multi_token_town_is_removed_as_a_phrase(self):
        assert remove_token_phrases("388 GREENWICH ST NEW YORK NY", ["NEW YORK"])[0] == (
            "388 GREENWICH ST NY"
        )

    def test_removing_everything_yields_an_empty_field(self):
        assert remove_token_phrases("AUCKLAND", ["AUCKLAND"])[0] == ""


class TestComment:
    def _comment(self, iso_provider, **kwargs):
        return retract(**kwargs, iso_provider=iso_provider).comment

    def test_both_retracted(self, iso_provider):
        comment = self._comment(
            iso_provider,
            values={"L1": "1 LINCOLN ST", "L2": "BOSTON MA 02111 US", "L3": ""},
            town="BOSTON", country="US", town_exists=True, country_exists=True,
        )
        assert comment == (
            "Retracted Town=BOSTON and Country=US from verified explicit address evidence."
        )

    def test_town_only(self, iso_provider):
        comment = self._comment(
            iso_provider,
            values={"L1": "LIMA", "L2": "", "L3": ""},
            town="LIMA", country="PE", town_exists=True, country_exists=False,
        )
        assert comment.startswith("Retracted Town=LIMA.")
        assert "retained only as a prediction" in comment

    def test_nothing_retracted(self, iso_provider):
        comment = self._comment(
            iso_provider,
            values={"L1": "HEAD OFFICE", "L2": "", "L3": ""},
            town="TAIPEI", country="TW", town_exists=False, country_exists=False,
        )
        assert comment.startswith("No retraction:")

    @pytest.mark.parametrize(
        "town_exists,country_exists",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_comment_is_at_most_three_lines(self, iso_provider, town_exists, country_exists):
        comment = self._comment(
            iso_provider,
            values={"L1": "1 LINCOLN ST", "L2": "BOSTON MA 02111 US", "L3": ""},
            town="BOSTON", country="US",
            town_exists=town_exists, country_exists=country_exists,
        )
        assert len(comment.splitlines()) <= 3

    def test_comment_is_deterministic(self, iso_provider):
        kwargs = dict(
            values={"L1": "1 LINCOLN ST", "L2": "BOSTON MA 02111 US", "L3": ""},
            town="BOSTON", country="US", town_exists=True, country_exists=True,
        )
        assert self._comment(iso_provider, **kwargs) == self._comment(
            iso_provider, **kwargs
        )


class TestSourceColumnLevelWork:
    def test_before_and_after_are_reported_per_column(self, iso_provider):
        result = retract(
            {"L1": "23 CUSTOMS STREET EAST LEVEL 11",
             "L2": "CITIGROUP CENTRE AUCKLAND AUCKLAND", "L3": "1140 NZ"},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert set(result.before) == set(result.after) == set(FIELDS)
        assert result.before["L2"] == "CITIGROUP CENTRE AUCKLAND AUCKLAND"
        assert result.after["L2"] == "CITIGROUP CENTRE AUCKLAND"

    def test_combined_is_rebuilt_from_the_after_values(self, iso_provider):
        from models.swft_tc.src.cleaning import clean_address
        from models.swft_tc.src.grouping import build_combined_address

        result = retract(
            {"L1": "23 CUSTOMS STREET EAST LEVEL 11",
             "L2": "CITIGROUP CENTRE AUCKLAND AUCKLAND", "L3": "1140 NZ"},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        rebuilt = clean_address(
            build_combined_address([result.after[name] for name in FIELDS])
        )
        assert result.combined_address_retracted == rebuilt

    def test_emptied_fields_drop_out_of_the_rebuild(self, iso_provider):
        result = retract(
            {"L1": "AUCKLAND", "L2": "1140 NZ", "L3": "0"},
            town="AUCKLAND", country="NZ",
            town_exists=True, country_exists=True, iso_provider=iso_provider,
        )
        assert result.after["L1"] == ""
        assert result.combined_address_retracted == "1140"

    def test_source_values_mapping_is_not_mutated(self, iso_provider):
        values = {"L1": "AUCKLAND", "L2": "1140 NZ", "L3": ""}
        snapshot = dict(values)
        retract(values, town="AUCKLAND", country="NZ",
                town_exists=True, country_exists=True, iso_provider=iso_provider)
        assert values == snapshot

    def test_null_retraction_is_empty(self):
        result = null_retraction(FIELDS)
        assert result.after == {name: "" for name in FIELDS}
        assert result.combined_address_retracted == ""
        assert result.comment == ""


class TestPipelineIntegration:
    def test_original_input_columns_are_unchanged(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        original = read_input_csv(sample_input_path)
        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(original.copy())

        pd.testing.assert_frame_equal(
            result.frame[original.columns], original, check_dtype=False
        )

    def test_retracted_columns_are_populated(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        frame = result.frame.set_index("RECORD_ID")
        auckland = frame.loc["CA0000001855"]
        assert auckland["combined_address_retracted_group_15"] == (
            "23 CUSTOMS STREET EAST LEVEL 11 CITIGROUP CENTRE AUCKLAND 1140"
        )
        assert "AUCKLAND" in auckland["combined_address_retracted_group_comments_15"]

        # The substring trap, end to end.
        assert frame.loc["CA0000002679", "combined_address_retracted_group_15"] == (
            "AERONAUTICA"
        )

    def test_empty_group_has_blank_retraction_fields(
        self, config, group_config, reference_provider, town_country_provider,
        mock_client, sample_input_path,
    ):
        from models.swft_tc.src.io import read_input_csv
        from models.swft_tc.src.pipeline import Phase1Pipeline

        result = Phase1Pipeline(
            config, group_config, client=mock_client,
            reference_provider=reference_provider,
            town_country_provider=town_country_provider, mode="dry_run",
        ).run(read_input_csv(sample_input_path))

        row = result.frame.set_index("RECORD_ID").loc["CA0000000863"]
        assert row["combined_address_retracted_group_15"] == ""
        assert row["combined_address_retracted_group_comments_15"] == ""
