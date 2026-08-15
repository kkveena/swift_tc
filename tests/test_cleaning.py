"""Deterministic cleaning and the token-boundary anti-hallucination guard."""

from __future__ import annotations

import pytest

from swift_address.cleaning import (
    clean_address,
    contains_token_phrase,
    is_missing_field,
    normalize_for_matching,
    token_phrase_positions,
    tokenize,
    trim_field,
)


class TestMissingFieldPolicy:
    """A field is missing when null, blank, or entirely the literal "0"."""

    @pytest.mark.parametrize(
        "value",
        [None, "", "   ", "\t\n", "0", " 0 ", float("nan")],
    )
    def test_missing_values(self, value):
        assert is_missing_field(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "0 MAIN STREET",     # leading zero is part of the address
            "02111",             # postal code
            "10013-2632",        # hyphenated ZIP+4
            "LEVEL 10",          # digits inside legitimate text
            "1140 NZ",
            "00",                # two zeros is not the literal "0"
            "0.0",
            "PO BOX 0",
        ],
    )
    def test_retained_values(self, value):
        assert is_missing_field(value) is False

    def test_zero_policy_is_configurable(self):
        assert is_missing_field("0", zero_is_missing=False) is False

    def test_trim_field_handles_nan_and_none(self):
        assert trim_field(None) == ""
        assert trim_field(float("nan")) == ""
        assert trim_field("  PADDED  ") == "PADDED"


class TestCleanAddress:
    def test_collapses_whitespace_and_trims(self):
        assert clean_address("  23 CUSTOMS   STREET \t EAST  ") == "23 CUSTOMS STREET EAST"

    def test_line_separators_become_spaces(self):
        assert clean_address("LINE ONE\nLINE TWO\r\nLINE THREE") == (
            "LINE ONE LINE TWO LINE THREE"
        )

    def test_nfkc_normalization(self):
        # Fullwidth characters and a non-breaking space both normalize under NFKC.
        assert clean_address("ＴＯＫＹＯ １２３") == "TOKYO 123"

    def test_empty_input_stays_empty(self):
        assert clean_address("") == ""

    def test_does_not_rewrite_semantically(self):
        original = "388 GREENWICH STREET NEW YORK NY 10013-2632 US"
        assert clean_address(original) == original

    def test_is_idempotent(self):
        once = clean_address("  A   B  ")
        assert clean_address(once) == once


class TestTokenBoundaryMatching:
    """The deterministic half of the anti-substring-hallucination guarantee."""

    def test_aeronautica_does_not_contain_rona(self):
        assert contains_token_phrase("AERONAUTICA", "RONA") is False

    @pytest.mark.parametrize(
        "haystack,needle",
        [
            ("AERONAUTICA", "RONA"),
            ("AERONAUTICA", "AERO"),
            ("SANTANDER", "SANTA"),
            ("BOSTONIAN CLUB", "BOSTON"),
            ("NEWARK", "NEW"),
        ],
    )
    def test_substrings_are_rejected(self, haystack, needle):
        assert contains_token_phrase(haystack, needle) is False

    @pytest.mark.parametrize(
        "haystack,needle",
        [
            ("1 LINCOLN STREET BOSTON MA 02111 US", "BOSTON"),
            ("388 GREENWICH STREET NEW YORK NY 10013-2632 US", "NEW YORK"),
            ("25A CASTLE ROAD ACCRA GREATER ACCRA GH", "ACCRA"),
            ("boston ma", "BOSTON"),                     # case-insensitive
            ("HONG KONG SAR", "hong kong"),              # multi-token phrase
            ("SAINT-DENIS PARIS", "SAINT DENIS"),        # punctuation is a boundary
        ],
    )
    def test_whole_tokens_match(self, haystack, needle):
        assert contains_token_phrase(haystack, needle) is True

    def test_empty_needle_never_matches(self):
        assert contains_token_phrase("BOSTON MA", "") is False

    def test_positions_reports_every_occurrence(self):
        positions = token_phrase_positions("ACCRA GREATER ACCRA GH", "ACCRA")
        assert positions == (0, 2)

    def test_needle_longer_than_haystack(self):
        assert contains_token_phrase("US", "UNITED STATES OF AMERICA") is False


class TestNormalizationForMatching:
    def test_punctuation_folds_to_boundaries(self):
        assert normalize_for_matching("10013-2632, U.S.") == "10013 2632 U S"

    def test_case_is_preserved(self):
        # Presence checks need the original case to tell an ISO code from a word.
        assert normalize_for_matching("Boston ma US") == "Boston ma US"

    def test_tokenize_splits_on_non_alphanumerics(self):
        assert tokenize("441-445 JIRON") == ("441", "445", "JIRON")
