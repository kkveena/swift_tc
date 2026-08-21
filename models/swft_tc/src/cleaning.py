"""Deterministic address cleaning and token-boundary matching.

Nothing in this module is semantic. It normalizes, it does not rewrite: no
invented location data, no removed digits, no reordered lines.

The token-boundary matcher here is the deterministic half of the
anti-hallucination guarantee. ``AERONAUTICA`` contains the letters ``RONA``,
but ``RONA`` is not a *token* of ``AERONAUTICA``, so presence verification
rejects it. That check runs on the Python side and cannot be talked out of its
answer by the model.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

__all__ = [
    "MISSING_FIELD_LITERAL",
    "clean_address",
    "contains_token_phrase",
    "is_missing_field",
    "join_lines",
    "normalize_for_matching",
    "normalize_whitespace",
    "token_phrase_positions",
    "tokenize",
    "tokens_casefolded",
    "trim_field",
]

#: A source field whose *entire* trimmed value equals this literal is treated
#: as missing. Digits inside legitimate values ("10013-2632", "LEVEL 10",
#: "1140 NZ") are never touched.
MISSING_FIELD_LITERAL = "0"

_WHITESPACE = re.compile(r"\s+")


def trim_field(value: object) -> str:
    """Return ``value`` as a trimmed string; ``None``/NaN become ``""``."""
    if value is None:
        return ""
    # pandas is deliberately not imported here so this module stays
    # dependency-free and reusable. NaN is the only float unequal to itself.
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def is_missing_field(value: object, *, zero_is_missing: bool = True) -> bool:
    """Report whether a single configured source field contributes nothing.

    A field is missing when it is ``None``/NaN, empty or whitespace-only, or —
    when ``zero_is_missing`` — when its entire trimmed value is exactly ``"0"``.

    ``"0"`` alone is missing; ``"0 MAIN ST"``, ``"02111"`` and ``"LEVEL 10"``
    are not.
    """
    text = trim_field(value)
    if not text:
        return True
    if zero_is_missing and text == MISSING_FIELD_LITERAL:
        return True
    return False


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace (including newlines/tabs) to one space."""
    return _WHITESPACE.sub(" ", text).strip()


def clean_address(combined: str) -> str:
    """Deterministically clean a combined address.

    Unicode NFKC normalization, line separators folded to spaces, repeated
    whitespace collapsed, ends trimmed. No semantic rewriting whatsoever.

    NFKC already folds NBSP and other compatibility spaces into U+0020, so the
    whitespace collapse below catches them.
    """
    if not combined:
        return ""
    text = unicodedata.normalize("NFKC", combined)
    return normalize_whitespace(text)


def normalize_for_matching(text: str) -> str:
    """Normalize text for presence verification, preserving original case.

    NFKC, every non-alphanumeric character folded to a space, whitespace
    collapsed. Case is deliberately preserved so the caller can still
    distinguish an uppercase ISO code token from an ordinary lowercase word.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    folded = "".join(char if char.isalnum() else " " for char in normalized)
    return normalize_whitespace(folded)


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into alphanumeric tokens, preserving original case."""
    normalized = normalize_for_matching(text)
    if not normalized:
        return ()
    return tuple(normalized.split(" "))


def tokens_casefolded(text: str) -> tuple[str, ...]:
    """Uppercase token tuple used for case-insensitive comparison."""
    return tuple(token.upper() for token in tokenize(text))


def token_phrase_positions(
    haystack: str | Sequence[str],
    needle: str | Sequence[str],
) -> tuple[int, ...]:
    """Return every index where ``needle``'s tokens occur in ``haystack``'s.

    Matching is case-insensitive and respects token boundaries: the needle must
    align to whole tokens. Returns an empty tuple when the needle is empty or
    absent.
    """
    hay = _as_upper_tokens(haystack)
    ned = _as_upper_tokens(needle)
    if not ned or not hay or len(ned) > len(hay):
        return ()
    span = len(ned)
    return tuple(
        index
        for index in range(len(hay) - span + 1)
        if hay[index : index + span] == ned
    )


def contains_token_phrase(
    haystack: str | Sequence[str],
    needle: str | Sequence[str],
) -> bool:
    """Whether ``needle`` occurs in ``haystack`` on token boundaries.

    This is the substring-hallucination guard::

        contains_token_phrase("AERONAUTICA", "RONA")            -> False
        contains_token_phrase("ACCRA GREATER ACCRA GH", "ACCRA") -> True
    """
    return bool(token_phrase_positions(haystack, needle))


def join_lines(lines: Iterable[str], separator: str = " ") -> str:
    """Join already-trimmed, already-filtered address lines."""
    return separator.join(line for line in lines if line)


def _as_upper_tokens(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tokens_casefolded(value)
    return tuple(str(token).upper() for token in value if str(token))
