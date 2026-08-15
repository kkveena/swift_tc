"""The Gemini structured-output contract.

This module owns three things and nothing else owns them:

1. the prompt text, loaded from ``prompts/GEMINI_EXTRACTION_PROMPT.md`` — the
   single source of truth, never re-embedded as a string literal elsewhere;
2. the JSON Schema sent to the model as ``response_schema``;
3. :class:`ExtractionResponse`, which every model response is validated
   against before anything touches the dataframe.

A response that violates the schema raises :class:`MalformedExtractionResponse`.
It never degrades into a plausible-looking ``NO_TOWN``/``NO_COUNTRY`` row — an
API or parsing failure is an error, not a business conclusion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "ExtractionResponse",
    "MalformedExtractionResponse",
    "NO_COUNTRY",
    "NO_TOWN",
    "PromptContract",
    "REQUIRED_RESPONSE_FIELDS",
    "RESPONSE_JSON_SCHEMA",
    "build_user_payload",
    "load_prompt_contract",
    "parse_extraction_response",
]

#: Sentinels written to the CSV when no defensible value exists. They are
#: outputs of the *pipeline*, never values the model is asked to invent.
NO_TOWN = "NO_TOWN"
NO_COUNTRY = "NO_COUNTRY"

#: Internal response fields the model must return. Not all reach the CSV; the
#: evidence and flag fields drive Python-side verification and the audit trail.
REQUIRED_RESPONSE_FIELDS: tuple[str, ...] = (
    "town",
    "country_candidates",
    "town_evidence",
    "country_evidence",
    "town_is_explicit",
    "country_is_explicit",
    "town_ambiguous",
    "country_ambiguous",
    "town_model_confidence",
    "country_model_confidence",
    "town_rationale",
    "country_rationale",
    "reference_basis",
)

_ALPHA2_RE = re.compile(r"^[A-Z]{2}$")

#: JSON Schema handed to the Gemini SDK as the response schema. Kept as a plain
#: dict (rather than generated from the pydantic model) so what the model is
#: told matches, field for field, the prompt contract in
#: prompts/GEMINI_EXTRACTION_PROMPT.md.
RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "town": {
            "type": "string",
            "description": "Uppercase town name, or NO_TOWN when not defensible.",
        },
        "country_candidates": {
            "type": "array",
            "description": (
                "All defensible ISO 3166-1 alpha-2 uppercase codes. Empty when no "
                "country is defensible. More than one entry means unresolved "
                "ambiguity; never pick one arbitrarily."
            ),
            "items": {"type": "string"},
        },
        "town_evidence": {
            "type": "string",
            "description": "Exact evidence span from the address, or empty string.",
        },
        "country_evidence": {
            "type": "string",
            "description": "Exact evidence span from the address, or empty string.",
        },
        "town_is_explicit": {"type": "boolean"},
        "country_is_explicit": {"type": "boolean"},
        "town_ambiguous": {"type": "boolean"},
        "country_ambiguous": {"type": "boolean"},
        "town_model_confidence": {"type": "number"},
        "country_model_confidence": {"type": "number"},
        "town_rationale": {"type": "string"},
        "country_rationale": {"type": "string"},
        "reference_basis": {
            "type": "array",
            "description": (
                "Evidence sources actually used, e.g. input_text or a key supplied "
                "in reference_context. Never name a source the caller did not "
                "supply."
            ),
            "items": {"type": "string"},
        },
    },
    "required": list(REQUIRED_RESPONSE_FIELDS),
    "propertyOrdering": list(REQUIRED_RESPONSE_FIELDS),
}


class MalformedExtractionResponse(ValueError):
    """A model response could not be parsed or violated the response schema."""

    def __init__(self, message: str, *, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class ExtractionResponse(BaseModel):
    """A validated Gemini extraction result.

    Validation is normalizing, not forgiving: casing and whitespace are
    canonicalized, but a non-ISO country code or an out-of-range confidence is
    an error rather than something quietly coerced.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    town: str
    country_candidates: tuple[str, ...] = ()
    town_evidence: str = ""
    country_evidence: str = ""
    town_is_explicit: bool = False
    country_is_explicit: bool = False
    town_ambiguous: bool = False
    country_ambiguous: bool = False
    town_model_confidence: float = 0.0
    country_model_confidence: float = 0.0
    town_rationale: str = ""
    country_rationale: str = ""
    reference_basis: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("town", mode="before")
    @classmethod
    def _normalize_town(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        # Collapse internal whitespace so "NEW   YORK" and "NEW YORK" are one value.
        text = " ".join(text.split())
        return text.upper()

    @field_validator("country_candidates", mode="before")
    @classmethod
    def _normalize_candidates(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            # Tolerate a scalar/comma-joined string; the schema asks for a list.
            raw_items: list[str] = [part for part in value.split(",")]
        else:
            raw_items = [str(item) for item in value]

        seen: set[str] = set()
        cleaned: list[str] = []
        for item in raw_items:
            code = item.strip().upper()
            if not code:
                continue
            if code in {NO_COUNTRY, "NONE", "NULL", "N/A"}:
                # The model must express "no country" as an empty list; the
                # sentinel is the pipeline's to write, not the model's.
                continue
            if not _ALPHA2_RE.match(code):
                raise ValueError(
                    f"country_candidates entry {item!r} is not an ISO 3166-1 "
                    "alpha-2 uppercase code"
                )
            if code not in seen:
                seen.add(code)
                cleaned.append(code)
        return tuple(cleaned)

    @field_validator("reference_basis", mode="before")
    @classmethod
    def _normalize_basis(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = [value]
        return tuple(str(item).strip() for item in value if str(item).strip())

    @field_validator(
        "town_evidence", "country_evidence", "town_rationale", "country_rationale",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    @field_validator("town_model_confidence", "country_model_confidence")
    @classmethod
    def _confidence_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"model confidence {value} is outside [0, 1]")
        return float(value)

    @property
    def has_town(self) -> bool:
        """Whether the model returned a defensible town at all."""
        return bool(self.town) and self.town != NO_TOWN

    @property
    def has_country(self) -> bool:
        return bool(self.country_candidates)

    @property
    def is_country_ambiguous(self) -> bool:
        """Ambiguity as determined by the candidate set, not by the model's flag.

        More than one surviving candidate *is* unresolved ambiguity whatever
        ``country_ambiguous`` claims. The flag alone cannot create ambiguity
        out of a single candidate — that would let prompt behaviour move the
        business decision.
        """
        return len(self.country_candidates) > 1

    def to_audit_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["country_candidates"] = list(self.country_candidates)
        payload["reference_basis"] = list(self.reference_basis)
        return payload


def parse_extraction_response(payload: str | Mapping[str, Any]) -> ExtractionResponse:
    """Parse and validate a raw model response.

    Accepts a JSON string or an already-decoded mapping. Raises
    :class:`MalformedExtractionResponse` for invalid JSON, a non-object
    top-level value, missing required fields, or schema violations.
    """
    raw_text = payload if isinstance(payload, str) else None

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise MalformedExtractionResponse("model returned an empty response", raw=payload)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedExtractionResponse(
                f"model response is not valid JSON: {exc}", raw=payload
            ) from exc
    else:
        decoded = payload

    if not isinstance(decoded, Mapping):
        raise MalformedExtractionResponse(
            f"model response must be a JSON object, got {type(decoded).__name__}",
            raw=raw_text,
        )

    missing = [field for field in REQUIRED_RESPONSE_FIELDS if field not in decoded]
    if missing:
        raise MalformedExtractionResponse(
            "model response is missing required field(s): " + ", ".join(missing),
            raw=raw_text,
        )

    try:
        return ExtractionResponse.model_validate(dict(decoded))
    except ValidationError as exc:
        raise MalformedExtractionResponse(
            f"model response failed schema validation: {exc}", raw=raw_text
        ) from exc


@dataclass(frozen=True)
class PromptContract:
    """The prompt loaded from disk, plus the version it is pinned to."""

    text: str
    version: str
    source_path: Path

    @property
    def system_instruction(self) -> str:
        return self.text


@lru_cache(maxsize=8)
def _read_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Gemini prompt contract not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Gemini prompt contract is empty: {prompt_path}")
    return text


def load_prompt_contract(path: str | Path, version: str) -> PromptContract:
    """Load the prompt contract from its single on-disk source."""
    resolved = Path(path)
    return PromptContract(
        text=_read_prompt(str(resolved)),
        version=version,
        source_path=resolved,
    )


def build_user_payload(
    address: str, reference_context: Mapping[str, Any] | None = None
) -> str:
    """Render the user payload described by the prompt's template section.

    ``reference_context`` is whatever the *program* supplied. When it is empty
    the model is told so explicitly, which is what keeps rule 3 of the prompt
    enforceable: it cannot claim a SWIFTRef or ISO lookup that never happened.
    """
    return json.dumps(
        {"address": address, "reference_context": dict(reference_context or {})},
        ensure_ascii=False,
        sort_keys=True,
    )
