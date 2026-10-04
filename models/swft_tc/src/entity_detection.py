"""EXP-03 — principal organisation/entity identification for retraction protection.

A **separate** Gemini prompt and result. It never touches Town/Country
extraction: the extraction prompt, its cache, its predictions, verification,
scoring, cross-entropy and HITL are reused exactly as they were. Entity
detection controls ONE thing — which text spans retraction may not modify.

Three safeguards:

1. **Grounded spans only.** Every span the model returns is re-verified in
   deterministic Python: the named source field must be one of the group's
   configured fields and the text must occur there on whole-token boundaries
   (case-insensitive, original text preserved). Anything else is rejected,
   recorded as rejected, and protects nothing.
2. **Confidence gate.** Verified spans protect text only when
   ``entity_confidence`` is strictly greater than the configured threshold.
3. **Own cache.** Entity responses live in their own JSONL cache, keyed on the
   entity prompt version, the model and a fingerprint of the field-bounded
   payload — never in the Town/Country extraction cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .cache import AddressCache, CacheEntry
from .gemini_client import (
    DEFAULT_MALFORMED_RETRIES,
    GeminiClient,
    MalformedExtractionResponse,
    _CountingClient,
)
from .retraction import token_phrase_matches, token_spans
from .schemas import PromptContract

__all__ = [
    "ENTITY_FIELD_KEYS",
    "ENTITY_RESPONSE_JSON_SCHEMA",
    "ENTITY_TYPES",
    "NO_ENTITY",
    "EntityDetectionResult",
    "EntityDetector",
    "EntityOutcome",
    "EntityResponse",
    "GeminiEntityClient",
    "MockEntityClient",
    "RejectedSpan",
    "VerifiedSpan",
    "build_entity_client",
    "build_entity_payload",
    "entity_payload_fingerprint",
    "make_entity_cache_key",
    "parse_entity_response",
    "verify_entity_spans",
]

logger = logging.getLogger(__name__)

#: Controlled taxonomy. BANK is the category that matters most to the business.
ENTITY_TYPES: tuple[str, ...] = (
    "BANK",
    "OTHER_FINANCIAL",
    "RETAIL",
    "BIOTECH_HEALTHCARE",
    "INDUSTRIAL_ENERGY",
    "GOVERNMENT_PUBLIC",
    "OTHER",
    "NO_ENTITY",
)
NO_ENTITY = "NO_ENTITY"

#: The three flat CSV fields EXP-03 adds per group, in order. Audit depth
#: (confidence, spans, rejections) stays in the detailed JSON.
ENTITY_FIELD_KEYS: tuple[str, ...] = ("entity_name", "entity_type", "entity_rationale")

ENTITY_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity_name": {"type": "string"},
        "entity_type": {"type": "string", "enum": list(ENTITY_TYPES)},
        "entity_rationale": {"type": "string"},
        "entity_confidence": {"type": "number"},
        "entity_spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_field": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["source_field", "text"],
            },
        },
    },
    "required": ["entity_name", "entity_type", "entity_rationale", "entity_confidence", "entity_spans"],
}


# --------------------------------------------------------------------------
# Response model
# --------------------------------------------------------------------------


class EntitySpanClaim(BaseModel):
    """A span as the model returned it — unverified."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    source_field: str = ""
    text: str = ""


class EntityResponse(BaseModel):
    """The structured entity response, normalised but not yet verified."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    entity_name: str = NO_ENTITY
    entity_type: str = NO_ENTITY
    entity_rationale: str = ""
    entity_confidence: float = 0.0
    entity_spans: tuple[EntitySpanClaim, ...] = ()
    #: Set when the model's type was outside the taxonomy and was normalised.
    normalisation_notes: tuple[str, ...] = Field(default=())

    @field_validator("entity_confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number != number:  # NaN
            return 0.0
        return min(1.0, max(0.0, number))

    @property
    def is_entity(self) -> bool:
        return self.entity_type != NO_ENTITY and self.entity_name.strip() not in {"", NO_ENTITY}

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "entity_rationale": self.entity_rationale,
            "entity_confidence": self.entity_confidence,
            "entity_spans": [s.model_dump() for s in self.entity_spans],
            "normalisation_notes": list(self.normalisation_notes),
        }


def parse_entity_response(payload: str | Mapping[str, Any]) -> EntityResponse:
    """Parse and normalise a model (or cached) entity response.

    Unknown ``entity_type`` values are normalised to ``OTHER`` with a note —
    the taxonomy is closed, but a model drifting outside it should not fail
    the record. A ``NO_ENTITY`` type empties the name and spans so nothing
    can be protected by accident.
    """
    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except json.JSONDecodeError as exc:
        raise MalformedExtractionResponse(f"entity response is not JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise MalformedExtractionResponse("entity response must be a JSON object")

    notes: list[str] = []
    raw_type = str(data.get("entity_type", NO_ENTITY) or NO_ENTITY).strip().upper()
    if raw_type not in ENTITY_TYPES:
        notes.append(f"entity_type {raw_type!r} is outside the taxonomy; normalised to OTHER")
        raw_type = "OTHER"

    name = str(data.get("entity_name", "") or "").strip()
    spans_raw = data.get("entity_spans") or []
    if not isinstance(spans_raw, list):
        notes.append("entity_spans was not a list; ignored")
        spans_raw = []
    if raw_type == NO_ENTITY or name in {"", NO_ENTITY}:
        raw_type, name, spans_raw = NO_ENTITY, NO_ENTITY, []

    try:
        return EntityResponse(
            entity_name=name,
            entity_type=raw_type,
            entity_rationale=str(data.get("entity_rationale", "") or "").strip(),
            entity_confidence=data.get("entity_confidence", 0.0),
            entity_spans=tuple(
                EntitySpanClaim(source_field=str(s.get("source_field", "")), text=str(s.get("text", "")))
                for s in spans_raw if isinstance(s, Mapping)
            ),
            normalisation_notes=tuple(notes),
        )
    except ValidationError as exc:
        raise MalformedExtractionResponse(str(exc)) from exc


# --------------------------------------------------------------------------
# Payload, fingerprint, cache key
# --------------------------------------------------------------------------


def build_entity_payload(source_fields: Sequence[str], source_values: Mapping[str, Any]) -> str:
    """The field-bounded user payload: the model must see where each line ends."""
    return json.dumps(
        {
            "source_fields": [
                {"field_name": name, "value": str(source_values.get(name, "") or "").strip()}
                for name in source_fields
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def entity_payload_fingerprint(source_fields: Sequence[str], source_values: Mapping[str, Any]) -> str:
    """Deterministic identity of one field-bounded payload (names + values)."""
    return hashlib.sha256(build_entity_payload(source_fields, source_values).encode("utf-8")).hexdigest()


def make_entity_cache_key(*, prompt_version: str, model: str, fingerprint: str) -> str:
    return hashlib.sha256("|".join(("entity", prompt_version, model, fingerprint)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Span verification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedSpan:
    """A model-claimed span that really occurs in the stated field."""

    source_field: str
    text: str          # exactly as written in the source field
    start: int         # character offsets into the ORIGINAL field value
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"source_field": self.source_field, "text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class RejectedSpan:
    source_field: str
    text: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_field": self.source_field, "text": self.text, "reason": self.reason}


REJECT_UNKNOWN_FIELD = "unknown_source_field"
REJECT_EMPTY = "empty_text"
REJECT_NOT_FOUND = "text_not_found_on_token_boundaries"


def verify_entity_spans(
    response: EntityResponse,
    source_values: Mapping[str, Any],
    source_fields: Sequence[str],
) -> tuple[tuple[VerifiedSpan, ...], tuple[RejectedSpan, ...]]:
    """Re-verify every claimed span deterministically. Never trust the model.

    A span verifies when its field is configured and its text occurs in that
    field on whole-token boundaries (the pipeline's own matcher, so
    ``BOSTONIAN`` never verifies ``BOSTON``). Matching is case-insensitive;
    the recorded text and offsets are the field's original characters. When
    the phrase occurs more than once in the field, the first occurrence is the
    one protected — the model named a phrase, not a position.
    """
    verified: list[VerifiedSpan] = []
    rejected: list[RejectedSpan] = []
    configured = list(source_fields)
    for claim in response.entity_spans:
        field_name, text = claim.source_field.strip(), claim.text.strip()
        if field_name not in configured:
            rejected.append(RejectedSpan(field_name, text, REJECT_UNKNOWN_FIELD)); continue
        if not text:
            rejected.append(RejectedSpan(field_name, text, REJECT_EMPTY)); continue
        original = str(source_values.get(field_name, "") or "")
        matches = token_phrase_matches(original, text)
        if not matches:
            rejected.append(RejectedSpan(field_name, text, REJECT_NOT_FOUND)); continue
        spans = token_spans(original)
        first_tok, last_tok = matches[0]
        start, end = spans[first_tok].start, spans[last_tok].end
        verified.append(VerifiedSpan(field_name, original[start:end], start, end))
    return tuple(verified), tuple(rejected)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityDetectionResult:
    """One group instance's entity decision, verified and ready for retraction."""

    entity_name: str = NO_ENTITY
    entity_type: str = NO_ENTITY
    entity_rationale: str = ""
    entity_confidence: float = 0.0
    returned_spans: tuple[dict[str, str], ...] = ()
    verified_spans: tuple[VerifiedSpan, ...] = ()
    rejected_spans: tuple[RejectedSpan, ...] = ()
    prompt_version: str = ""
    model: str = ""
    protection_confidence_threshold: float | None = None
    cache_hit: bool = False
    status: str = "not_run"      # not_run | detected | no_entity | error
    error: str = ""
    normalisation_notes: tuple[str, ...] = ()

    @property
    def is_entity(self) -> bool:
        return self.entity_type != NO_ENTITY and self.entity_name not in {"", NO_ENTITY}

    @property
    def protection_applies(self) -> bool:
        """Strict gate: verified spans protect only when confidence > threshold."""
        if not self.is_entity or not self.verified_spans:
            return False
        if self.protection_confidence_threshold is None:
            return True
        return float(self.entity_confidence) > float(self.protection_confidence_threshold)

    def protected_spans_by_field(self) -> dict[str, tuple[tuple[int, int], ...]]:
        if not self.protection_applies:
            return {}
        out: dict[str, list[tuple[int, int]]] = {}
        for span in self.verified_spans:
            out.setdefault(span.source_field, []).append((span.start, span.end))
        return {k: tuple(v) for k, v in out.items()}

    def flat_values(self) -> list[str]:
        """The three CSV fields, in ENTITY_FIELD_KEYS order."""
        return [self.entity_name, self.entity_type, self.entity_rationale]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "entity_rationale": self.entity_rationale,
            "entity_confidence": self.entity_confidence,
            "protection_confidence_threshold": self.protection_confidence_threshold,
            "protection_applies": self.protection_applies,
            "returned_spans": [dict(s) for s in self.returned_spans],
            "verified_spans": [s.to_dict() for s in self.verified_spans],
            "rejected_spans": [s.to_dict() for s in self.rejected_spans],
            "prompt_version": self.prompt_version,
            "model": self.model,
            "cache_hit": self.cache_hit,
            "error": self.error,
            "normalisation_notes": list(self.normalisation_notes),
        }

    @classmethod
    def from_response(
        cls,
        response: EntityResponse,
        *,
        source_fields: Sequence[str],
        source_values: Mapping[str, Any],
        prompt_version: str,
        model: str,
        threshold: float | None,
        cache_hit: bool,
    ) -> "EntityDetectionResult":
        verified, rejected = verify_entity_spans(response, source_values, source_fields)
        return cls(
            entity_name=response.entity_name,
            entity_type=response.entity_type,
            entity_rationale=response.entity_rationale,
            entity_confidence=float(response.entity_confidence),
            returned_spans=tuple(s.model_dump() for s in response.entity_spans),
            verified_spans=verified,
            rejected_spans=rejected,
            prompt_version=prompt_version,
            model=model,
            protection_confidence_threshold=threshold,
            cache_hit=cache_hit,
            status="detected" if response.is_entity else "no_entity",
            normalisation_notes=response.normalisation_notes,
        )


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityOutcome:
    response: EntityResponse
    model: str
    attempts: int = 1
    usage: Mapping[str, Any] = field(default_factory=dict)


class GeminiEntityClient(GeminiClient):
    """The extraction client's transport and retry machinery, bound to the entity prompt and schema."""

    def __init__(self, *, model: str, prompt: PromptContract, **kwargs: Any) -> None:
        super().__init__(model=model, prompt=prompt, **kwargs)
        self._response_schema = ENTITY_RESPONSE_JSON_SCHEMA

    def identify(self, payload: str) -> EntityOutcome:
        attempts = 0
        last_malformed: MalformedExtractionResponse | None = None
        for malformed_attempt in range(self._malformed_retries + 1):
            raw_text, usage, transport_attempts = self._call_with_retry(payload)
            attempts += transport_attempts
            try:
                response = parse_entity_response(raw_text)
            except MalformedExtractionResponse as exc:
                last_malformed = exc
                logger.warning("malformed entity response (attempt %d/%d): %s",
                               malformed_attempt + 1, self._malformed_retries + 1, exc)
                continue
            return EntityOutcome(response=response, model=self.model, attempts=attempts, usage=usage)
        assert last_malformed is not None
        raise last_malformed


_STUB_BANK_TOKENS = ("BANK", "BANCO", "BANQUE", "CITIBANK", "BK")
_STUB_ORG_TOKENS = ("LTD", "LIMITED", "LLC", "INC", "PLC", "CORP", "CORPORATION", "SA", "SAE", "AG",
                    "NV", "TRUST", "SECURITIES", "CAPITAL", "INVESTMENT", "AUTHORITY", "INDUSTRIES",
                    "SERVICES", "COMPANY", "GROUP", "HO", "BRANCH")


class MockEntityClient(_CountingClient):
    """Offline stub for dry runs. **Not an entity model.**

    Picks the first field carrying an organisation-like token and returns that
    whole field as the entity span, BANK if a bank token is present, OTHER
    otherwise, NO_ENTITY when nothing matches. Exists only so the plumbing —
    cache, verification, retraction protection, columns — can be exercised
    without a model. Every rationale it emits says so.
    """

    def __init__(self, *, model: str = "mock-entity-dry-run") -> None:
        super().__init__()
        self.model = model

    def identify(self, payload: str) -> EntityOutcome:
        self._record_call()
        data = json.loads(payload)
        for item in data.get("source_fields", []):
            text = str(item.get("value", "")).strip()
            if not text:
                continue
            is_bank = any(token_phrase_matches(text, t) for t in _STUB_BANK_TOKENS)
            is_org = is_bank or any(token_phrase_matches(text, t) for t in _STUB_ORG_TOKENS)
            if is_org:
                response = EntityResponse(
                    entity_name=text,
                    entity_type="BANK" if is_bank else "OTHER",
                    entity_rationale=f"[offline stub, not a model] {item['field_name']} carries an organisation-like token.",
                    entity_confidence=0.95,
                    entity_spans=(EntitySpanClaim(source_field=item["field_name"], text=text),),
                )
                return EntityOutcome(response=response, model=self.model)
        return EntityOutcome(
            response=EntityResponse(entity_rationale="[offline stub, not a model] no organisation-like token found."),
            model=self.model,
        )


def build_entity_client(model_config: Any, prompt: PromptContract, *, model: str, dry_run: bool = False,
                        max_output_tokens: int | None = None) -> Any:
    """Construct the entity client. ``dry_run`` yields the offline stub."""
    if dry_run:
        return MockEntityClient()
    return GeminiEntityClient(
        model=model,
        prompt=prompt,
        temperature=model_config.temperature,
        max_output_tokens=max_output_tokens or model_config.max_output_tokens,
        max_retries=model_config.max_retries,
        request_timeout_seconds=model_config.request_timeout_seconds,
        retry_initial_seconds=model_config.retry_initial_seconds,
        retry_max_seconds=model_config.retry_max_seconds,
        retry_jitter_seconds=model_config.retry_jitter_seconds,
        enable_google_search_grounding=model_config.enable_google_search_grounding,
    )


# --------------------------------------------------------------------------
# Detector: client + own cache
# --------------------------------------------------------------------------


class EntityDetector:
    """Runs entity identification once per unique field-bounded payload, with its own cache."""

    def __init__(self, *, client: Any, cache: AddressCache, prompt_version: str,
                 protection_confidence_threshold: float | None) -> None:
        self.client = client
        self.cache = cache
        self.prompt_version = prompt_version
        self.threshold = protection_confidence_threshold
        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entity_cache_hits": self._hits, "entity_cache_misses": self._misses,
                    "entity_errors": self._errors, "entity_backend_calls": self.client.call_count}

    def detect(self, source_fields: Sequence[str], source_values: Mapping[str, Any]) -> EntityDetectionResult:
        payload = build_entity_payload(source_fields, source_values)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        key = make_entity_cache_key(prompt_version=self.prompt_version, model=self.client.model,
                                    fingerprint=fingerprint)
        cached = self.cache.get(key)
        if cached is not None:
            try:
                response = parse_entity_response(cached.response)
                with self._lock:
                    self._hits += 1
                return EntityDetectionResult.from_response(
                    response, source_fields=source_fields, source_values=source_values,
                    prompt_version=self.prompt_version, model=cached.model or self.client.model,
                    threshold=self.threshold, cache_hit=True)
            except MalformedExtractionResponse:
                logger.warning("discarding unreadable entity cache entry %s", key[:12])
        with self._lock:
            self._misses += 1
        try:
            outcome = self.client.identify(payload)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal to the record
            with self._lock:
                self._errors += 1
            logger.error("entity identification failed for payload %s: %s", fingerprint[:12], type(exc).__name__)
            return EntityDetectionResult(prompt_version=self.prompt_version, model=self.client.model,
                                         protection_confidence_threshold=self.threshold,
                                         status="error", error=f"{type(exc).__name__}: {exc}")
        self.cache.put(CacheEntry(
            key=key, address_hash=fingerprint, address=payload, prompt_version=self.prompt_version,
            model=outcome.model, reference_context_version="",
            response=outcome.response.to_audit_dict(),
            metadata={"attempts": outcome.attempts, "usage": dict(outcome.usage), "kind": "entity"},
        ))
        return EntityDetectionResult.from_response(
            outcome.response, source_fields=source_fields, source_values=source_values,
            prompt_version=self.prompt_version, model=outcome.model, threshold=self.threshold, cache_hit=False)
