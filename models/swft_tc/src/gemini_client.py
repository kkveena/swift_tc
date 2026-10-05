"""Extraction clients: the Gemini implementation, plus offline doubles.

The pipeline depends on the :class:`AddressExtractionClient` protocol, never on
the Google SDK directly, so dry runs and tests substitute a double without any
pipeline change.

Retry policy is deliberately asymmetric. Transient transport failures (429,
5xx, timeouts) are retried with bounded exponential backoff and jitter.
Malformed *business* output is retried at most once and then recorded as an
error — a model that keeps returning an invalid country code will not be
retried until the quota is gone.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .reference_data import ReferenceContext
from .schemas import (
    RESPONSE_JSON_SCHEMA,
    ExtractionResponse,
    MalformedExtractionResponse,
    PromptContract,
    build_user_payload,
    parse_extraction_response,
)

__all__ = [
    "AddressExtractionClient",
    "ExtractionError",
    "ExtractionOutcome",
    "GeminiClient",
    "GenerationDiagnostics",
    "TruncatedExtractionResponse",
    "build_client",
    "MockExtractionClient",
    "PermanentExtractionError",
    "ScriptedExtractionClient",
    "TransientExtractionError",
    "is_transient_error",
]

logger = logging.getLogger(__name__)

#: One re-ask for malformed structured output, then stop. Structured output can
#: truncate; a schema-violating model does not improve with repetition.
DEFAULT_MALFORMED_RETRIES = 1

#: Status codes and API status strings treated as retryable transport failures.
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_MARKERS = (
    "resource_exhausted",
    "unavailable",
    "deadline_exceeded",
    "internal error",
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
)


class ExtractionError(RuntimeError):
    """Base class for extraction failures."""


class TransientExtractionError(ExtractionError):
    """A retryable transport-level failure (429, 5xx, network, timeout)."""


class PermanentExtractionError(ExtractionError):
    """A failure that will not be fixed by retrying (auth, bad request, schema)."""


class TruncatedExtractionResponse(MalformedExtractionResponse):
    """The model stopped on ``MAX_TOKENS`` before closing its JSON object.

    Distinct from an ordinary malformed response because the cause is the
    request budget, not the model's answer: at temperature 0 an identical
    re-ask truncates identically, so the malformed retry is skipped and the
    error carries the finish reason and token usage instead.
    """


#: Finish reason reported by the SDK when ``max_output_tokens`` was exhausted.
FINISH_REASON_MAX_TOKENS = "MAX_TOKENS"


@dataclass(frozen=True)
class GenerationDiagnostics:
    """Safe-to-log facts about one model response. Carries NO address text.

    Captured on every call and logged when the response fails to parse. The
    payload is identified by a hash only; the response body is described by
    its length and whether it opens and closes a JSON object.
    """

    payload_sha256: str
    attempt: int
    finish_reason: str | None
    response_chars: int
    starts_with_object_brace: bool
    ends_with_object_brace: bool
    usage: Mapping[str, Any]
    request: Mapping[str, Any]

    @property
    def truncated(self) -> bool:
        return self.finish_reason == FINISH_REASON_MAX_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_sha256": self.payload_sha256,
            "attempt": self.attempt,
            "finish_reason": self.finish_reason,
            "response_chars": self.response_chars,
            "starts_with_object_brace": self.starts_with_object_brace,
            "ends_with_object_brace": self.ends_with_object_brace,
            "usage": dict(self.usage),
            "request": dict(self.request),
        }


@dataclass(frozen=True)
class _Generation:
    """One raw model answer plus its diagnostics."""

    text: str
    diagnostics: GenerationDiagnostics

    @property
    def usage(self) -> Mapping[str, Any]:
        return self.diagnostics.usage


@dataclass(frozen=True)
class ExtractionOutcome:
    """A successful extraction plus the metadata worth auditing."""

    response: ExtractionResponse
    model: str
    attempts: int = 1
    usage: Mapping[str, Any] = field(default_factory=dict)
    reference_sources: tuple[str, ...] = ()


@runtime_checkable
class AddressExtractionClient(Protocol):
    """What the pipeline needs from an extraction backend."""

    model: str

    def extract(
        self, address: str, reference_context: ReferenceContext
    ) -> ExtractionOutcome:
        """Extract Town/Country for one cleaned address."""
        ...

    @property
    def call_count(self) -> int:
        """Backend calls actually issued. Asserted on by the null-skip test."""
        ...


class _CountingClient:
    """Shared call accounting for the concrete clients below."""

    def __init__(self) -> None:
        self._call_count = 0
        self._count_lock = threading.Lock()

    def _record_call(self) -> None:
        with self._count_lock:
            self._call_count += 1

    @property
    def call_count(self) -> int:
        with self._count_lock:
            return self._call_count


class GeminiClient(_CountingClient):
    """Structured-output extraction via the Google Gen AI SDK.

    The SDK client is constructed with no credentials in source: it reads
    ``GEMINI_API_KEY``/``GOOGLE_API_KEY``, or Vertex AI settings
    (``GOOGLE_GENAI_USE_VERTEXAI``, ``GOOGLE_CLOUD_PROJECT``,
    ``GOOGLE_CLOUD_LOCATION``) from the environment. Switching between the
    Developer API and enterprise Vertex AI is therefore an environment change,
    not a code change. No credential value is read, stored, or logged here.
    """

    def __init__(
        self,
        *,
        model: str,
        prompt: PromptContract,
        temperature: float = 0.0,
        max_output_tokens: int = 700,
        max_retries: int = 5,
        request_timeout_seconds: float = 60.0,
        retry_initial_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        retry_jitter_seconds: float = 0.5,
        malformed_retries: int = DEFAULT_MALFORMED_RETRIES,
        enable_google_search_grounding: bool = False,
        disable_automatic_function_calling: bool = False,
        client: Any = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._prompt = prompt
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries
        self._timeout_seconds = request_timeout_seconds
        self._retry_initial = retry_initial_seconds
        self._retry_max = retry_max_seconds
        self._retry_jitter = retry_jitter_seconds
        self._malformed_retries = max(0, malformed_retries)

        if enable_google_search_grounding:
            # Public web search is not a substitute for licensed enterprise
            # reference data, and this pipeline may carry customer/payment
            # addresses. Enabling it is a data-governance decision.
            raise PermanentExtractionError(
                "Google Search grounding is disabled for this pipeline. Customer "
                "and payment address data must not be sent to public web search "
                "without explicit data-governance approval."
            )

        #: The structured-output schema this client binds to. Subclasses bound
        #: to a different prompt (entity identification) replace it.
        self._response_schema: Any = RESPONSE_JSON_SCHEMA
        #: Short label for the bound schema, used in diagnostics only.
        self._schema_label: str = "town_country"
        #: When set, the SDK's client-side automatic-function-calling loop is
        #: switched off explicitly. This pipeline declares no tools, so the
        #: flag changes nothing on the wire (the SDK never sends it); it only
        #: stops the SDK's unconditional "AFC is enabled" log line and makes
        #: the no-tools intent visible in the request config.
        self._disable_afc = bool(disable_automatic_function_calling)
        self._client = client if client is not None else self._build_client()
        (
            self._config_cls,
            self._http_options_cls,
            self._afc_config_cls,
        ) = self._load_config_types()

    @staticmethod
    def _build_client() -> Any:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on install
            raise PermanentExtractionError(
                "google-genai is not installed. Install it (pip install "
                "google-genai) or run the pipeline in dry-run/mock mode."
            ) from exc
        # No arguments: every credential and endpoint setting comes from the
        # environment so the same code serves Developer API and Vertex AI.
        return genai.Client()

    @staticmethod
    def _load_config_types() -> tuple[Any, Any, Any]:
        try:
            from google.genai import types
        except ImportError:  # pragma: no cover - depends on install
            return None, None, None
        return (
            types.GenerateContentConfig,
            types.HttpOptions,
            types.AutomaticFunctionCallingConfig,
        )

    # -- extraction --------------------------------------------------------

    def extract(
        self, address: str, reference_context: ReferenceContext
    ) -> ExtractionOutcome:
        payload = build_user_payload(address, reference_context.to_prompt_dict())
        attempts = 0
        last_malformed: MalformedExtractionResponse | None = None

        for malformed_attempt in range(self._malformed_retries + 1):
            try:
                generation, transport_attempts = self._call_with_retry(
                    payload, attempt=malformed_attempt + 1
                )
                attempts += transport_attempts
                response = self._parse_generation(generation, parse_extraction_response)
            except TruncatedExtractionResponse as exc:
                last_malformed = exc
                break  # same budget, same truncation: do not re-ask
            except MalformedExtractionResponse as exc:
                last_malformed = exc
                continue
            return ExtractionOutcome(
                response=response,
                model=self.model,
                attempts=attempts,
                usage=generation.usage,
                reference_sources=reference_context.sources,
            )

        assert last_malformed is not None  # loop always sets it before exhausting
        raise last_malformed

    def _parse_generation(self, generation: _Generation, parser: Any) -> Any:
        """Parse one answer; on failure log the safe diagnostics and re-raise.

        A ``MAX_TOKENS`` finish is reported as :class:`TruncatedExtractionResponse`
        whether or not the partial text happens to parse, because a truncated
        structured answer is never a complete one.
        """
        diag = generation.diagnostics
        if diag.truncated:
            logger.warning(
                "%s response truncated at max_output_tokens; diagnostics=%s",
                self._schema_label, diag.to_dict(),
            )
            raise TruncatedExtractionResponse(
                f"{self._schema_label} response hit finish_reason={diag.finish_reason} "
                f"(max_output_tokens={self._max_output_tokens}); "
                f"response_chars={diag.response_chars}, usage={dict(diag.usage)}"
            )
        try:
            return parser(generation.text)
        except MalformedExtractionResponse as exc:
            logger.warning(
                "malformed %s response (attempt %d/%d): %s; diagnostics=%s",
                self._schema_label, diag.attempt, self._malformed_retries + 1,
                exc, diag.to_dict(),
            )
            raise

    def _call_with_retry(self, payload: str, *, attempt: int = 1) -> tuple[_Generation, int]:
        attempts = 0

        def _attempt() -> _Generation:
            nonlocal attempts
            attempts += 1
            self._record_call()
            return self._generate(payload, attempt=attempt)

        retrying = Retrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential_jitter(
                initial=self._retry_initial,
                max=self._retry_max,
                jitter=self._retry_jitter,
            ),
            retry=retry_if_exception_type(TransientExtractionError),
            reraise=True,
        )
        try:
            generation = retrying(_attempt)
        except RetryError as exc:  # pragma: no cover - reraise=True covers this
            raise TransientExtractionError(str(exc)) from exc
        return generation, attempts

    def _generate(self, payload: str, *, attempt: int = 1) -> _Generation:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=payload,
                config=self._request_config(),
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises provider-specific types
            if is_transient_error(exc):
                raise TransientExtractionError(f"{type(exc).__name__}: {exc}") from exc
            raise PermanentExtractionError(f"{type(exc).__name__}: {exc}") from exc

        text = getattr(response, "text", None)
        finish_reason = _finish_reason(response)
        usage = _usage_dict(getattr(response, "usage_metadata", None))
        stripped = (text or "").strip()
        diagnostics = GenerationDiagnostics(
            payload_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            attempt=attempt,
            finish_reason=finish_reason,
            response_chars=len(text or ""),
            starts_with_object_brace=stripped.startswith("{"),
            ends_with_object_brace=stripped.endswith("}"),
            usage=usage,
            request=self.request_config_summary(),
        )
        if not text:
            if finish_reason == FINISH_REASON_MAX_TOKENS:
                # Nothing but thoughts fitted in the budget. Retrying the same
                # request cannot help; report it as truncation instead.
                logger.warning(
                    "%s response truncated at max_output_tokens with no text; "
                    "diagnostics=%s", self._schema_label, diagnostics.to_dict(),
                )
                raise TruncatedExtractionResponse(
                    f"{self._schema_label} response hit finish_reason={finish_reason} "
                    f"with no text (max_output_tokens={self._max_output_tokens}); "
                    f"usage={usage}"
                )
            # An empty body with no exception is treated as transient: it is
            # usually a filtered generation, not a business answer.
            raise TransientExtractionError("model returned an empty response body")
        return _Generation(text=text, diagnostics=diagnostics)

    def request_config_summary(self) -> dict[str, Any]:
        """The request settings worth auditing, with no prompt or payload text."""
        return {
            "model": self.model,
            "response_mime_type": "application/json",
            "response_schema": self._schema_label,
            "max_output_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "tools": None,
            "automatic_function_calling_disabled": self._disable_afc,
        }

    def _request_config(self) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": self._prompt.system_instruction,
            # Deterministic extraction: no sampling temperature, JSON only.
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": self._response_schema,
        }
        if self._config_cls is None:  # pragma: no cover - depends on install
            if self._disable_afc:
                kwargs["automatic_function_calling"] = {"disable": True}
            return kwargs
        if self._http_options_cls is not None:
            kwargs["http_options"] = self._http_options_cls(
                timeout=int(self._timeout_seconds * 1000)
            )
        if self._disable_afc and self._afc_config_cls is not None:
            # No tools are declared, so this never changes the request body;
            # it only turns off the SDK's client-side AFC loop and its log.
            kwargs["automatic_function_calling"] = self._afc_config_cls(disable=True)
        return self._config_cls(**kwargs)


class MockExtractionClient(_CountingClient):
    """Offline stub for dry runs. **Not an extraction model.**

    It exists so the notebook and the tests can exercise the full pipeline —
    grouping, dedupe, verification, scoring, output — with no credentials and
    no network. Its "predictions" come from two deterministic sources only:

    * Country: ISO codes/aliases the program can actually find in the address.
    * Town: a tiny demo gazetteer, below, covering the documented sample
      addresses. Anything outside it returns ``NO_TOWN``.

    Every rationale it emits says so, and run metrics record ``mode:
    dry_run``, so mock output can never be mistaken for a model conclusion.
    """

    #: Demo-only. Deliberately tiny; this is not a gazetteer product.
    DEMO_TOWNS: tuple[str, ...] = (
        "NEW YORK", "BOSTON", "LIMA", "ACCRA", "AUCKLAND", "TAIPEI",
        "LONDON", "SINGAPORE", "HONG KONG", "TOKYO", "SYDNEY", "TORONTO",
        "PARIS", "FRANKFURT", "ZURICH", "DUBLIN", "MUMBAI", "DUBAI",
    )

    #: Demo-only country inference for towns with no explicit country code.
    #: A town mapping to several countries stays ambiguous on purpose.
    DEMO_TOWN_COUNTRIES: Mapping[str, tuple[str, ...]] = {
        "BOSTON": ("US",),
        "NEW YORK": ("US",),
        "LIMA": ("PE",),
        "ACCRA": ("GH",),
        "AUCKLAND": ("NZ",),
        "TAIPEI": ("TW",),
        "LONDON": ("GB", "CA"),  # London GB / London ON — deliberately ambiguous
        "SINGAPORE": ("SG",),
        "HONG KONG": ("HK",),
        "TOKYO": ("JP",),
        "SYDNEY": ("AU",),
        "TORONTO": ("CA",),
        "PARIS": ("FR",),
        "FRANKFURT": ("DE",),
        "ZURICH": ("CH",),
        "DUBLIN": ("IE",),
        "MUMBAI": ("IN",),
        "DUBAI": ("AE",),
    }

    _MOCK_NOTE = "Offline dry-run stub, not a model conclusion."

    def __init__(self, *, model: str = "mock-dry-run", iso_provider: Any = None) -> None:
        super().__init__()
        self.model = model
        self._iso = iso_provider

    def extract(
        self, address: str, reference_context: ReferenceContext
    ) -> ExtractionOutcome:
        from .cleaning import contains_token_phrase

        self._record_call()

        town = next(
            (name for name in self.DEMO_TOWNS if contains_token_phrase(address, name)),
            "",
        )
        town_explicit = bool(town)

        explicit_codes: tuple[str, ...] = ()
        if self._iso is not None:
            payload = reference_context.payload.get("iso3166_codes_present_in_address")
            if payload:
                explicit_codes = tuple(payload)

        if explicit_codes:
            candidates = explicit_codes
            country_explicit = True
        elif town:
            candidates = self.DEMO_TOWN_COUNTRIES.get(town, ())
            country_explicit = False
        else:
            candidates = ()
            country_explicit = False

        response = parse_extraction_response(
            {
                "town": town or "NO_TOWN",
                "country_candidates": list(candidates),
                "town_evidence": town,
                "country_evidence": " ".join(candidates) if country_explicit else "",
                "town_is_explicit": town_explicit,
                "country_is_explicit": country_explicit,
                "town_ambiguous": False,
                "country_ambiguous": len(candidates) > 1,
                # Fixed placeholder confidences: a stub has no calibrated belief,
                # and pretending otherwise would corrupt the scoring demo.
                "town_model_confidence": 0.98 if town else 0.0,
                "country_model_confidence": (
                    0.99 if country_explicit else (0.95 if candidates else 0.0)
                ),
                "town_rationale": (
                    f"{self._MOCK_NOTE} Town matched the demo gazetteer on token "
                    "boundaries."
                    if town
                    else f"{self._MOCK_NOTE} No demo-gazetteer town matched."
                ),
                "country_rationale": (
                    f"{self._MOCK_NOTE} ISO code found in the address text."
                    if country_explicit
                    else (
                        f"{self._MOCK_NOTE} Country inferred from the demo town map."
                        if candidates
                        else f"{self._MOCK_NOTE} No defensible country."
                    )
                ),
                "reference_basis": ["mock_offline_stub"],
            }
        )
        return ExtractionOutcome(
            response=response,
            model=self.model,
            attempts=1,
            usage={},
            reference_sources=reference_context.sources,
        )


class ScriptedExtractionClient(_CountingClient):
    """Test double driven by a script of responses/exceptions per address.

    Each address maps to a list consumed one entry per call: a mapping becomes
    a validated response, an exception instance is raised. This is what the
    retry test uses to serve a 429 and then a success.
    """

    def __init__(
        self,
        script: Mapping[str, Any],
        *,
        model: str = "scripted-test-client",
        default: Any = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._script = {key: list(value) for key, value in script.items()}
        self._default = default
        self.seen_addresses: list[str] = []

    def extract(
        self, address: str, reference_context: ReferenceContext
    ) -> ExtractionOutcome:
        self._record_call()
        self.seen_addresses.append(address)

        queue = self._script.get(address)
        if queue:
            item = queue.pop(0)
        elif self._default is not None:
            item = self._default
        else:
            raise PermanentExtractionError(f"no scripted response for {address!r}")

        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(address)

        return ExtractionOutcome(
            response=parse_extraction_response(item),
            model=self.model,
            attempts=1,
            usage={},
            reference_sources=reference_context.sources,
        )


def is_transient_error(exc: BaseException) -> bool:
    """Classify an SDK/transport exception as retryable.

    Checks structured status codes first, then falls back to the API status
    strings and network phrasings that carry the same meaning.
    """
    if isinstance(exc, TransientExtractionError):
        return True
    if isinstance(exc, PermanentExtractionError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    for attribute in ("status_code", "code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and value in _TRANSIENT_STATUS_CODES:
            return True

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS_CODES:
        return True

    message = str(exc).lower()
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return True
    return any(str(code) in message for code in _TRANSIENT_STATUS_CODES if code >= 429)


def build_client(
    config: Any,
    prompt: PromptContract,
    *,
    model: str,
    iso_provider: Any = None,
    dry_run: bool = False,
) -> AddressExtractionClient:
    """Construct the configured extraction client.

    ``dry_run`` (or absent credentials, decided by the caller) yields the
    offline stub; otherwise the real Gemini client is built.
    """
    if dry_run:
        return MockExtractionClient(iso_provider=iso_provider)
    return GeminiClient(
        model=model,
        prompt=prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        max_retries=config.max_retries,
        request_timeout_seconds=config.request_timeout_seconds,
        retry_initial_seconds=config.retry_initial_seconds,
        retry_max_seconds=config.retry_max_seconds,
        retry_jitter_seconds=config.retry_jitter_seconds,
        enable_google_search_grounding=config.enable_google_search_grounding,
    )


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    fields = (
        "prompt_token_count",
        "candidates_token_count",
        # Thinking models bill reasoning tokens against max_output_tokens;
        # recording them is how a MAX_TOKENS finish is explained.
        "thoughts_token_count",
        "total_token_count",
        "cached_content_token_count",
    )
    return {
        name: getattr(usage, name)
        for name in fields
        if getattr(usage, name, None) is not None
    }


def _finish_reason(response: Any) -> str | None:
    """The first candidate's finish reason as a plain string, or ``None``."""
    candidates = getattr(response, "candidates", None) or ()
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is None:
            return None
        return str(getattr(reason, "name", None) or getattr(reason, "value", None) or reason)
    return None
