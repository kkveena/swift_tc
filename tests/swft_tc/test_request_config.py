"""The ACTUAL request config each Gemini client sends, and what happens when
the answer comes back truncated.

These tests render ``GeminiClient._request_config()`` and
``GeminiEntityClient._request_config()`` with the real google-genai types
(no network), so they pin what the SDK would transmit rather than what a
mock pretends to receive:

* both clients request ``application/json``;
* the entity client binds the ENTITY schema, the Town/Country client the
  Town/Country schema, and neither leaks into the other;
* the Town/Country request is unchanged by EXP-03 (budget, schema, no AFC
  override, no tools, no thinking override);
* the entity request declares no tools and switches the SDK's client-side
  automatic-function-calling loop off;
* a ``MAX_TOKENS`` finish is reported as a truncation with safe diagnostics
  (payload hash, finish reason, lengths, token usage) and is NOT re-asked,
  while an ordinary malformed answer still gets its one re-ask;
* no diagnostic line carries address text.
"""

from __future__ import annotations

import json
import logging

import pytest

from models.swft_tc.src.entity_detection import (
    ENTITY_RESPONSE_JSON_SCHEMA,
    GeminiEntityClient,
    build_entity_client,
    build_entity_payload,
)
from models.swft_tc.src.gemini_client import (
    GeminiClient,
    MalformedExtractionResponse,
    TruncatedExtractionResponse,
    _usage_dict,
)
from models.swft_tc.src.schemas import RESPONSE_JSON_SCHEMA
from models.swft_tc.src.settings import EntityDetectionConfig, ModelConfig, load_config

types = pytest.importorskip("google.genai.types")

ADDRESS_LINE = "FUBON BANK HONG KONG LTD 38 DES VOEUX ROAD CENTRAL"
FIELDS = ("address_line_1", "address_line_2", "address_line_3", "address_line_4")
VALID_ENTITY = {
    "entity_name": "FUBON BANK HONG KONG LTD",
    "entity_type": "BANK",
    "entity_rationale": "Named bank on line 1.",
    "entity_confidence": 0.97,
    "entity_spans": [{"source_field": "address_line_1", "text": "FUBON BANK HONG KONG LTD"}],
}


class _Candidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, text, finish_reason, usage=None):
        self.text = text
        self.candidates = [_Candidate(finish_reason)]
        self.usage_metadata = usage


class _FakeSdk:
    """Stand-in for ``genai.Client``: records every config and replays canned answers."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.configs = []
        self.models = self

    def generate_content(self, *, model, contents, config):  # noqa: ARG002
        self.configs.append(config)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _tc_client(prompt_contract, sdk=None, **kwargs) -> GeminiClient:
    return GeminiClient(model="gemini-3.5-flash", prompt=prompt_contract,
                        client=sdk or _FakeSdk([_Response("{}", types.FinishReason.STOP)]),
                        retry_initial_seconds=0.001, retry_max_seconds=0.002,
                        retry_jitter_seconds=0.0, **kwargs)


def _entity_client(prompt_contract, sdk=None, **kwargs) -> GeminiEntityClient:
    return GeminiEntityClient(model="gemini-3.5-flash", prompt=prompt_contract,
                              client=sdk or _FakeSdk([_Response("{}", types.FinishReason.STOP)]),
                              retry_initial_seconds=0.001, retry_max_seconds=0.002,
                              retry_jitter_seconds=0.0, **kwargs)


@pytest.fixture
def entity_prompt(model_root):
    from models.swft_tc.src.schemas import load_prompt_contract

    return load_prompt_contract(model_root / "prompts" / "GEMINI_ENTITY_PROMPT_EXP03.md",
                                "exp03-entity-identification-v1")


# --------------------------------------------------------------------------
# What is actually sent
# --------------------------------------------------------------------------


class TestEntityRequestConfig:
    def test_entity_request_uses_application_json(self, entity_prompt):
        cfg = _entity_client(entity_prompt)._request_config()
        assert isinstance(cfg, types.GenerateContentConfig)
        assert cfg.response_mime_type == "application/json"

    def test_entity_request_binds_the_entity_schema_not_town_country(self, entity_prompt):
        cfg = _entity_client(entity_prompt)._request_config()
        assert cfg.response_schema is not None
        required = set(cfg.response_schema["required"])
        assert required == set(ENTITY_RESPONSE_JSON_SCHEMA["required"])
        assert "entity_spans" in required and "entity_confidence" in required
        assert not required & set(RESPONSE_JSON_SCHEMA["required"])
        assert "town" not in cfg.response_schema["properties"]

    def test_entity_request_declares_no_tools_and_disables_afc(self, entity_prompt):
        cfg = _entity_client(entity_prompt)._request_config()
        assert cfg.tools is None
        assert cfg.tool_config is None
        assert cfg.automatic_function_calling is not None
        assert cfg.automatic_function_calling.disable is True
        # Only `disable` is set explicitly: the SDK warns when a caller also
        # sets max_remote_calls (its own default of 10 is not "set").
        assert cfg.automatic_function_calling.model_fields_set == {"disable"}

    def test_sdk_treats_entity_config_as_afc_disabled(self, entity_prompt):
        from google.genai import _extra_utils

        assert _extra_utils.should_disable_afc(_entity_client(entity_prompt)._request_config()) is True

    def test_afc_setting_is_client_side_only_and_never_sent(self, entity_prompt):
        """The flag must not change the request body: the SDK has no wire mapping for it."""
        from google.genai import models as sdk_models

        class _VertexApi:  # the transformer only reads this attribute
            vertexai = True

        cfg = _entity_client(entity_prompt)._request_config()
        parent: dict = {}
        body = sdk_models._GenerateContentConfig_to_vertex(_VertexApi(), cfg, parent)
        rendered = json.dumps(body, default=str) + json.dumps(parent, default=str)
        assert "automatic_function_calling" not in rendered
        assert "automaticFunctionCalling" not in rendered
        assert body["responseMimeType"] == "application/json"
        assert body["maxOutputTokens"] == 700  # default here; config supplies 3000
        assert list(body["responseSchema"].required) == list(ENTITY_RESPONSE_JSON_SCHEMA["required"])
        assert "tools" not in body and "tools" not in parent

    def test_entity_system_instruction_is_the_entity_prompt(self, entity_prompt, prompt_contract):
        cfg = _entity_client(entity_prompt)._request_config()
        assert cfg.system_instruction == entity_prompt.system_instruction
        assert cfg.system_instruction != prompt_contract.system_instruction
        assert cfg.thinking_config is None

    def test_entity_budget_flows_from_entity_detection_config(self, entity_prompt, monkeypatch):
        # The real builder path, minus credentials: no SDK client is constructed.
        monkeypatch.setattr(GeminiClient, "_build_client", staticmethod(lambda: _FakeSdk([])))
        model_cfg = ModelConfig()
        client = build_entity_client(model_cfg, entity_prompt, model="gemini-3.5-flash",
                                     max_output_tokens=EntityDetectionConfig().max_output_tokens)
        assert isinstance(client, GeminiEntityClient)
        cfg = client._request_config()
        assert cfg.max_output_tokens == 3000
        assert cfg.max_output_tokens != model_cfg.max_output_tokens
        assert cfg.automatic_function_calling.disable is True
        assert client.request_config_summary() == {
            "model": "gemini-3.5-flash", "response_mime_type": "application/json",
            "response_schema": "entity", "max_output_tokens": 3000, "temperature": 0.0,
            "tools": None, "automatic_function_calling_disabled": True,
        }

    def test_exp03_config_file_carries_the_raised_budget(self, model_root):
        config = load_config(
            model_root / "config" / "config_exp03_entity_span_protected_confidence_gated.yaml",
            base_dir=model_root,
        )
        assert config.entity_detection.enabled is True
        assert config.entity_detection.max_output_tokens == 3000
        assert config.model.max_output_tokens == 700  # Town/Country budget untouched

    def test_entity_budget_below_floor_is_rejected(self):
        with pytest.raises(ValueError, match="at least 50"):
            EntityDetectionConfig(max_output_tokens=10)


class TestTownCountryRequestUnchanged:
    def test_town_country_request_shape(self, prompt_contract):
        cfg = _tc_client(prompt_contract)._request_config()
        assert isinstance(cfg, types.GenerateContentConfig)
        assert cfg.response_mime_type == "application/json"
        assert set(cfg.response_schema["required"]) == set(RESPONSE_JSON_SCHEMA["required"])
        assert "entity_spans" not in cfg.response_schema["properties"]
        assert cfg.max_output_tokens == 700
        assert cfg.temperature == 0.0
        assert cfg.system_instruction == prompt_contract.system_instruction
        assert cfg.tools is None
        assert cfg.tool_config is None
        assert cfg.thinking_config is None
        # EXP-03 did not touch the Town/Country request: no AFC override here.
        assert cfg.automatic_function_calling is None

    def test_town_country_summary(self, prompt_contract):
        assert _tc_client(prompt_contract).request_config_summary() == {
            "model": "gemini-3.5-flash", "response_mime_type": "application/json",
            "response_schema": "town_country", "max_output_tokens": 700, "temperature": 0.0,
            "tools": None, "automatic_function_calling_disabled": False,
        }

    def test_two_clients_do_not_share_schema_state(self, prompt_contract, entity_prompt):
        tc = _tc_client(prompt_contract)
        ent = _entity_client(entity_prompt)
        assert tc._request_config().response_schema == RESPONSE_JSON_SCHEMA
        assert ent._request_config().response_schema == ENTITY_RESPONSE_JSON_SCHEMA
        assert tc._request_config().response_schema != ent._request_config().response_schema
        assert GeminiClient(model="m", prompt=prompt_contract,
                            client=_FakeSdk([]))._response_schema is RESPONSE_JSON_SCHEMA


# --------------------------------------------------------------------------
# Truncation and malformed-answer handling
# --------------------------------------------------------------------------


def _usage(prompt=1200, candidates=120, thoughts=280):
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=candidates,
        thoughts_token_count=thoughts, total_token_count=prompt + candidates + thoughts,
    )


class TestTruncationDiagnostics:
    def test_usage_dict_records_thinking_tokens(self):
        usage = _usage_dict(_usage())
        assert usage["thoughts_token_count"] == 280
        assert usage["candidates_token_count"] == 120
        assert usage["total_token_count"] == 1600

    def test_max_tokens_truncation_is_reported_once_and_not_re_asked(self, entity_prompt, caplog):
        truncated = json.dumps(VALID_ENTITY)[:60]  # cut mid-string, like the live failures
        sdk = _FakeSdk([_Response(truncated, types.FinishReason.MAX_TOKENS, _usage())])
        client = _entity_client(entity_prompt, sdk, max_output_tokens=400)
        payload = build_entity_payload(FIELDS, {"address_line_1": ADDRESS_LINE})

        with caplog.at_level(logging.WARNING, logger="models.swft_tc.src.gemini_client"):
            with pytest.raises(TruncatedExtractionResponse) as info:
                client.identify(payload)

        assert client.call_count == 1          # no identical re-ask
        assert len(sdk.configs) == 1
        message = str(info.value)
        assert "MAX_TOKENS" in message and "max_output_tokens=400" in message
        assert "thoughts_token_count" in message
        assert ADDRESS_LINE not in message and "FUBON" not in message

        [record] = [r for r in caplog.records if "truncated" in r.getMessage()]
        text = record.getMessage()
        assert "'finish_reason': 'MAX_TOKENS'" in text
        assert "'attempt': 1" in text
        assert f"'response_chars': {len(truncated)}" in text
        assert "'starts_with_object_brace': True" in text
        assert "'ends_with_object_brace': False" in text
        assert "'thoughts_token_count': 280" in text
        assert "'response_mime_type': 'application/json'" in text
        assert "'response_schema': 'entity'" in text
        assert "'automatic_function_calling_disabled': True" in text
        assert "'payload_sha256': '" in text
        # Never the address, the payload, or the response body.
        assert "FUBON" not in text and ADDRESS_LINE not in text and truncated not in text

    def test_truncation_is_a_malformed_response_for_the_pipeline(self):
        assert issubclass(TruncatedExtractionResponse, MalformedExtractionResponse)

    def test_truncation_with_no_text_is_truncation_not_transient(self, entity_prompt):
        sdk = _FakeSdk([_Response("", types.FinishReason.MAX_TOKENS, _usage(candidates=0, thoughts=400))])
        client = _entity_client(entity_prompt, sdk, max_output_tokens=400)
        with pytest.raises(TruncatedExtractionResponse, match="with no text"):
            client.identify(build_entity_payload(FIELDS, {"address_line_1": ADDRESS_LINE}))
        assert client.call_count == 1

    def test_ordinary_malformed_answer_still_gets_one_re_ask(self, entity_prompt, caplog):
        sdk = _FakeSdk([
            _Response('{"entity_name": "x", "entity_type": ', types.FinishReason.STOP, _usage()),
            _Response(json.dumps(VALID_ENTITY), types.FinishReason.STOP, _usage()),
        ])
        client = _entity_client(entity_prompt, sdk)
        with caplog.at_level(logging.WARNING, logger="models.swft_tc.src.gemini_client"):
            outcome = client.identify(build_entity_payload(FIELDS, {"address_line_1": ADDRESS_LINE}))

        assert client.call_count == 2
        assert outcome.response.entity_name == "FUBON BANK HONG KONG LTD"
        assert outcome.usage["thoughts_token_count"] == 280
        [record] = [r for r in caplog.records if "malformed entity response" in r.getMessage()]
        text = record.getMessage()
        assert "(attempt 1/2)" in text
        assert "'finish_reason': 'STOP'" in text
        assert "'ends_with_object_brace': False" in text
        assert "FUBON" not in text and ADDRESS_LINE not in text

    def test_complete_answer_parses_and_keeps_usage(self, entity_prompt):
        sdk = _FakeSdk([_Response(json.dumps(VALID_ENTITY), types.FinishReason.STOP, _usage())])
        outcome = _entity_client(entity_prompt, sdk).identify(
            build_entity_payload(FIELDS, {"address_line_1": ADDRESS_LINE}))
        assert outcome.attempts == 1
        assert outcome.response.entity_type == "BANK"
        assert outcome.usage["candidates_token_count"] == 120

    def test_town_country_truncation_is_handled_the_same_way(self, prompt_contract, caplog):
        from models.swft_tc.src.reference_data import ReferenceContext

        sdk = _FakeSdk([_Response('{"town": "BOS', types.FinishReason.MAX_TOKENS, _usage())])
        client = _tc_client(prompt_contract, sdk)
        with caplog.at_level(logging.WARNING, logger="models.swft_tc.src.gemini_client"):
            with pytest.raises(TruncatedExtractionResponse, match="town_country"):
                client.extract("1 LINCOLN STREET BOSTON MA 02111 US", ReferenceContext())
        assert client.call_count == 1
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "'response_schema': 'town_country'" in joined
        assert "LINCOLN" not in joined
