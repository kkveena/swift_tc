"""Typed runtime configuration.

Configuration comes from YAML; credentials and per-run overrides come from the
environment. Secrets are never read from, written to, or logged by this module
— only the *names* of credential variables appear here.

Every business number the pipeline uses (reliability weights, HITL threshold,
retry bounds, output column templates) is loaded from YAML. There are no magic
numbers in the scoring or grouping code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AppConfig",
    "InputConfig",
    "ModelConfig",
    "OutputConfig",
    "ProcessingConfig",
    "ProjectConfig",
    "ReferenceDataConfig",
    "ReportingConfig",
    "ScenarioWeights",
    "ScoringConfig",
    "OUTPUT_FIELD_KEYS",
    "load_config",
    "resolve_model_name",
    "credentials_available",
    "dry_run_requested",
    "raw_logs_allowed",
]

#: The output fields produced for every enabled group, in CSV order.
#: Changing this tuple changes the per-group column count everywhere; nothing
#: else in the codebase hard-codes the count. `predicted_country_name` sits
#: directly after `predicted_country` and is derived deterministically from it
#: by the ISO reference layer — the model is never asked for a country name.
OUTPUT_FIELD_KEYS: tuple[str, ...] = (
    "combined_address",
    "combined_address_cleaned",
    "predicted_town",
    "predicted_country",
    "predicted_country_name",
    "predicted_town_probability",
    "predicted_country_probability",
    "predicted_town_exists",
    "predicted_country_exists",
    "composite_weighted_score",
    "rationale_town",
    "rationale_country",
)

#: Environment variables the Google Gen AI SDK accepts for the Developer API.
API_KEY_ENV_VARS: tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

_DEFAULT_CONFIG_PATH = "config/config.yaml"
_FALLBACK_CONFIG_PATH = "config/config.example.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Base):
    prompt_version: str = "v2-composite-weighted"
    record_id_column: str = "RECORD_ID"
    prompt_path: str = "prompts/GEMINI_EXTRACTION_PROMPT.md"
    group_config_path: str = "config/group_config.csv"


class InputConfig(_Base):
    preserve_all_columns: bool = True
    zero_field_is_missing: bool = True


class OutputConfig(_Base):
    naming_style: str = "canonical"
    country_candidate_separator: str = ","
    country_candidate_sort: str = "alphabetical"
    templates: dict[str, str]
    legacy_templates: dict[str, str] = Field(default_factory=dict)

    @field_validator("naming_style")
    @classmethod
    def _known_style(cls, value: str) -> str:
        if value not in {"canonical", "legacy"}:
            raise ValueError(
                f"output.naming_style must be 'canonical' or 'legacy', got {value!r}"
            )
        return value

    @field_validator("country_candidate_sort")
    @classmethod
    def _known_sort(cls, value: str) -> str:
        if value not in {"alphabetical", "model_order"}:
            raise ValueError(
                "output.country_candidate_sort must be 'alphabetical' or "
                f"'model_order', got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _templates_complete(self) -> "OutputConfig":
        active = self.active_templates
        missing = [key for key in OUTPUT_FIELD_KEYS if key not in active]
        if missing:
            raise ValueError(
                f"output.templates for naming_style={self.naming_style!r} is missing "
                f"entries for: {', '.join(missing)}"
            )
        for key, template in active.items():
            if "{id}" not in template:
                raise ValueError(
                    f"output template {key!r} must contain '{{id}}', got {template!r}"
                )
        return self

    @property
    def active_templates(self) -> Mapping[str, str]:
        """The single naming set in force. Both sets are never emitted together."""
        if self.naming_style == "legacy":
            return self.legacy_templates or self.templates
        return self.templates

    def column_name(self, field_key: str, group_id: str) -> str:
        try:
            template = self.active_templates[field_key]
        except KeyError as exc:  # pragma: no cover - guarded by _templates_complete
            raise KeyError(f"unknown output field key: {field_key!r}") from exc
        return template.format(id=group_id)


class ModelConfig(_Base):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_env_var: str = "GEMINI_MODEL"
    default_model: str = "gemini-3.5-flash"
    temperature: float = 0.0
    max_output_tokens: int = 700
    concurrency: int = 4
    max_retries: int = 5
    request_timeout_seconds: float = 60.0
    checkpoint_every_unique_addresses: int = 100
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    retry_jitter_seconds: float = 0.5
    max_in_flight_requests: int = 4
    enable_google_search_grounding: bool = False

    @model_validator(mode="after")
    def _sane_bounds(self) -> "ModelConfig":
        if self.concurrency < 1:
            raise ValueError("model.concurrency must be >= 1")
        if self.max_in_flight_requests < 1:
            raise ValueError("model.max_in_flight_requests must be >= 1")
        if self.max_retries < 0:
            raise ValueError("model.max_retries must be >= 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("model.temperature must be within [0, 2]")
        return self

    @property
    def effective_concurrency(self) -> int:
        """Worker count, never above the in-flight request ceiling."""
        return min(self.concurrency, self.max_in_flight_requests)


class ProcessingConfig(_Base):
    deduplicate_addresses: bool = True
    cache_enabled: bool = True
    cache_path: str = "outputs/address_cache.jsonl"
    errors_path: str = "outputs/processing_errors.csv"
    metrics_path: str = "outputs/run_metrics.json"
    output_path: str = "outputs/phase1_output.csv"
    redact_raw_address_in_logs: bool = True


class ReferenceDataConfig(_Base):
    provider: str = "iso3166"
    iso3166_path: str = "data/reference/iso3166.csv"
    swiftref_enabled: bool = False
    reference_context_version: str = "iso3166-2026-08"
    ambiguous_alpha2_tokens: tuple[str, ...] = ()
    trailing_country_token_window: int = 3

    # -- Town/Country development reference ------------------------------
    town_country_enabled: bool = False
    town_country_path: str = "data/reference/town_country_reference.csv"
    town_country_source: str = "development_local_reference"
    town_country_approved_for_production: bool = False
    #: Physical -> logical column mapping, for a reference file whose headers
    #: differ from the expected schema. Empty means "use the defaults".
    town_country_column_map: dict[str, str] = Field(default_factory=dict)
    #: 0 = unlimited. A positive value caps how many candidate codes are written
    #: to the CSV when reference ambiguity is escalated; the full set is always
    #: kept in the audit payload.
    town_country_max_candidates: int = 0
    #: How a multi-country town name affects the result. See scoring.py.
    town_country_ambiguity_policy: str = "escalate"

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        if value not in {"null", "iso3166", "composite"}:
            raise ValueError(
                "reference_data.provider must be 'null', 'iso3166' or 'composite', "
                f"got {value!r}"
            )
        return value

    @field_validator("town_country_ambiguity_policy")
    @classmethod
    def _known_policy(cls, value: str) -> str:
        if value not in {"escalate", "annotate"}:
            raise ValueError(
                "reference_data.town_country_ambiguity_policy must be 'escalate' "
                f"or 'annotate', got {value!r}"
            )
        return value

    @field_validator("ambiguous_alpha2_tokens", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        if value is None:
            return ()
        return tuple(str(token).strip().upper() for token in value)


class ReportingConfig(_Base):
    """Executive-report output paths and the score-distribution band edges."""

    enabled: bool = True
    reports_dir: str = "outputs/reports"
    charts_dir: str = "outputs/charts"
    executive_summary_filename: str = "executive_summary.json"
    score_distribution_filename: str = "score_distribution.csv"
    threshold_sensitivity_filename: str = "threshold_sensitivity.csv"
    scenario_distribution_filename: str = "scenario_distribution.csv"
    histogram_filename: str = "composite_score_histogram.png"
    #: Lower edges of the score bands. Each band is [edge, next_edge), and the
    #: final band is closed at 1.0 so a perfect score is never dropped.
    score_band_edges: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    #: Thresholds evaluated by the HITL sensitivity table.
    sensitivity_thresholds: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
    #: Provisional routing recommendation surfaced by the report. Not calibrated
    #: accuracy — see SCORING_SPEC.md.
    recommended_threshold: float = 0.90

    @model_validator(mode="after")
    def _validate_edges(self) -> "ReportingConfig":
        if not self.score_band_edges:
            raise ValueError("reporting.score_band_edges must not be empty")
        edges = list(self.score_band_edges)
        if edges != sorted(edges) or len(set(edges)) != len(edges):
            raise ValueError("reporting.score_band_edges must be strictly increasing")
        if not all(0.0 < edge < 1.0 for edge in edges):
            raise ValueError("reporting.score_band_edges must lie strictly within (0, 1)")
        for threshold in self.sensitivity_thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"reporting.sensitivity_thresholds entry {threshold} is outside [0, 1]"
                )
        if not 0.0 <= self.recommended_threshold <= 1.0:
            raise ValueError("reporting.recommended_threshold must lie within [0, 1]")
        return self


class ScenarioWeights(_Base):
    town_weight: float
    country_weight: float

    @model_validator(mode="after")
    def _within_unit_interval(self) -> "ScenarioWeights":
        for name, weight in (
            ("town_weight", self.town_weight),
            ("country_weight", self.country_weight),
        ):
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"scoring rule {name}={weight} must be within [0, 1]")
        return self


class ScoringConfig(_Base):
    mode: str = "composite_weighted"
    hitl_threshold: float = 0.80
    ambiguous_country_probability_override: float = 0.0
    force_ambiguous_country_to_hitl: bool = True
    rules: dict[str, ScenarioWeights]

    @model_validator(mode="after")
    def _required_scenarios_present(self) -> "ScoringConfig":
        from .scoring import REQUIRED_SCENARIOS  # local import avoids a cycle

        missing = [name for name in REQUIRED_SCENARIOS if name not in self.rules]
        if missing:
            raise ValueError(
                "scoring.rules is missing required scenarios: " + ", ".join(missing)
            )
        if not 0.0 <= self.hitl_threshold <= 1.0:
            raise ValueError("scoring.hitl_threshold must be within [0, 1]")
        return self

    def weights_for(self, scenario: str) -> ScenarioWeights:
        try:
            return self.rules[scenario]
        except KeyError as exc:
            raise KeyError(
                f"no reliability weights configured for scenario {scenario!r}; "
                f"known scenarios: {', '.join(sorted(self.rules))}"
            ) from exc


class AppConfig(_Base):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    reference_data: ReferenceDataConfig = Field(default_factory=ReferenceDataConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    scoring: ScoringConfig

    #: Directory every relative path in the config is resolved against.
    base_dir: Path = Field(default_factory=Path.cwd)

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.base_dir / candidate

    def group_column_names(self, group_id: str) -> tuple[str, ...]:
        """The 11 output column names for one group, in CSV order."""
        return tuple(
            self.output.column_name(key, group_id) for key in OUTPUT_FIELD_KEYS
        )

    @property
    def fields_per_group(self) -> int:
        return len(OUTPUT_FIELD_KEYS)


@dataclass(frozen=True)
class _Loaded:
    data: dict[str, Any]
    source: Path


def _read_yaml(path: Path) -> _Loaded:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return _Loaded(data=data, source=path)


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load and validate the runtime configuration.

    Resolution order for the config file: explicit argument, then
    ``SWIFT_ADDRESS_CONFIG``, then ``config/config.yaml``, then
    ``config/config.example.yaml``.

    ``overrides`` is a shallow per-section mapping applied on top of the YAML,
    used by tests and by the notebook to flip individual settings without
    editing the file.
    """
    root = Path(base_dir) if base_dir is not None else Path.cwd()

    if config_path is not None:
        resolved = Path(config_path)
        resolved = resolved if resolved.is_absolute() else root / resolved
    else:
        env_path = os.environ.get("SWIFT_ADDRESS_CONFIG")
        if env_path:
            resolved = Path(env_path)
            resolved = resolved if resolved.is_absolute() else root / resolved
        else:
            resolved = root / _DEFAULT_CONFIG_PATH
            if not resolved.exists():
                resolved = root / _FALLBACK_CONFIG_PATH

    if not resolved.exists():
        raise FileNotFoundError(
            f"configuration file not found: {resolved}. Provide config/config.yaml "
            "or set SWIFT_ADDRESS_CONFIG."
        )

    loaded = _read_yaml(resolved)
    data = dict(loaded.data)

    if overrides:
        for section, values in overrides.items():
            if isinstance(values, Mapping) and isinstance(data.get(section), Mapping):
                merged = dict(data[section])
                merged.update(values)
                data[section] = merged
            else:
                data[section] = values

    group_override = os.environ.get("SWIFT_ADDRESS_GROUP_CONFIG")
    if group_override:
        project = dict(data.get("project") or {})
        project["group_config_path"] = group_override
        data["project"] = project

    data["base_dir"] = root
    return AppConfig.model_validate(data)


def resolve_model_name(config: AppConfig, env: Mapping[str, str] | None = None) -> str:
    """Model ID from the configured environment variable, else the YAML default."""
    environ = os.environ if env is None else env
    return environ.get(config.model.model_env_var) or config.model.default_model


def credentials_available(env: Mapping[str, str] | None = None) -> bool:
    """Whether a real Gemini call could be attempted.

    True when a Developer API key is present, or when Vertex AI mode is enabled
    (Vertex uses Application Default Credentials rather than an API key). The
    value of any credential is never read, returned, or logged.
    """
    environ = os.environ if env is None else env
    if any(environ.get(name) for name in API_KEY_ENV_VARS):
        return True
    return _is_truthy(environ.get("GOOGLE_GENAI_USE_VERTEXAI"))


def dry_run_requested(env: Mapping[str, str] | None = None) -> bool:
    """Whether ``SWIFT_ADDRESS_DRY_RUN`` forces mock extraction."""
    environ = os.environ if env is None else env
    return _is_truthy(environ.get("SWIFT_ADDRESS_DRY_RUN"))


def raw_logs_allowed(env: Mapping[str, str] | None = None) -> bool:
    """Whether raw addresses may appear in DEBUG logs (off unless opted in)."""
    environ = os.environ if env is None else env
    return _is_truthy(environ.get("SWIFT_ADDRESS_ALLOW_RAW_LOGS"))


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
