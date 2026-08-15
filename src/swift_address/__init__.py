"""SWIFT address Town/Country extraction — Phase 1.

Notebook-first, module-backed. Every piece of business logic lives in this
package so Phase 2 can reuse it without lifting code out of a notebook.

Module map
----------
settings        typed runtime configuration (YAML + environment)
io              CSV read/write, error sidecar, run metrics
grouping        group config loading/validation, combined-address construction
cleaning        deterministic normalization and token-boundary matching
schemas         Gemini structured-response schema and validation
reference_data  ReferenceDataProvider abstraction (null / ISO 3166 / Town-Country
                / SWIFTRef stub)
gemini_client   extraction client protocol, Gemini implementation, mock client
scoring         verified scenario selection, reliability weights, composite score
evaluation      nullable ground-truth labels + binary cross-entropy
retraction      deterministic removal of verified Town/Country evidence
cache           stable cache key + JSONL result cache
pipeline        Pass 1 / Pass 2 orchestration
reporting       executive KPIs, score distribution, HITL threshold sensitivity
serialization   streaming nested per-record detail (JSONL)

Two pairs of concepts are deliberately kept apart:

predicted_*_exists   is the value explicitly present in the input text?
*_exists_ok          when independent evidence exists, was it correct? (unknown -> False)

composite_weighted_score   operational HITL routing — HIGHER is better
cross_entropy              confidence calibration vs truth — LOWER is better
"""

__version__ = "1.0.0"

__all__ = [
    "cache",
    "cleaning",
    "evaluation",
    "gemini_client",
    "grouping",
    "io",
    "pipeline",
    "reference_data",
    "reporting",
    "retraction",
    "schemas",
    "scoring",
    "serialization",
    "settings",
]
