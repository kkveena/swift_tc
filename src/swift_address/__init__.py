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
reference_data  ReferenceDataProvider abstraction (null / ISO 3166 / SWIFTRef)
gemini_client   extraction client protocol, Gemini implementation, mock client
scoring         verified scenario selection, reliability weights, composite score
cache           stable cache key + JSONL result cache
pipeline        Pass 1 / Pass 2 orchestration
"""

__version__ = "1.0.0"

__all__ = [
    "cache",
    "cleaning",
    "gemini_client",
    "grouping",
    "io",
    "pipeline",
    "reference_data",
    "schemas",
    "scoring",
    "settings",
]
