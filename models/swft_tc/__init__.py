"""SWIFT address Town/Country model package.

Enterprise layout — the reusable implementation and everything it needs at
runtime live together under this directory:

    config/   runtime configuration (YAML + group config CSV)
    data/     sample input and the reference-data location
    outputs/  generated run artifacts
    prompts/  the single source of the extraction prompt
    src/      the reusable modules

Notebooks, executable scripts and tests deliberately live outside this package
(`notebooks/swft_tc/`, `scripts/swft_tc/`, `tests/swft_tc/`) and import from it
rather than carrying their own copies of the business logic.

Import convention::

    from models.swft_tc.src.pipeline import Phase1Pipeline, run_phase1
    from models.swft_tc.src.settings import load_config
"""

from pathlib import Path

#: Root of this model package. Every configured relative path resolves against
#: it, derived from the module location rather than the working directory, so
#: the pipeline behaves identically however it is launched.
MODEL_ROOT = Path(__file__).resolve().parent

#: Enterprise repository root — `models/swft_tc/` -> `models/` -> repo root.
REPO_ROOT = MODEL_ROOT.parents[1]

__all__ = ["MODEL_ROOT", "REPO_ROOT"]
