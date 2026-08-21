"""Shared fixtures.

Every test runs against the model's real config and group config: the
acceptance criteria are about *this* configuration (16 groups, 50-column
sample input), so a synthetic stand-in would not prove them.

Paths are anchored on ``MODEL_ROOT`` (derived from the package location) and
on this file, never on the current working directory, so the suite runs
identically from the enterprise repository root or from anywhere else. No
``sys.path`` manipulation is needed: the repository root is on the import
path via ``pytest.ini``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.swft_tc.src.gemini_client import MockExtractionClient
from models.swft_tc.src.grouping import load_group_config
from models.swft_tc.src.reference_data import build_provider
from models.swft_tc.src.schemas import load_prompt_contract
from models.swft_tc.src.settings import MODEL_ROOT, REPO_ROOT, load_config

TESTS_ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The enterprise repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def model_root() -> Path:
    """The model package root that owns config, prompts, data and outputs."""
    return MODEL_ROOT


@pytest.fixture(scope="session")
def sample_input_path(model_root: Path) -> Path:
    return model_root / "data" / "sample_input.csv"


@pytest.fixture(scope="session")
def town_country_fixture_path() -> Path:
    """The tiny committed Town/Country fixture.

    Tests never touch the real reference file: it is a large, environment-
    specific external dependency that is not in version control.
    """
    return TESTS_ROOT / "fixtures" / "town_country_reference_test.csv"


@pytest.fixture
def config(model_root: Path, tmp_path: Path, town_country_fixture_path: Path):
    """Model config with outputs redirected and the reference file stubbed."""
    return load_config(
        model_root / "config" / "config.yaml",
        base_dir=model_root,
        overrides={
            "processing": {
                "cache_path": str(tmp_path / "address_cache.jsonl"),
                "errors_path": str(tmp_path / "processing_errors.csv"),
                "metrics_path": str(tmp_path / "run_metrics.json"),
                "output_path": str(tmp_path / "phase1_output.csv"),
            },
            "reporting": {
                "reports_dir": str(tmp_path / "reports"),
                "charts_dir": str(tmp_path / "charts"),
            },
            "reference_data": {
                "town_country_path": str(town_country_fixture_path),
            },
        },
    )


@pytest.fixture
def group_config(config):
    return load_group_config(config.path(config.project.group_config_path))


@pytest.fixture
def reference_provider(config):
    return build_provider(config.reference_data, base_dir=config.base_dir)


@pytest.fixture
def iso_provider(reference_provider):
    from models.swft_tc.src.reference_data import find_iso_provider

    return find_iso_provider(reference_provider)


@pytest.fixture
def town_country_provider(config):
    from models.swft_tc.src.reference_data import build_town_country_provider

    return build_town_country_provider(config.reference_data, base_dir=config.base_dir)


@pytest.fixture
def prompt_contract(config):
    return load_prompt_contract(
        config.path(config.project.prompt_path), config.project.prompt_version
    )


@pytest.fixture
def mock_client(iso_provider) -> MockExtractionClient:
    return MockExtractionClient(iso_provider=iso_provider)
