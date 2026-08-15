"""Shared fixtures.

Every test runs against the repository's real config and group config: the
acceptance criteria are about *this* configuration (16 groups, 50-column
sample input), so a synthetic stand-in would not prove them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(SRC))

from swift_address.gemini_client import MockExtractionClient  # noqa: E402
from swift_address.grouping import load_group_config  # noqa: E402
from swift_address.reference_data import build_provider  # noqa: E402
from swift_address.schemas import load_prompt_contract  # noqa: E402
from swift_address.settings import load_config  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_input_path(repo_root: Path) -> Path:
    return repo_root / "data" / "sample_input.csv"


@pytest.fixture(scope="session")
def town_country_fixture_path(repo_root: Path) -> Path:
    """The tiny committed Town/Country fixture.

    Tests never touch the real reference file: it is a large, environment-
    specific external dependency that is not in version control.
    """
    return repo_root / "tests" / "fixtures" / "town_country_reference_test.csv"


@pytest.fixture
def config(repo_root: Path, tmp_path: Path, town_country_fixture_path: Path):
    """Repository config with outputs redirected and the reference file stubbed."""
    return load_config(
        repo_root / "config" / "config.yaml",
        base_dir=repo_root,
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
    from swift_address.reference_data import find_iso_provider

    return find_iso_provider(reference_provider)


@pytest.fixture
def town_country_provider(config):
    from swift_address.reference_data import build_town_country_provider

    return build_town_country_provider(config.reference_data, base_dir=config.base_dir)


@pytest.fixture
def prompt_contract(config):
    return load_prompt_contract(
        config.path(config.project.prompt_path), config.project.prompt_version
    )


@pytest.fixture
def mock_client(iso_provider) -> MockExtractionClient:
    return MockExtractionClient(iso_provider=iso_provider)
