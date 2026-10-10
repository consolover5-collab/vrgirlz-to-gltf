from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic"
LUCID_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_lucid"
INSTANCES_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic_instances"


@pytest.fixture
def fixture_dir() -> Path:
    if not (FIXTURE_DIR / "scene.json").exists():
        pytest.fail(f"{FIXTURE_DIR} missing — run `uv run python scripts/make-fixture.py` first")
    return FIXTURE_DIR


@pytest.fixture
def lucid_fixture_dir() -> Path:
    if not (LUCID_FIXTURE_DIR / "scene.json").exists():
        pytest.fail(f"{LUCID_FIXTURE_DIR} missing — run `uv run python scripts/make-fixture-lucid.py` first")
    return LUCID_FIXTURE_DIR


@pytest.fixture
def instances_fixture_dir() -> Path:
    if not (INSTANCES_FIXTURE_DIR / "scene.json").exists():
        pytest.fail(f"{INSTANCES_FIXTURE_DIR} missing — run `uv run python scripts/make-fixture-instances.py` first")
    return INSTANCES_FIXTURE_DIR
