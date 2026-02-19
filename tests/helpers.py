from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def fixture_path(name: str) -> Path:
    return FIXTURES_DIR / name


def assert_no_traceback(stderr_text: str) -> None:
    assert "Traceback" not in stderr_text
