import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "jupyter" / "notebooks"))
import lab_utils  # noqa: E402


def test_missing_endpoint_env_raises_clear_error(monkeypatch):
    monkeypatch.delenv("TRINO_HOST", raising=False)
    with pytest.raises(RuntimeError, match="run this notebook in the lab's Jupyter"):
        lab_utils.endpoint("TRINO_HOST")


def test_endpoint_returns_env(monkeypatch):
    monkeypatch.setenv("TRINO_HOST", "t")
    assert lab_utils.endpoint("TRINO_HOST") == "t"
