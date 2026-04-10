"""Tests for the Underfit CLI."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from underfit.cli import main


def test_view_errors_when_logdir_is_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit with an error when the log directory does not exist."""
    assert main(["view", "--logdir", str(tmp_path / "missing")]) == 1
    assert "logdir does not exist" in capsys.readouterr().err


def test_view_uses_default_logdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the local viewer against the default log directory."""
    calls: dict[str, object] = {}

    def run(app: object) -> None:
        calls["app"] = app
        calls["config"] = Path(os.environ["UNDERFIT_CONFIG"]).read_text()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "underfit").mkdir()
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.__dict__["run"] = run
    underfit_api = types.ModuleType("underfit_api")
    underfit_api.__dict__["__path__"] = []
    api_main = types.ModuleType("underfit_api.main")
    api_main.__dict__["app"] = object()
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setitem(sys.modules, "underfit_api", underfit_api)
    monkeypatch.setitem(sys.modules, "underfit_api.main", api_main)

    assert main(["view"]) == 0
    assert calls["app"] is api_main.app
    assert str((tmp_path / "underfit").resolve()) in str(calls["config"])
    assert "UNDERFIT_CONFIG" not in os.environ
