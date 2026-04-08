"""Tests for `underfit.init`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import underfit
from underfit.backends.local import LocalBackend


def test_init_captures_terminal_output(tmp_path: Path) -> None:
    """Capture stdout and stderr into backend logs."""
    run = underfit.init("project", log_dir=tmp_path)
    assert re.match(r"^[a-z]+-[a-z]+$", run.name)
    sys.stdout.write("hello")
    sys.stdout.write(" world\n")
    sys.stderr.write("oops")
    underfit.finish()
    assert isinstance(run.backend, LocalBackend)
    assert (run.backend.run_dir / "logs" / "log.log").read_text() == "hello world\noops\n"


def test_init_supports_context_manager(tmp_path: Path) -> None:
    """Finish the active run when exiting a context."""
    with underfit.init("project", log_dir=tmp_path) as run:
        assert underfit.run is run
    assert underfit.run is None
    with pytest.raises(RuntimeError, match="run is already finished"):
        run.log({"x": 1})
    assert run.backend.run_dir.exists()
