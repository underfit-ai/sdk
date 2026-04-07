"""Tests for `underfit.init`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
