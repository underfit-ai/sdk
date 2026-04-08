"""Tests for `underfit.init`."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

import underfit
from underfit.backends.local import LocalBackend
from underfit.run import Run


def test_init_captures_terminal_output(tmp_path: Path) -> None:
    """Capture stdout and stderr into backend logs."""
    run = underfit.init("project", log_dir=tmp_path, worker_label="worker-0")
    assert re.match(r"^[a-z]+-[a-z]+$", run.name)
    sys.stdout.write("hello")
    sys.stdout.write(" world\n")
    sys.stderr.write("oops")
    underfit.finish()
    assert isinstance(run.backend, LocalBackend)
    assert (run.backend.run_dir / "logs" / "worker-0" / "segments" / "0.log").read_text() == "hello world\noops\n"


@pytest.mark.parametrize(
    ("error", "state"),
    [(None, "finished"), (KeyboardInterrupt, "cancelled"), (RuntimeError, "failed")],
)
def test_init_supports_context_manager(
    tmp_path: Path, mocker: MockerFixture, error: type[BaseException] | None, state: str,
) -> None:
    """Finish the active run when exiting a context."""
    spy = mocker.spy(Run, "finish")

    def exercise() -> None:
        with underfit.init("project", log_dir=tmp_path) as run:
            assert underfit.run is run
            if error is not None:
                raise error()

    if error is None:
        exercise()
    else:
        with pytest.raises(error):
            exercise()
    assert underfit.run is None
    spy.assert_called_once()
    assert spy.call_args.args[1] == state
