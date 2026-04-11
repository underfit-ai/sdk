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


def test_init_remote_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require an API key before creating a remote run."""
    monkeypatch.delenv("UNDERFIT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="UNDERFIT_API_KEY"):
        underfit.init("project", remote_url="https://example.com")


def test_init_uses_env_log_dir_and_reuses_active_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Honor UNDERFIT_LOG_DIR and return the existing run on repeated init calls."""
    monkeypatch.setenv("UNDERFIT_LOG_DIR", str(tmp_path))
    run = underfit.init("project", worker_label="worker-0")
    assert isinstance(run.backend, LocalBackend)
    assert run.backend.run_dir.parent == tmp_path.resolve()
    assert underfit.init("other-project", worker_label="worker-1") is run
    underfit.finish()
