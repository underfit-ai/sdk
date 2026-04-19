"""Shared fixtures for end-to-end SDK ↔ API tests."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

os.environ.setdefault("UNDERFIT_APP_SECRET", "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=")

import underfit_api.db as db  # noqa: E402
import underfit_api.storage as storage_mod  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from underfit_api.config import BackfillConfig, FileStorageConfig, SqliteDatabaseConfig, config  # noqa: E402
from underfit_api.main import app  # noqa: E402
from underfit_api.repositories import accounts as accounts_repo  # noqa: E402
from underfit_api.repositories import api_keys as api_keys_repo  # noqa: E402
from underfit_api.repositories import projects as projects_repo  # noqa: E402
from underfit_api.repositories import users as users_repo  # noqa: E402
from underfit_api.schema import metadata  # noqa: E402

import underfit  # noqa: E402


@pytest.fixture
def api_tmp_path(tmp_path: Path) -> Iterator[Path]:
    """Per-test temp dir that resets the API config and disposes resources on teardown."""
    original_database = config.database
    original_storage = config.storage
    original_auth = config.auth_enabled
    db.engine.dispose()
    yield tmp_path
    db.engine.dispose()
    config.database = original_database
    config.storage = original_storage
    config.auth_enabled = original_auth
    db.engine = db.build_engine()
    storage_mod.storage = storage_mod.build_storage()


def _reset_sdk_state() -> None:
    underfit.run = None
    underfit._capture_context = None  # noqa: SLF001


@pytest.fixture
def reset_sdk() -> Iterator[None]:
    """Ensure the global SDK run handle is cleared even when a test fails mid-run."""
    _reset_sdk_state()
    yield
    if underfit.run is not None:
        try:
            underfit.finish()
        except Exception:
            _reset_sdk_state()


@pytest.fixture
def remote_env(api_tmp_path: Path, reset_sdk: None) -> Iterator[dict[str, Any]]:
    """Boot the API with auth + a user/project/API key, and patch the SDK's urlopen."""
    config.database = SqliteDatabaseConfig(path=str(api_tmp_path / "db.sqlite"))
    config.storage = FileStorageConfig(base=str(api_tmp_path / "storage"))
    config.auth_enabled = True
    db.engine = db.build_engine()
    storage_mod.storage = storage_mod.build_storage()
    metadata.drop_all(db.engine)
    metadata.create_all(db.engine)

    handle, project_name = "owner", "vision"
    with db.engine.begin() as conn:
        user = users_repo.create(conn, f"{handle}@example.com", handle, "Test User")
        accounts_repo.create_alias(conn, user.id, handle)
        project = projects_repo.create(conn, user.id, project_name, "e2e", "private", {})
        projects_repo.create_alias(conn, project.id, user.id, project_name)
        api_key = api_keys_repo.create(conn, user.id, "e2e")

    local_file = api_tmp_path / "payload.json"
    local_file.write_bytes(b'{"y": 2}')
    reference_file = api_tmp_path / "model-card.txt"
    reference_file.write_text("model-card\n", encoding="utf-8")
    os.environ["UNDERFIT_API_KEY"] = api_key.token
    with TestClient(app) as client, patch(
        "underfit.backends.remote.urllib.request.urlopen", side_effect=_make_urlopen_shim(client),
    ):
        yield {
            "client": client, "handle": handle, "project": project_name, "api_key": api_key.token,
            "local_file": local_file, "reference_file": reference_file,
        }


@pytest.fixture
def local_env(api_tmp_path: Path, reset_sdk: None) -> dict[str, Any]:  # noqa: ARG001
    """Run the SDK against a local logdir then boot a backfill-enabled API for verification."""
    log_dir = api_tmp_path / "logs"
    log_dir.mkdir()
    return {"log_dir": log_dir, "api_tmp_path": api_tmp_path}


def boot_backfill_client(api_tmp_path: Path, log_dir: Path) -> TestClient:
    """Configure the API for backfill from ``log_dir`` and return an entered TestClient."""
    backfill = BackfillConfig(enabled=True, scan_interval_s=1, debounce_ms=50)
    config.database = SqliteDatabaseConfig(path=str(api_tmp_path / "db.sqlite"))
    config.storage = FileStorageConfig(base=str(log_dir), backfill=backfill)
    config.auth_enabled = False
    db.engine = db.build_engine()
    storage_mod.storage = storage_mod.build_storage()
    metadata.drop_all(db.engine)
    metadata.create_all(db.engine)
    return TestClient(app)


def _make_urlopen_shim(client: TestClient) -> Callable[..., Any]:
    """Translate ``urllib.request.urlopen`` calls into in-process TestClient requests."""
    def shim(req: Any, **_: Any) -> _ShimResponse:
        method = req.get_method()
        url = req.full_url
        headers = {k: v for k, v in req.header_items()}
        body = req.data
        response = client.request(method, url, content=body, headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {url} -> {response.status_code}: {response.text}")
        return _ShimResponse(response.content)

    return shim


class _ShimResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def read(self) -> bytes:
        return self._content

    def __enter__(self) -> _ShimResponse:  # noqa: PYI034
        return self

    def __exit__(self, *_: object) -> None:
        pass
