"""Tests for `underfit.lib.metrics`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from underfit.lib import metrics


def test_system_metrics_samples_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample CPU and memory metrics when psutil is available."""
    monkeypatch.setattr(metrics, "_init_gpu", lambda: None)
    monkeypatch.setattr(metrics, "_has_psutil", True)
    monkeypatch.setattr(metrics, "psutil", SimpleNamespace(
        cpu_percent=lambda: 12.5, virtual_memory=lambda: SimpleNamespace(percent=34.5, used=3.456 * 1024**3),
    ))
    assert metrics.SystemMetrics("worker13").sample() == {
        "system/cpu/worker13": 12.5, "system/memory/worker13": 34.5, "system/memory_used/worker13": 3.46,
    }


def test_system_metrics_handles_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return no metrics when psutil and GPU support are unavailable."""
    monkeypatch.setattr(metrics, "_has_psutil", False)
    monkeypatch.setattr(metrics, "_init_gpu", lambda: None)
    collector = metrics.SystemMetrics("0")
    assert collector.available is False
    assert collector.sample() == {}
    collector.shutdown()
