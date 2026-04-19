"""System metrics sampling."""

from __future__ import annotations

from typing import Any

_has_psutil = False
try:
    import psutil  # noqa: PLC0415
    _has_psutil = True
except ImportError:
    pass


def _init_gpu() -> tuple[Any, list[Any]] | None:
    try:
        import pynvml  # ty: ignore[unresolved-import]  # noqa: PLC0415

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            pynvml.nvmlShutdown()
            return None
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        return pynvml, handles
    except Exception:  # noqa: BLE001
        return None


def _nvml_query(func: Any, *args: Any) -> Any | None:
    try:
        return func(*args)
    except Exception:  # noqa: BLE001
        return None


class SystemMetrics:
    """Sample CPU, memory, and NVIDIA GPU metrics.

    Requires ``psutil`` for CPU/memory and ``pynvml`` for GPU metrics.
    Missing libraries are silently skipped.
    """

    def __init__(self, worker_label: str) -> None:
        """Initialize a system metrics collector.

        Args:
            worker_label: Worker label appended to each metric key to keep
                scalar keys unique across workers in a run.
        """
        self._worker_label = worker_label
        self._gpu = _init_gpu()
        self.available = _has_psutil or self._gpu is not None

    def sample(self) -> dict[str, float]:
        """Return a snapshot of current system metrics."""
        metrics: dict[str, float] = {}
        suffix = self._worker_label
        if _has_psutil:
            mem = psutil.virtual_memory()
            metrics[f"system/cpu/{suffix}"] = psutil.cpu_percent()
            metrics[f"system/memory/{suffix}"] = mem.percent
            metrics[f"system/memory_used/{suffix}"] = round(mem.used / (1024**3), 2)
        if self._gpu is not None:
            pynvml, handles = self._gpu
            for i, handle in enumerate(handles):
                prefix = f"system/gpu.{i}"
                if util := _nvml_query(pynvml.nvmlDeviceGetUtilizationRates, handle):
                    metrics[f"{prefix}/utilization/{suffix}"] = util.gpu
                    metrics[f"{prefix}/memory_percent/{suffix}"] = util.memory
                if mem_info := _nvml_query(pynvml.nvmlDeviceGetMemoryInfo, handle):
                    metrics[f"{prefix}/memory_used/{suffix}"] = round(mem_info.used / (1024**3), 2)
                if (power := _nvml_query(pynvml.nvmlDeviceGetPowerUsage, handle)) is not None:
                    metrics[f"{prefix}/power/{suffix}"] = round(power / 1000, 1)
                temp = _nvml_query(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU)
                if temp is not None:
                    metrics[f"{prefix}/temperature/{suffix}"] = temp
        return metrics

    def shutdown(self) -> None:
        """Release GPU resources."""
        if self._gpu is not None:
            self._gpu[0].nvmlShutdown()
            self._gpu = None
