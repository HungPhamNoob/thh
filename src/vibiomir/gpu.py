"""Hard CUDA memory controls and lightweight runtime telemetry.

The production target is a GPU with at most 4 GiB VRAM.  PyTorch's allocator
limit is therefore derived from an absolute GiB budget, not from the capacity
of the development GPU.  The default 3.0 GiB allocator budget leaves room for
the CUDA context and non-PyTorch allocations so the process stays below the
3.5 GiB operational target.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

GPU_MAX_GIB_ENV = "VIBIOMIR_GPU_MAX_GIB"
GPU_MEM_FRACTION_ENV = "VIBIOMIR_GPU_MEM_FRACTION"
GPU_REPORT_ENV = "VIBIOMIR_GPU_REPORT"
DEFAULT_GPU_MAX_GIB = 3.0
HARD_GPU_MAX_GIB = 4.0

_applied: tuple[int, float] | None = None
_oom_count = 0


def gpu_max_gib() -> float:
    """Return the requested absolute allocator budget, capped at 4 GiB."""
    raw = os.getenv(GPU_MAX_GIB_ENV, str(DEFAULT_GPU_MAX_GIB)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{GPU_MAX_GIB_ENV} must be a number in (0, 4], got {raw!r}") from exc
    if not 0.0 < value <= HARD_GPU_MAX_GIB:
        raise ValueError(f"{GPU_MAX_GIB_ENV} must be in (0, 4], got {value!r}")
    return value


def _optional_fraction() -> float | None:
    raw = os.getenv(GPU_MEM_FRACTION_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{GPU_MEM_FRACTION_ENV} must be in (0, 1], got {raw!r}") from exc
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{GPU_MEM_FRACTION_ENV} must be in (0, 1], got {value!r}")
    return value


def apply_gpu_memory_limit(device: int = 0) -> dict[str, Any] | None:
    """Apply the absolute allocator cap before loading a model.

    Returns a report when CUDA is available and ``None`` on CPU-only hosts.
    Repeated calls are idempotent.
    """
    global _applied
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional
        return None
    if not torch.cuda.is_available():
        return None
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    requested_bytes = int(gpu_max_gib() * 1024**3)
    fraction_override = _optional_fraction()
    if fraction_override is not None:
        requested_bytes = min(requested_bytes, int(total_bytes * fraction_override))
    fraction = min(1.0, requested_bytes / total_bytes)
    if _applied != (device, fraction):
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        _applied = (device, fraction)
    report = gpu_report(device)
    _write_report(report)
    return report


def gpu_report(device: int = 0) -> dict[str, Any]:
    """Return current PyTorch CUDA telemetry without exposing credentials."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return {"cuda_available": False, "oom_count": _oom_count}
    if not torch.cuda.is_available():
        return {"cuda_available": False, "oom_count": _oom_count}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_vram_bytes": int(total_bytes),
        "available_vram_bytes": int(free_bytes),
        "allocator_limit_gib": gpu_max_gib(),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "oom_count": _oom_count,
    }


def is_cuda_oom(exc: BaseException) -> bool:
    """Recognize PyTorch CUDA OOM errors without importing torch eagerly."""
    return "out of memory" in str(exc).casefold() and "cuda" in str(exc).casefold()


def record_cuda_oom() -> None:
    """Increment telemetry and release unused cached CUDA blocks."""
    global _oom_count
    _oom_count += 1
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
    _write_report(gpu_report())


def _write_report(report: dict[str, Any]) -> None:
    path = os.getenv(GPU_REPORT_ENV, "").strip()
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
