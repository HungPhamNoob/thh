"""GPU resource limits for on-device encode/rerank jobs.

Competition machines are shared laptops: long indexing jobs must leave the GPU
usable for everything else. Set ``VIBIOMIR_GPU_MEM_FRACTION=0.4`` (any value in
(0, 1]) to cap this process's share of each CUDA device's memory; the PyTorch
allocator then refuses to grow past that fraction instead of eating all VRAM.

The cap is applied lazily before a model is loaded and is a no-op when CUDA or
PyTorch is unavailable, so CPU smoke tests keep working unchanged.
"""

from __future__ import annotations

import os

GPU_MEM_FRACTION_ENV = "VIBIOMIR_GPU_MEM_FRACTION"

_applied_fraction: float | None = None


def gpu_memory_fraction() -> float | None:
    """Parse the requested CUDA memory fraction, or None when unset."""
    raw = os.getenv(GPU_MEM_FRACTION_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{GPU_MEM_FRACTION_ENV} must be a number in (0, 1], got {raw!r}") from None
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{GPU_MEM_FRACTION_ENV} must be in (0, 1], got {value!r}")
    return value


def apply_gpu_memory_fraction() -> float | None:
    """Cap the CUDA allocator share for this process; returns the fraction applied.

    No-op (returns None) when the fraction is unset, torch is missing, or no
    CUDA device is visible. Safe to call repeatedly and from every model loader.
    """
    global _applied_fraction
    fraction = gpu_memory_fraction()
    if fraction is None:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional extra
        return None
    if not torch.cuda.is_available():
        return None
    if _applied_fraction != fraction:
        torch.cuda.set_per_process_memory_fraction(fraction)
        _applied_fraction = fraction
    return fraction
