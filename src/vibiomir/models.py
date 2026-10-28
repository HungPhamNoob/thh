from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def resolve_hf_model(
    model_name: str,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> str:
    """Resolve a pinned Hub revision to a local snapshot for third-party loaders."""
    if revision is None or Path(model_name).exists():
        return model_name
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install `huggingface_hub` to load pinned model revisions") from exc
    return snapshot_download(
        repo_id=model_name,
        revision=revision,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )


def repair_1d_position_id_buffers(model: object, torch_module: Any | None = None) -> int:
    """Recreate non-persistent one-dimensional position-id buffers after HF loading.

    Alibaba's ``NewEmbeddings`` registers a one-dimensional, non-persistent
    ``position_ids`` buffer.  Recent Transformers low-memory loading can leave
    that buffer allocated but uninitialised when the model is materialised on
    CUDA.  Recreating the deterministic ``arange`` buffer restores the exact
    model implementation without modifying weights or inference semantics.

    Standard Transformers models use a two-dimensional position-id buffer and
    are intentionally left untouched.
    """
    if torch_module is None:
        import torch as torch_module

    repaired = 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return repaired
    for module in modules():
        position_ids = getattr(module, "position_ids", None)
        if position_ids is None or getattr(position_ids, "ndim", None) != 1:
            continue
        size = int(position_ids.numel())
        if size <= 0:
            continue
        replacement = torch_module.arange(
            size,
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
        module.register_buffer("position_ids", replacement, persistent=False)
        repaired += 1
    return repaired
