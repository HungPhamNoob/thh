from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


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
