from __future__ import annotations

from typing import Any

from .base import Fusion
from .rrf import ReciprocalRankFusion
from .score import NormalizedScoreFusion


def build_fusion(config: dict[str, Any]) -> Fusion:
    name = config.get("type", "rrf")
    params = {k: v for k, v in config.items() if k != "type"}
    if name == "rrf":
        return ReciprocalRankFusion(**params)
    if name == "normalized_sum":
        return NormalizedScoreFusion(**params)
    raise ValueError(f"Unknown fusion type: {name}")

