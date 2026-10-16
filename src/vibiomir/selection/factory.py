from __future__ import annotations

from typing import Any

from .selectors import AdaptiveGapSelector, MMRSelector, ThresholdTopKSelector


def build_selector(config: dict[str, Any]):
    name = config.get("type", "threshold_topk")
    params = {k: v for k, v in config.items() if k not in {"type", "per_document", "global_max"}}
    if name == "threshold_topk":
        return ThresholdTopKSelector(**params)
    if name == "adaptive_gap":
        return AdaptiveGapSelector(**params)
    if name == "mmr":
        return MMRSelector(**params)
    raise ValueError(f"Unknown selector type: {name}")

