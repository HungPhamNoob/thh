from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    if override.get("_replace_") is True:
        return {key: copy.deepcopy(value) for key, value in override.items() if key != "_replace_"}
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("extends", None)
    if parent is None:
        return current
    parents = [parent] if isinstance(parent, str) else parent
    merged: dict[str, Any] = {}
    for parent_path in parents:
        candidate = (path.parent / parent_path).resolve()
        merged = deep_merge(merged, load_config(candidate, seen))
    return deep_merge(merged, current)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must be key=value, got {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot set nested key below non-mapping: {dotted_key}")
            cursor = child
        cursor[parts[-1]] = value
    return result

