from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and preserve its location for reproducibility."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    base_value = config.pop("_base_", None)
    if base_value is not None:
        base_path = Path(base_value)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base_config = load_config(base_path)
        base_config.pop("_config_path", None)
        config = _deep_merge(base_config, config)
    else:
        config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a config after removing runtime-only keys."""
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(serializable, handle, allow_unicode=True, sort_keys=False)


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = Path(base).resolve() if base is not None else Path.cwd().resolve()
    return (anchor / path).resolve()