from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    model: dict[str, Any]
    training: dict[str, Any]
    windowing: dict[str, Any]
    decision: dict[str, Any]
    collector: dict[str, Any]
    online: dict[str, Any]


def load_config(path: str | Path = "config.json") -> ProjectConfig:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {source}: {exc}") from exc

    expected = {"model", "training", "windowing", "decision", "collector", "online"}
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise ConfigError("; ".join(details))

    _positive(raw["model"], "max_sequence_length")
    _positive(raw["windowing"], "training_content_tokens")
    _positive(raw["windowing"], "deployment_content_tokens")
    if raw["windowing"]["training_content_tokens"] > raw["model"]["max_sequence_length"] - 2:
        raise ConfigError("training_content_tokens exceeds the model content window")
    if raw["windowing"]["deployment_content_tokens"] > raw["model"]["max_sequence_length"] - 2:
        raise ConfigError("deployment_content_tokens exceeds the model content window")
    _probability(raw["decision"], "mean_mining_probability")
    _probability(raw["decision"], "max_mining_probability")
    _positive(raw["decision"], "history_windows")
    policy = raw["decision"].get("policy")
    if not isinstance(policy, str) or ":" not in policy:
        raise ConfigError("decision.policy must use 'module:ClassName' syntax")
    _required_string(raw["collector"], "listen_address")
    _required_string(raw["collector"], "processor_socket")
    for key in (
        "frame_max_bytes",
        "launch_batch_size",
        "flush_interval_ms",
        "max_queued_launches",
    ):
        _positive(raw["collector"], key)
    _positive(raw["online"], "analysis_workers")
    _positive(raw["online"], "inference_workers")
    _required_string(raw["online"], "device")

    return ProjectConfig(path=source, **{key: raw[key] for key in expected})


def _positive(section: dict[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{key} must be positive")


def _probability(section: dict[str, Any], key: str) -> None:
    value = section.get(key)
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ConfigError(f"{key} must be in [0, 1]")


def _required_string(section: dict[str, Any], key: str) -> None:
    if not isinstance(section.get(key), str) or not section[key].strip():
        raise ConfigError(f"{key} must be a non-empty string")
