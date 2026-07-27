"""The one supported stream-window configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from gpu_sentry.configuration import load_config

from .sass_tokens import CONTENT_TOKEN_BUDGET


class L0ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class L0RollingWindowConfig:
    content_token_budget: int


@dataclass(frozen=True)
class L0WindowConfig:
    window: L0RollingWindowConfig
    config_path: str

    def with_content_token_budget(self, content_token_budget: int) -> "L0WindowConfig":
        if not 0 < content_token_budget <= CONTENT_TOKEN_BUDGET:
            raise L0ConfigError(f"content token budget must be in [1, {CONTENT_TOKEN_BUDGET}]")
        return replace(self, window=replace(self.window, content_token_budget=int(content_token_budget)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_l0_config(path: str | Path = "config.json", *, training: bool = False) -> L0WindowConfig:
    config = load_config(path)
    key = "training_content_tokens" if training else "deployment_content_tokens"
    return L0WindowConfig(
        window=L0RollingWindowConfig(int(config.windowing[key])),
        config_path=str(config.path),
    )


def config_with_budget(content_token_budget: int) -> L0WindowConfig:
    return L0WindowConfig(L0RollingWindowConfig(content_token_budget), "<command line>").with_content_token_budget(
        content_token_budget
    )
