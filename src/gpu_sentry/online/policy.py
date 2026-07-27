from __future__ import annotations

import importlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class Decision:
    suspicious: bool
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class PolicyInput:
    session_id: str
    window_id: str
    mining_probability: float
    prediction: Mapping[str, Any]
    window_features: Mapping[str, Any]
    kernel_launches: tuple[Mapping[str, Any], ...]
    trigger_reason: tuple[str, ...]
    process_info: Mapping[str, Any]
    signature_cache_hit: bool


class DecisionPolicy(Protocol):
    def decide(self, observation: PolicyInput) -> Decision: ...


class RollingMeanMaxPolicy:
    """Terminate when both the rolling mean and maximum cross their thresholds."""

    def __init__(self, settings: Mapping[str, Any]):
        self.scores: deque[float] = deque(maxlen=int(settings["history_windows"]))
        self.mean_threshold = float(settings["mean_mining_probability"])
        self.max_threshold = float(settings["max_mining_probability"])

    def decide(self, observation: PolicyInput) -> Decision:
        self.scores.append(observation.mining_probability)
        mean = sum(self.scores) / len(self.scores)
        maximum = max(self.scores)
        suspicious = mean >= self.mean_threshold and maximum >= self.max_threshold
        return Decision(
            suspicious=suspicious,
            reason="rolling_mean_and_max" if suspicious else "below_rolling_thresholds",
            details={
                "policy": type(self).__name__,
                "observed_windows": len(self.scores),
                "history_windows": self.scores.maxlen,
                "mean_mining_probability": mean,
                "max_mining_probability": maximum,
                "mean_threshold": self.mean_threshold,
                "max_threshold": self.max_threshold,
            },
        )


def load_policy(path: str, settings: Mapping[str, Any]) -> DecisionPolicy:
    module_name, separator, class_name = path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("decision.policy must use 'module:ClassName' syntax")

    policy_class = getattr(importlib.import_module(module_name), class_name)
    policy = policy_class(settings)
    if not callable(getattr(policy, "decide", None)):
        raise TypeError(f"{path} must provide decide(observation)")
    return policy
