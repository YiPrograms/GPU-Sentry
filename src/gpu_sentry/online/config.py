"""Runtime settings derived from config.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_sentry.configuration import ConfigError, load_config


DEFAULT_ONLINE_CONFIG = Path("config.json")
OnlineConfigError = ConfigError


def load_online_config(
    path: str | Path = DEFAULT_ONLINE_CONFIG,
    model_dir: str | Path = "artifacts/model",
) -> dict[str, Any]:
    config = load_config(path)
    root = config.path.parent
    collector = config.collector
    online = config.online
    decision = config.decision
    model_path = repo_path(model_dir)
    return {
        "_config_path": str(config.path),
        "_root": str(root),
        "transport": {
            "processor_socket": collector["processor_socket"],
            "read_timeout_ms": 1000,
            "frame_max_bytes": collector["frame_max_bytes"],
        },
        "storage": {
            "processor_work_dir": str(root / "artifacts/online"),
        },
        "l0": {"config_path": str(config.path)},
        "kernel_analysis": {
            "workers": online["analysis_workers"],
            "short_kernel_threshold": config.windowing["short_kernel_instructions"],
            "readiness_timeout_ms": 30000,
        },
        "l1": {
            "training_config_path": str(config.path),
            "checkpoint_path": str(model_path),
            "device": online["device"],
            "inference_workers": online["inference_workers"],
            "batch_size": online["batch_size"],
        },
        "verdict": {
            "policy": decision["policy"],
            "history_windows": decision["history_windows"],
            "mean_mining_probability": decision["mean_mining_probability"],
            "max_mining_probability": decision["max_mining_probability"],
        },
        "enforcement": {"message": "[GPU-Sentry] Cryptomining detected. Terminating process."},
    }


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
