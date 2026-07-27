"""Online runtime settings derived from config.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_sentry.configuration import load_config


DEFAULT_ONLINE_CONFIG = Path("config.json")


@dataclass(frozen=True)
class OnlineConfig:
    config_path: Path
    processor_socket: Path
    frame_max_bytes: int
    work_dir: Path
    analysis_workers: int
    short_kernel_threshold: int
    checkpoint_path: Path
    device: str
    inference_workers: int
    policy: str
    decision_settings: dict[str, Any]


def load_online_config(
    path: str | Path = DEFAULT_ONLINE_CONFIG,
    model_dir: str | Path = "artifacts/model",
) -> OnlineConfig:
    config = load_config(path)
    root = config.path.parent
    model_path = Path(model_dir).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    return OnlineConfig(
        config_path=config.path,
        processor_socket=Path(config.collector["processor_socket"]),
        frame_max_bytes=int(config.collector["frame_max_bytes"]),
        work_dir=root / "artifacts/online",
        analysis_workers=int(config.online["analysis_workers"]),
        short_kernel_threshold=int(config.windowing["short_kernel_instructions"]),
        checkpoint_path=model_path.resolve(),
        device=str(config.online["device"]),
        inference_workers=int(config.online["inference_workers"]),
        policy=str(config.decision["policy"]),
        decision_settings=dict(config.decision),
    )
