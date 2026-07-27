"""Model settings derived from the single project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_sentry.configuration import load_config


DEFAULT_CONFIG = Path("config.json")


@dataclass(frozen=True)
class ModernBertPaths:
    repo_root: Path
    config_path: Path
    splits_dir: Path
    tokenizer_dir: Path
    checkpoint_dir: Path
    model_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class ModernBertRunConfig:
    raw: dict[str, Any]
    paths: ModernBertPaths

    label_column = "binary_label"
    label2id = {"benign": 0, "mining": 1}
    id2label = {0: "benign", 1: "mining"}

    @property
    def max_seq_length(self) -> int:
        return int(self.raw["max_seq_length"])

    @property
    def stride(self) -> int:
        return 1024


def load_run_config(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    data_dir: str | Path = "artifacts/data/training",
    model_dir: str | Path = "artifacts/model",
    work_dir: str | Path = "artifacts/training",
    reports_dir: str | Path = "artifacts/reports",
) -> ModernBertRunConfig:
    config = load_config(config_path)
    root = config.path.parent
    training = config.training
    model = config.model
    training_dir = _path(root, work_dir)
    paths = ModernBertPaths(
        repo_root=root,
        config_path=config.path,
        splits_dir=_path(root, data_dir) / "splits",
        tokenizer_dir=training_dir / "tokenizer",
        checkpoint_dir=training_dir / "checkpoints",
        model_dir=_path(root, model_dir),
        reports_dir=_path(root, reports_dir),
    )
    raw = {
        "max_seq_length": int(model["max_sequence_length"]),
        "tokenizer": {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "min_frequency": 1,
        },
        "modernbert": {
            "hidden_size": int(model["hidden_size"]),
            "num_hidden_layers": int(model["layers"]),
            "num_attention_heads": int(model["attention_heads"]),
            "intermediate_size": int(model["intermediate_size"]),
            "local_attention": int(model["local_attention"]),
            "classifier_pooling": "cls",
            "attention_dropout": float(model["attention_dropout"]),
            "embedding_dropout": float(model["embedding_dropout"]),
            "mlp_dropout": float(model["mlp_dropout"]),
            "classifier_dropout": float(model["classifier_dropout"]),
        },
        "mlm": _trainer_section(training, "mlm"),
        "classifier": _trainer_section(training, "classifier"),
        "decision": {
            "history_windows": int(config.decision["history_windows"]),
            "mean_mining_probability": float(config.decision["mean_mining_probability"]),
            "max_mining_probability": float(config.decision["max_mining_probability"]),
        },
    }
    return ModernBertRunConfig(raw=raw, paths=paths)


def ensure_output_dirs(config: ModernBertRunConfig) -> None:
    for path in (
        config.paths.tokenizer_dir,
        config.paths.checkpoint_dir,
        config.paths.model_dir,
        config.paths.reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _trainer_section(training: dict[str, Any], stage: str) -> dict[str, Any]:
    return {
        "epochs": int(training[f"{stage}_epochs"]),
        "learning_rate": float(training[f"{stage}_learning_rate"]),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 2 if stage == "classifier" else 1,
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "weight_decay": float(training[f"{stage}_weight_decay"]),
        "seed": int(training["seed"]),
    }


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
