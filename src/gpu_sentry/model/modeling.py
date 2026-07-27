"""ModernBERT model/config helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def build_modernbert_config(
    run_config: Any,
    tokenizer: Any,
    num_labels: int | None = None,
):
    try:
        from transformers import ModernBertConfig
    except ImportError as exc:
        raise RuntimeError("transformers>=4.48.0 is required for ModernBERT") from exc

    model_kwargs = dict(run_config.raw["modernbert"])
    model_kwargs.update(
        {
            "vocab_size": len(tokenizer),
            "max_position_embeddings": run_config.max_seq_length,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.cls_token_id,
            "eos_token_id": tokenizer.sep_token_id,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
        }
    )
    model_kwargs["reference_compile"] = False
    if num_labels is not None:
        model_kwargs.update(
            {
                "num_labels": num_labels,
                "id2label": {int(idx): label for idx, label in run_config.id2label.items()},
                "label2id": {label: int(idx) for label, idx in run_config.label2id.items()},
                "problem_type": "single_label_classification",
            }
        )
    return ModernBertConfig(**model_kwargs)


def make_mlm_model(run_config: Any, tokenizer: Any):
    try:
        from transformers import ModernBertForMaskedLM
    except ImportError as exc:
        raise RuntimeError("transformers>=4.48.0 is required for ModernBERT") from exc
    return ModernBertForMaskedLM(build_modernbert_config(run_config, tokenizer))


def make_classifier_model(run_config: Any, tokenizer: Any, source_checkpoint: Path | None = None):
    try:
        from transformers import ModernBertForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("transformers>=4.48.0 is required for ModernBERT") from exc

    num_labels = len(run_config.label2id)
    model_config = build_modernbert_config(run_config, tokenizer, num_labels=num_labels)
    if source_checkpoint and (source_checkpoint / "config.json").exists():
        return ModernBertForSequenceClassification.from_pretrained(
            source_checkpoint,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
    return ModernBertForSequenceClassification(model_config)


def mlm_output_dir(run_config: Any) -> Path:
    return run_config.paths.checkpoint_dir / "mlm"


def classifier_output_dir(run_config: Any) -> Path:
    return run_config.paths.model_dir


def training_args(output_dir: Path, section: dict[str, Any], metric_name: str, greater_is_better: bool):
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is required for training arguments") from exc

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(section["epochs"]),
        "learning_rate": float(section["learning_rate"]),
        "per_device_train_batch_size": int(section["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(section["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(section["gradient_accumulation_steps"]),
        "warmup_ratio": float(section["warmup_ratio"]),
        "weight_decay": float(section["weight_decay"]),
        "logging_steps": 25,
        "save_total_limit": 2,
        "save_strategy": "epoch",
        "save_safetensors": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_name,
        "greater_is_better": greater_is_better,
        "seed": int(section["seed"]),
        "fp16": True,
        "report_to": [],
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
    }
    return TrainingArguments(eval_strategy="epoch", **kwargs)
