"""Evaluate the GPU-Sentry classifier with the deployment decision policy."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ensure_output_dirs, load_run_config
from .data import ChunkDataset, WorkloadRecord, load_all_splits, make_chunks, records_from_rows
from .metrics import (
    add_no_window_predictions,
    aggregate_chunk_probabilities,
    grouped_workload_metrics,
    softmax_rows,
    write_json_report,
)
from .modeling import classifier_output_dir
from .tokenization import load_sass_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--reports", type=Path, default=Path("artifacts/reports/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for evaluation") from exc

    config = load_run_config(
        args.config,
        data_dir=args.dataset_dir or "artifacts/data/training",
        model_dir=args.checkpoint or "artifacts/model",
        reports_dir=args.reports,
    )
    ensure_output_dirs(config)
    checkpoint = (args.checkpoint or classifier_output_dir(config)).resolve()
    tokenizer_path = checkpoint if (checkpoint / "tokenizer.json").exists() else config.paths.tokenizer_dir
    tokenizer = load_sass_tokenizer(tokenizer_path)
    records = load_records(args, config)
    inference_records = unique_sources(records)
    chunks = make_chunks(inference_records, tokenizer, config.max_seq_length, config.stride)

    model_config = AutoConfig.from_pretrained(checkpoint)
    model_config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=model_config)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(config.paths.reports_dir / ".evaluation"),
            per_device_eval_batch_size=int(config.raw["classifier"]["per_device_eval_batch_size"]),
            fp16=bool(config.raw["classifier"]["fp16"]) and torch.cuda.is_available(),
            report_to=[],
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    output = trainer.predict(ChunkDataset(chunks), metric_key_prefix=args.split)
    source_predictions = aggregate_chunk_probabilities(chunks, softmax_rows(output.predictions))
    predictions = expand_predictions(records, source_predictions)
    add_no_window_predictions(records, predictions)
    decision = config.raw["decision"]
    report = grouped_workload_metrics(
        records,
        predictions,
        config.id2label,
        history_windows=int(decision["history_windows"]),
        mean_threshold=float(decision["mean_mining_probability"]),
        max_threshold=float(decision["max_mining_probability"]),
    )
    report["source_inference_records"] = len(inference_records)
    report["dynamic_window_records"] = len(records)
    path = config.paths.reports_dir / f"evaluate_{args.split}_report.json"
    write_json_report(path, report)
    print(f"Saved evaluation report to {path}")


def load_records(args: argparse.Namespace, config: Any) -> list[WorkloadRecord]:
    if args.dataset_dir:
        dataset = args.dataset_dir.expanduser().resolve()
        from gpu_sentry.sass.splits import load_workload_records

        rows = load_workload_records(dataset / "workloads", dataset)
        return records_from_rows(
            rows,
            "test",
            dataset.parent,
            config.label_column,
            config.label2id,
            source_name=str(dataset / "workloads"),
        )
    split_records = load_all_splits(
        config.paths.splits_dir,
        config.paths.repo_root,
        config.label_column,
        config.label2id,
    )
    if args.split == "all":
        return [record for split in ("train", "val", "test") for record in split_records[split]]
    return split_records[args.split]


def unique_sources(records: list[WorkloadRecord]) -> list[WorkloadRecord]:
    output: list[WorkloadRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.row.get("no_l0_window"):
            continue
        key = str(record.source_path)
        if key not in seen:
            seen.add(key)
            output.append(replace(record, workload=key))
    return output


def expand_predictions(
    records: list[WorkloadRecord], source_predictions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.row.get("no_l0_window"):
            continue
        prediction = copy.deepcopy(source_predictions[str(record.source_path)])
        prediction.update({
            "workload": record.workload,
            "label": record.label,
            "label_id": record.label_id,
            "source_path": str(record.source_path),
        })
        output[record.workload] = prediction
    return output


if __name__ == "__main__":
    main()
