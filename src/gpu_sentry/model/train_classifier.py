"""Train the ModernBERT SASS classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG, ensure_output_dirs, load_run_config
from .data import ChunkDataset, class_weights, load_all_splits, make_chunks
from .metrics import softmax_rows, workload_metrics, write_json_report
from .modeling import classifier_output_dir, make_classifier_model, mlm_output_dir, training_args
from .tokenization import load_sass_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/training"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--reports", type=Path, default=Path("artifacts/reports/training"))
    parser.add_argument(
        "--mlm-checkpoint",
        type=Path,
        default=None,
        help="Optional MLM checkpoint to initialize from. Defaults to config checkpoint_dir/mlm/final.",
    )
    return parser.parse_args()


def make_trainer(
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    weights: list[float],
    workload_records: Any,
    workload_chunks: Any,
    id2label: Any,
    decision: dict[str, Any],
):
    try:
        import torch
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for classifier training") from exc

    class WorkloadTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                weight=self.class_weights.to(logits.device),
            )
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval"):
            metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
            if eval_dataset is not None:
                return metrics
            predictions = self.predict(self.eval_dataset, metric_key_prefix=f"{metric_key_prefix}_chunks")
            probabilities = softmax_rows(predictions.predictions)
            workload_report = workload_metrics(
                workload_records,
                workload_chunks,
                probabilities,
                id2label,
                history_windows=int(decision["history_windows"]),
                mean_threshold=float(decision["mean_mining_probability"]),
                max_threshold=float(decision["max_mining_probability"]),
            )
            for name in ("accuracy", "macro_f1", "micro_f1", "weighted_f1"):
                metrics[f"{metric_key_prefix}_{name}"] = workload_report[name]
            return metrics

    trainer = WorkloadTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.class_weights = torch.tensor(weights, dtype=torch.float)
    return trainer


def main() -> None:
    args = parse_args()
    try:
        from transformers import DataCollatorWithPadding
    except ImportError as exc:
        raise RuntimeError("transformers is required for classifier training") from exc

    run_config = load_run_config(
        args.config,
        data_dir=args.data,
        model_dir=args.model,
        work_dir=args.work_dir,
        reports_dir=args.reports,
    )
    ensure_output_dirs(run_config)
    tokenizer = load_sass_tokenizer(run_config.paths.tokenizer_dir)
    records_by_split = load_all_splits(
        run_config.paths.splits_dir,
        run_config.paths.repo_root,
        run_config.label_column,
        run_config.label2id,
    )
    chunks_by_split = {
        split: make_chunks(records, tokenizer, run_config.max_seq_length, run_config.stride)
        for split, records in records_by_split.items()
    }

    mlm_checkpoint = args.mlm_checkpoint or mlm_output_dir(run_config)
    model = make_classifier_model(run_config, tokenizer, source_checkpoint=mlm_checkpoint)
    weights = class_weights(records_by_split["train"], run_config.label2id)
    output_dir = classifier_output_dir(run_config)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    decision = run_config.raw["decision"]
    trainer = make_trainer(
        model=model,
        args=training_args(output_dir, run_config.raw["classifier"], "eval_macro_f1", True),
        train_dataset=ChunkDataset(chunks_by_split["train"]),
        eval_dataset=ChunkDataset(chunks_by_split["val"]),
        data_collator=collator,
        tokenizer=tokenizer,
        weights=weights,
        workload_records=records_by_split["val"],
        workload_chunks=chunks_by_split["val"],
        id2label=run_config.id2label,
        decision=decision,
    )

    train_result = trainer.train()
    final_output = output_dir
    if trainer.is_world_process_zero():
        trainer.save_model(final_output)
        tokenizer.save_pretrained(final_output)

    reports = {
        "train_chunks": len(chunks_by_split["train"]),
        "val_chunks": len(chunks_by_split["val"]),
        "test_chunks": len(chunks_by_split["test"]),
        "class_weights": weights,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "splits": {},
    }
    for split in ("val", "test"):
        predictions = trainer.predict(ChunkDataset(chunks_by_split[split]), metric_key_prefix=split)
        probabilities = softmax_rows(predictions.predictions)
        split_report = workload_metrics(
            records_by_split[split],
            chunks_by_split[split],
            probabilities,
            run_config.id2label,
            history_windows=int(decision["history_windows"]),
            mean_threshold=float(decision["mean_mining_probability"]),
            max_threshold=float(decision["max_mining_probability"]),
        )
        if trainer.is_world_process_zero():
            reports["splits"][split] = split_report
            write_json_report(run_config.paths.reports_dir / f"classifier_{split}_report.json", split_report)

    if trainer.is_world_process_zero():
        write_json_report(run_config.paths.reports_dir / "classifier_report.json", reports)
        print(f"Saved classifier checkpoint to {final_output}")
        print(f"Saved reports to {run_config.paths.reports_dir}")


if __name__ == "__main__":
    main()
