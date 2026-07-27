"""Masked-language-model warm-up for the SASS model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, ensure_output_dirs, load_run_config
from .data import ChunkDataset, load_all_splits, make_chunks
from .metrics import write_json_report
from .modeling import make_mlm_model, mlm_output_dir, training_args
from .tokenization import load_sass_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/training"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--reports", type=Path, default=Path("artifacts/reports/training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import DataCollatorForLanguageModeling, Trainer
    except ImportError as exc:
        raise RuntimeError("transformers is required for MLM warm-up") from exc

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

    train_chunks = make_chunks(
        records_by_split["train"], tokenizer, run_config.max_seq_length, run_config.stride
    )
    val_chunks = make_chunks(
        records_by_split["val"], tokenizer, run_config.max_seq_length, run_config.stride
    )
    output_dir = mlm_output_dir(run_config)
    model = make_mlm_model(run_config, tokenizer)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args(output_dir, run_config.raw["mlm"], "eval_loss", False),
        "train_dataset": ChunkDataset(train_chunks, include_labels=False),
        "eval_dataset": ChunkDataset(val_chunks, include_labels=False),
        "data_collator": collator,
    }
    trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        write_json_report(
            run_config.paths.reports_dir / "mlm_report.json",
            {
                "train_chunks": len(train_chunks),
                "val_chunks": len(val_chunks),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
            },
        )
        print(f"Saved MLM checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
