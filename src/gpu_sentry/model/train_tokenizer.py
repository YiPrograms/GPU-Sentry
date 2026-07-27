"""Train the SASS WordLevel tokenizer from the train split only."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, ensure_output_dirs, load_run_config
from .data import load_all_splits
from .metrics import write_json_report
from .tokenization import tokenizer_report, train_wordlevel_tokenizer


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
    run_config = load_run_config(
        args.config,
        data_dir=args.data,
        model_dir=args.model,
        work_dir=args.work_dir,
        reports_dir=args.reports,
    )
    ensure_output_dirs(run_config)
    records_by_split = load_all_splits(
        run_config.paths.splits_dir,
        run_config.paths.repo_root,
        run_config.label_column,
        run_config.label2id,
    )
    tokenizer = train_wordlevel_tokenizer(
        records_by_split["train"],
        run_config.raw["tokenizer"],
        run_config.max_seq_length,
        run_config.paths.tokenizer_dir,
    )
    report = tokenizer_report(tokenizer, records_by_split)
    write_json_report(run_config.paths.reports_dir / "tokenizer_report.json", report)
    print(f"Saved tokenizer to {run_config.paths.tokenizer_dir}")
    print(f"Saved tokenizer report to {run_config.paths.reports_dir / 'tokenizer_report.json'}")


if __name__ == "__main__":
    main()
