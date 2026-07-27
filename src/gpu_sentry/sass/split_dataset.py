#!/usr/bin/env python3
"""Create grouped train/validation/test split manifests for ModernBERT classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu_sentry.sass.splits import (
    DEFAULT_RATIOS,
    SplitError,
    load_workload_records,
    make_all_test_split,
    make_grouped_stratified_split,
    validate_splits,
    write_splits,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/splits"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="put every record in test.jsonl and leave train/val empty",
    )
    parser.add_argument(
        "--unique-score-sources",
        action="store_true",
        help=(
            "Use only first-seen dynamic-window signature rows when creating splits. "
            "This is intended for model training so repeated policy windows do not "
            "duplicate the same SASS input."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    try:
        records = load_workload_records(
            args.dataset_dir / "workloads",
            dataset_root=args.dataset_dir,
            unique_score_sources=args.unique_score_sources,
        )
        if args.all_test:
            ratios = {"train": 0.0, "val": 0.0, "test": 1.0}
            splits = make_all_test_split(records)
        else:
            splits = make_grouped_stratified_split(records, seed=args.seed, ratios=ratios)
        warnings = validate_splits(splits, records)
        manifest = write_splits(args.output_dir, splits, records, args.seed, ratios, warnings)
    except SplitError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    print(f"Wrote splits to {args.output_dir}")
    for split_name, summary in manifest["splits"].items():
        print(
            f"{split_name}: {summary['examples']} examples, "
            f"{summary['groups']} groups, labels={summary['labels']}, opts={summary['opt_levels']}"
        )
    for warning in warnings:
        print(f"[WARN] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
