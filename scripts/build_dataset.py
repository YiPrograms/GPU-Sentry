#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from gpu_sentry.sass.dataset import main as build_dataset
from gpu_sentry.sass.split_dataset import main as split_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and split a normalized SASS dataset")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--purpose", choices=("training", "evaluation"), required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--capture-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--content-token-budget", type=int)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    command = [
        "--config", str(args.config),
        "--purpose", args.purpose,
        "--capture-root", str(args.capture_root),
        "--output", str(args.output),
        "--jobs", str(args.jobs),
    ]
    for manifest in args.manifest:
        command.extend(("--manifest", manifest))
    if args.content_token_budget is not None:
        command.extend(("--content-token-budget", str(args.content_token_budget)))
    if args.replace:
        command.append("--replace")

    status = build_dataset(command)
    if status:
        return status
    split_mode = "--unique-score-sources" if args.purpose == "training" else "--all-test"
    return split_dataset([
        "--dataset-dir", str(args.output),
        "--output-dir", str(args.output / "splits"),
        split_mode,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
