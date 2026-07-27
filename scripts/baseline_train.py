#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gpu_sentry.experiments.runner import baseline_train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the paper's behavioral baseline")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/baseline"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return baseline_train(args, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
