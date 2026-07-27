#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gpu_sentry.experiments.runner import mixed_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mixed workloads against the behavioral baseline")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/experiments/baseline/model/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/mixed-baseline"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runtime", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return mixed_baseline(args, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
