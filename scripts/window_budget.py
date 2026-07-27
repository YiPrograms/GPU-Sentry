#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gpu_sentry.experiments.runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the paper's window-budget experiment")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--capture-root", type=Path, default=Path("."))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/window-budget"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.experiment = "window-budget"
    return run(args, args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
