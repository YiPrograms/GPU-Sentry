from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gpu_sentry.baseline.collector import RecordRequest, record_command
from gpu_sentry.baseline.features import build_point_features
from gpu_sentry.baseline.io import append_jsonl
from gpu_sentry.baseline.manifest import expand_manifest_patterns, load_manifest_rows
from gpu_sentry.baseline.model import score_points, train_point_random_forest
from gpu_sentry.configuration import load_config

from .presets import (
    BASELINE_MINERS,
    BASELINE_RATES,
    FAMILIES,
    GPU_SENTRY_MINERS,
    GPU_SENTRY_RATES,
    MIXED_RUNS,
    WINDOW_BUDGETS,
)


def run(args: argparse.Namespace, config_path: Path) -> int:
    if args.experiment == "window-budget":
        return window_budget(args, config_path)
    if args.experiment == "baseline-train":
        return baseline_train(args, config_path)
    if args.experiment == "throttle-gpu-sentry":
        return throttle_gpu_sentry(args, config_path)
    if args.experiment == "throttle-baseline":
        return throttle_baseline(args, config_path)
    if args.experiment == "mixed-baseline":
        return mixed_baseline(args, config_path)
    raise ValueError(args.experiment)


def window_budget(args: argparse.Namespace, config_path: Path) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    for budget in WINDOW_BUDGETS:
        data = args.output / f"budget-{budget}" / "data"
        reports = args.output / f"budget-{budget}" / "reports"
        build = _script(config_path, "build_dataset.py") + [
            "--config", str(config_path), "--purpose", "evaluation",
            "--capture-root", str(args.capture_root), "--output", str(data),
            "--content-token-budget", str(budget), "--jobs", str(args.jobs), "--replace",
        ]
        for manifest in args.manifest:
            build.extend(["--manifest", manifest])
        evaluate = _script(config_path, "evaluate.py") + [
            "--config", str(config_path), "--dataset-dir", str(data),
            "--checkpoint", str(args.model), "--split", "all", "--reports", str(reports),
        ]
        _run_or_print(build, args.dry_run)
        _run_or_print(evaluate, args.dry_run)
    if args.dry_run:
        return 0
    summaries = []
    for budget in WINDOW_BUDGETS:
        report = json.loads((args.output / f"budget-{budget}/reports/evaluate_all_report.json").read_text())
        mining = []
        benign = []
        for row in report.get("predictions", []):
            score = float(row.get("rolling_mean_mining_probability_max", row.get("mining_probability_mean", 0.0)))
            (mining if row.get("label") == "mining" else benign).append(score)
        q5 = percentile(mining, 5)
        q95 = percentile(benign, 95)
        summaries.append({
            "budget": budget,
            "mining_q5": q5,
            "benign_q95": q95,
            "tail_separation": min(q5 - 0.5, 0.5 - q95),
        })
    selected = max(summaries, key=lambda row: (round(row["tail_separation"], 3), row["budget"]))
    write_json(args.output / "selection.json", {"budgets": summaries, "selected_budget": selected["budget"]})
    return 0


def baseline_train(args: argparse.Namespace, config_path: Path) -> int:
    raw = args.output / "raw"
    rows = load_manifest_rows(expand_manifest_patterns(args.manifest))
    for row in rows:
        command = [str(item) for item in row.get("argv") or []]
        if not command:
            continue
        print("+", " ".join(command))
        if args.dry_run:
            continue
        metadata = record_command(RecordRequest(
            command=command,
            output_dir=raw,
            gpu=args.gpu,
            workload=str(row["workload"]),
            label=str(row.get("label") or row["binary_label"]),
            binary_label=str(row["binary_label"]),
            family=row.get("family"),
            program=row.get("program"),
            variant=row.get("variant"),
            cwd=Path(row.get("cwd") or "."),
            timeout_s=float(row.get("runtime_sec") or 60) + 90,
        ))
        append_jsonl(raw / "manifest.jsonl", metadata)
    if args.dry_run:
        return 0
    features = args.output / "features/points.csv"
    build_point_features(raw, features)
    seed = int(load_config(config_path).training["seed"])
    train_point_random_forest(features, args.output / "model", seed=seed, max_run_fpr=0.01)
    return 0


def throttle_baseline(args: argparse.Namespace, config_path: Path) -> int:
    root = config_path.parent
    raw = args.output / "raw"
    sleep = calibrate(root, BASELINE_MINERS, args.gpu, args.dry_run)
    for repeat in range(1, args.repeats + 1):
        for miner in BASELINE_MINERS:
            for rate in BASELINE_RATES:
                command = miner_command(root, miner, args.runtime, sleep[miner][rate], check=not args.dry_run)
                workload = f"{miner}_o3_{rate}pct_r{repeat}"
                print("+", " ".join(command))
                if args.dry_run:
                    continue
                metadata = record_command(RecordRequest(
                    command=command, output_dir=raw, gpu=args.gpu, workload=workload,
                    label="mining_like", binary_label="mining", family=FAMILIES[miner],
                    program=miner, variant="o3", cwd=root / "workloads/synthetic/scripts",
                    timeout_s=args.runtime + 90,
                    extra_metadata={"target_percent": rate, "repeat": repeat},
                ))
                append_jsonl(raw / "manifest.jsonl", metadata)
    if args.dry_run:
        return 0
    points = args.output / "features/points.csv"
    build_point_features(raw, points)
    score_points(points, args.model, args.output / "score")
    return 0


def throttle_gpu_sentry(args: argparse.Namespace, config_path: Path) -> int:
    config = load_config(config_path)
    root = config.path.parent
    capture_root = root / "artifacts/captures"
    if not args.dry_run and not capture_root.is_dir():
        raise RuntimeError("start `python scripts/deploy.py` before the GPU-Sentry throttle experiment")
    sleep = calibrate(root, GPU_SENTRY_MINERS, args.gpu, args.dry_run)
    manifest = args.output / "manifest.jsonl"
    for repeat in range(1, args.repeats + 1):
        for miner in GPU_SENTRY_MINERS:
            for rate in GPU_SENTRY_RATES:
                command = miner_command(root, miner, args.runtime, sleep[miner][rate], check=not args.dry_run)
                print("+", " ".join(command))
                if args.dry_run:
                    continue
                before = {path.name for path in capture_root.iterdir() if path.is_dir()}
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                env["GPU_SENTRY_SERVER_ADDR"] = config.collector["listen_address"]
                env.pop("GPU_SENTRY_DISABLE", None)
                library = str((root / "native/build").resolve())
                env["LD_LIBRARY_PATH"] = library + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
                subprocess.run(command, cwd=root / "workloads/synthetic/scripts", env=env, check=True)
                captures = wait_for_captures(capture_root, before)
                for capture in captures:
                    append_jsonl(manifest, {
                        "workload": f"{miner}_o3_{rate}pct_r{repeat}",
                        "program": miner, "family": FAMILIES[miner], "binary_label": "mining",
                        "label": "mining_like", "target_percent": rate, "repeat": repeat,
                        "capture_path": str(capture.relative_to(root)),
                    })
    return 0


def mixed_baseline(args: argparse.Namespace, config_path: Path) -> int:
    root = config_path.parent
    miners = tuple(dict.fromkeys(miner for _benign, miner, _rate in MIXED_RUNS))
    sleep = calibrate(root, miners, args.gpu, args.dry_run)
    raw = args.output / "raw"
    for index, (benign, miner, rate) in enumerate(MIXED_RUNS, 1):
        binary = root / f"workloads/synthetic/binaries/{miner}_o3"
        command = [
            sys.executable, "-m", "gpu_sentry.experiments.mixed_worker",
            "--benign", benign, "--miner", str(binary), "--sleep-us", str(sleep[miner][rate]),
            "--runtime", str(args.runtime),
        ]
        workload = f"mixed_{index}_{benign}_{miner}_{rate}pct"
        print("+", " ".join(command))
        if args.dry_run:
            continue
        metadata = record_command(RecordRequest(
            command=command, output_dir=raw, gpu=args.gpu, workload=workload,
            label="mixed_benign_mining", binary_label="mining", family="mixed",
            program="mixed_baseline", variant=f"{rate}pct", cwd=root,
            timeout_s=args.runtime + 90, extra_metadata={"target_percent": rate},
        ))
        append_jsonl(raw / "manifest.jsonl", metadata)
    if args.dry_run:
        return 0
    points = args.output / "features/points.csv"
    build_point_features(raw, points)
    score_points(points, args.model, args.output / "score")
    return 0


def calibrate(root: Path, miners: tuple[str, ...], gpu: int, dry_run: bool) -> dict[str, dict[int, int]]:
    rates = set(GPU_SENTRY_RATES) | set(BASELINE_RATES)
    if dry_run:
        return {miner: {rate: 0 for rate in rates} for miner in miners}
    result = {}
    for miner in miners:
        command = miner_command(root, miner, 15, 0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["GPU_SENTRY_DISABLE"] = "1"
        pilot = subprocess.run(command, cwd=root / "workloads/synthetic/scripts", env=env, text=True, capture_output=True, check=True)
        match = re.search(r"^total_launches=(\d+)$", pilot.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError(f"{miner} did not report total_launches")
        launches_per_second = int(match.group(1)) / 15
        result[miner] = {
            rate: 0 if rate == 100 else max(1, round(1_000_000 * ((100 / rate) - 1) / launches_per_second))
            for rate in rates
        }
    return result


def miner_command(root: Path, miner: str, runtime: int, sleep_us: int, *, check: bool = True) -> list[str]:
    binary = root / f"workloads/synthetic/binaries/{miner}_o3"
    if check and not binary.exists():
        raise FileNotFoundError(f"build synthetic workloads first: {binary}")
    return [str(binary), str(runtime), "--sleep-between-launches-us", str(sleep_us)]


def wait_for_captures(root: Path, before: set[str], timeout: float = 30) -> list[Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = [path for path in root.iterdir() if path.is_dir() and path.name not in before and (path / "events.jsonl").exists()]
        if found:
            return sorted(found)
        time.sleep(0.25)
    raise RuntimeError("collector produced no capture")


def percentile(values: list[float], percent: int) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from an empty class")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _script(config: Path, name: str) -> list[str]:
    return [sys.executable, str(config.parent / "scripts" / name)]


def _run_or_print(command: list[str], dry_run: bool) -> None:
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)
