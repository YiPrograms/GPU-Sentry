#!/usr/bin/env python3
"""Generate a capture-manifest JSONL for synthetic kernel captures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from gpu_sentry.sass.ingest import read_process, workload_name_from_process
from gpu_sentry.sass.manifest import ManifestError, load_binary_manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, default=Path("artifacts/captures"))
    parser.add_argument(
        "--binary-manifest",
        type=Path,
        default=Path("workloads/synthetic/binaries/manifest.jsonl"),
        help="Synthetic binary-label manifest used only to generate capture-manifest rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/captures/manifest.jsonl"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        binary_manifest = load_binary_manifest(args.binary_manifest)
    except ManifestError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    if not args.captures_dir.is_dir():
        print(f"[FATAL] missing captures directory: {args.captures_dir}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for capture_dir in sorted(path for path in args.captures_dir.iterdir() if path.is_dir()):
        process = read_process(capture_dir)
        workload = workload_name_from_process(process)
        if workload not in binary_manifest:
            print(f"[FATAL] missing manifest entry for workload: {workload}", file=sys.stderr)
            return 2
        event_summary = summarize_events(capture_dir / "events.jsonl")
        rows.append(build_row(capture_dir, process, event_summary, binary_manifest[workload], args.repo_root))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, sort_keys=True)
            fh.write("\n")

    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


def build_row(
    capture_dir: Path,
    process: dict[str, Any],
    event_summary: dict[str, Any],
    manifest_entry: dict[str, str],
    repo_root: Path,
) -> dict[str, Any]:
    workload = workload_name_from_process(process)
    program, variant = split_synthetic_workload(workload)

    row = {
        "argv": process.get("argv", []),
        "binary_label": "mining" if manifest_entry["label"] == "mining_like" else "benign",
        "capture_id": capture_dir.name,
        "capture_path": relative_path(capture_dir, repo_root),
        "code_file_count": event_summary["code_file_count"],
        "code_files": event_summary["code_files"],
        "code_total_bytes": event_summary["code_total_bytes"],
        "cwd": process.get("cwd"),
        "device_pci_bus_ids": event_summary["device_pci_bus_ids"],
        "event_count": event_summary["event_count"],
        "event_type_counts": event_summary["event_type_counts"],
        "exe_path": process.get("exe_path"),
        "family": manifest_entry["family"],
        "hostname": process.get("hostname"),
        "label": manifest_entry["label"],
        "observed_duration_s": event_summary["observed_duration_s"],
        "opt_level": manifest_entry["opt_level"],
        "pid": process.get("pid"),
        "process_received_at": process.get("received_at"),
        "program": program,
        "sample_kernel_names": event_summary["sample_kernel_names"],
        "synthetic": True,
        "variant": variant,
        "workload": workload,
    }
    return {key: value for key, value in row.items() if value is not None}


def summarize_events(path: Path) -> dict[str, Any]:
    event_type_counts: Counter[str] = Counter()
    code_files: list[str] = []
    code_total_bytes = 0
    device_pci_bus_ids: set[str] = set()
    sample_kernel_names: list[str] = []
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    event_count = 0

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            event_count += 1
            event_type = str(event.get("type"))
            event_type_counts[event_type] += 1
            timestamp = event.get("timestamp_ns")
            if isinstance(timestamp, int):
                min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
                max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)

            if event_type == "code":
                if event.get("path"):
                    code_files.append(str(event["path"]))
                code_total_bytes += int(event.get("size") or 0)
            elif event_type == "kernel_launch":
                if event.get("device_pci_bus_id"):
                    device_pci_bus_ids.add(str(event["device_pci_bus_id"]))
                name = event.get("kernel_name")
                if name and name not in sample_kernel_names and len(sample_kernel_names) < 12:
                    sample_kernel_names.append(str(name))

    timestamps = []
    if min_timestamp is not None:
        timestamps.append(min_timestamp)
    if max_timestamp is not None:
        timestamps.append(max_timestamp)
    return {
        "code_file_count": event_type_counts["code"],
        "code_files": code_files,
        "code_total_bytes": code_total_bytes,
        "device_pci_bus_ids": sorted(device_pci_bus_ids),
        "event_count": event_count,
        "event_type_counts": dict(event_type_counts),
        "observed_duration_s": observed_duration_s(timestamps),
        "sample_kernel_names": sample_kernel_names,
    }


def split_synthetic_workload(workload: str) -> tuple[str, str]:
    lower = workload.lower()
    for opt_suffix in ("_o2", "_o3"):
        if lower.endswith(opt_suffix):
            return workload[: -len(opt_suffix)], workload[-2:].lower()
    return workload, "default"


def observed_duration_s(timestamps: list[int]) -> float | None:
    if len(timestamps) < 2:
        return None
    return round((max(timestamps) - min(timestamps)) / 1_000_000_000, 6)


def relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
