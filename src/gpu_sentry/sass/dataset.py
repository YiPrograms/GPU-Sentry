#!/usr/bin/env python3
"""Build a GPU-Sentry dataset from CUDA capture directories."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from gpu_sentry.configuration import ConfigError, load_config
from gpu_sentry.sass.cfg import build_cfg_for_kernel
from gpu_sentry.sass.disassemble import (
    DisassemblyError,
    disassemble_code_objects,
    find_cuda_tools,
    write_extraction_report,
)
from gpu_sentry.sass.ingest import (
    IngestError,
    copy_code_objects,
    read_events,
    read_jsonl,
    split_events,
    write_launches,
)
from gpu_sentry.sass.l0_config import L0ConfigError, load_l0_config
from gpu_sentry.sass.l0_windows import L0Window, build_l0_windows
from gpu_sentry.sass.loop_extract import extract_main_loop_for_kernel
from gpu_sentry.sass.manifest import (
    ManifestError,
    write_json,
    write_workload_manifest,
)
from gpu_sentry.sass.normalize import normalize_kernel_files
from gpu_sentry.sass.split_kernels import split_launched_kernels
from gpu_sentry.sass.workload_sass import (
    WorkloadSassError,
    prepare_l0_launches,
    render_workload_sass,
)


class CaptureBuildError(RuntimeError):
    """Raised when one capture cannot produce a workload."""


OPT_LEVEL_CAPTURE = "capture"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        dest="capture_manifest",
        action="append",
        required=True,
        help="Capture manifest path or glob; repeat for multiple sources.",
    )
    parser.add_argument(
        "--capture-root",
        dest="capture_root",
        type=Path,
        default=Path("."),
        help="Root used to resolve relative capture_path entries from capture manifests.",
    )
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--purpose", choices=("training", "evaluation"), required=True)
    parser.add_argument(
        "--content-token-budget",
        dest="content_token_budget",
        type=int,
        help="Override the canonical budget for a window-budget experiment.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of capture worker processes to run in parallel.",
    )
    parser.add_argument("--replace", action="store_true", help="Replace existing workload output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    capture_manifest_patterns = list(args.capture_manifest)

    try:
        project_config = load_config(args.config)
        capture_specs = load_capture_manifest_specs(
            capture_manifest_patterns,
            captures_root=args.capture_root,
        )
        l0_config = load_l0_config(args.config, training=args.purpose == "training")
        if args.content_token_budget is not None:
            l0_config = l0_config.with_content_token_budget(args.content_token_budget)
        args.l0_config_obj = l0_config
        args.short_kernel_threshold = int(project_config.windowing["short_kernel_instructions"])
    except (ManifestError, ConfigError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    except L0ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    tools = find_cuda_tools()
    report = new_build_report()
    report["l0_config"] = args.l0_config_obj.to_dict()
    if args.jobs < 1:
        print("[FATAL] --jobs must be >= 1", file=sys.stderr)
        return 2

    total_captures = len(capture_specs)
    results = iter_capture_results(capture_specs, args, tools)
    for result in results:
        report["captures_scanned"] += 1
        progress = f"[{report['captures_scanned']}/{total_captures}]"
        if result["status"] == "failed":
            report["failed_captures"] += 1
            report["failures"].append({"capture": result["capture"], "reason": result["reason"]})
            print(f"{progress} [ERROR] {result['capture']}: {result['reason']}", flush=True)
            continue

        if result["status"] == "skipped":
            report["duplicates_skipped"] += 1
            if result.get("label"):
                report["labels"][result["label"]] += 1
            if result.get("opt_level"):
                report["opt_levels"][result["opt_level"]] += 1
            print(f"{progress} [SKIP] duplicate workload already exists: {result['workload']}", flush=True)
            continue
        if result["status"] == "skipped_empty":
            report["empty_captures_skipped"] += 1
            print(f"{progress} [SKIP] {result['capture']}: {result['reason']}", flush=True)
            continue
        if result["status"] == "skipped_unmapped":
            report["unmapped_captures_skipped"] += 1
            print(f"{progress} [SKIP] {result['capture']}: {result['reason']}", flush=True)
            continue
        if result["status"] == "skipped_no_windows":
            report["no_l0_windows_skipped"] += 1
            if result.get("label"):
                report["labels"][result["label"]] += 0
            print(f"{progress} [SKIP] {result['capture']}: {result['reason']}", flush=True)
            continue
        created = int(result.get("workloads_created", 1))
        report["workloads_created"] += created
        report["windows_created"] += int(result.get("windows_created", 0))
        report["no_l0_window_rows_created"] += int(result.get("no_l0_window_rows_created", 0))
        report["labels"][result["label"]] += created
        report["opt_levels"][result["opt_level"]] += created
        windows_created = int(result.get("windows_created", 0))
        if windows_created:
            suffix = f" ({windows_created} windows)"
        elif result.get("no_l0_window_rows_created"):
            suffix = " (no L0 window, default benign)"
        else:
            suffix = ""
        print(f"{progress} [OK] {result['workload']}{suffix}", flush=True)

    write_report(args.output_dir, report)
    print_summary(report)
    return 0 if report["failed_captures"] == 0 else 1


def iter_capture_results(
    capture_specs: list[dict[str, Any]],
    args: argparse.Namespace,
    tools: dict[str, Path],
):
    if args.jobs == 1 or len(capture_specs) <= 1:
        for capture_spec in capture_specs:
            yield process_capture_worker(capture_spec, args, tools)
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(_worker_star, (capture_spec, args, tools))
            for capture_spec in capture_specs
        ]
        for future in as_completed(futures):
            yield future.result()


def _worker_star(payload: tuple[dict[str, Any], argparse.Namespace, dict[str, Path]]):
    capture_spec, args, tools = payload
    return process_capture_worker(capture_spec, args, tools)


def process_capture_worker(
    capture_spec: dict[str, Any],
    args: argparse.Namespace,
    tools: dict[str, Path],
) -> dict[str, Any]:
    capture_dir = Path(capture_spec["capture_dir"])
    try:
        result = process_capture(capture_spec, args, tools)
        result["capture"] = capture_dir.name
        return result
    except CaptureBuildError as exc:
        return {"status": "failed", "capture": capture_dir.name, "reason": str(exc)}
    except Exception as exc:
        return {
            "status": "failed",
            "capture": capture_dir.name,
            "reason": f"unexpected {type(exc).__name__}: {exc}",
        }


def process_capture(
    capture_spec: dict[str, Any],
    args: argparse.Namespace,
    tools: dict[str, Path],
) -> dict[str, Any]:
    capture_dir = Path(capture_spec["capture_dir"])
    try:
        if capture_spec_has_no_kernel_launch(capture_spec):
            remove_existing_workload_if_overwriting(args, str(capture_spec["workload"]))
            return {
                "status": "skipped_empty",
                "workload": str(capture_spec["workload"]),
                "reason": "no kernel_launch event",
            }

        workload = str(capture_spec["workload"])
        manifest_entry = dict(capture_spec["manifest_entry"])
        l0_config = args.l0_config_obj

        workloads_root = args.output_dir / "workloads"
        final_dir = workloads_root / workload
        if final_dir.exists() and not args.replace:
            result = {"status": "skipped", "workload": workload}
            existing_manifest_path = final_dir / "manifest.json"
            if existing_manifest_path.exists():
                existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
                result["label"] = existing_manifest.get("label")
                result["opt_level"] = existing_manifest.get("opt_level")
            return result
        workloads_root.mkdir(parents=True, exist_ok=True)
        temp_dir = workloads_root / f".{workload}.tmp.{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        try:
            created = _build_l0_windows_from_capture(
                capture_dir, temp_dir, workload, args, manifest_entry, tools
            )
            if created == 0:
                _materialize_no_l0_window_row(temp_dir, workload, manifest_entry, args)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.replace(final_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "status": "created",
            "workload": workload,
            "label": manifest_entry["label"],
            "opt_level": manifest_entry["opt_level"],
            "workloads_created": 1,
            "windows_created": created,
            "no_l0_window_rows_created": 1 if created == 0 else 0,
        }
    except IngestError as exc:
        if str(exc) == "no kernel_launch event":
            remove_existing_workload_if_overwriting(args, str(capture_spec["workload"]))
            return {
                "status": "skipped_empty",
                "workload": str(capture_spec["workload"]),
                "reason": str(exc),
            }
        raise CaptureBuildError(str(exc)) from exc
    except WorkloadSassError as exc:
        if str(exc) == "no launched kernel could be mapped to disassembled SASS":
            remove_existing_workload_if_overwriting(args, str(capture_spec["workload"]))
            return {
                "status": "skipped_unmapped",
                "workload": str(capture_spec["workload"]),
                "reason": str(exc),
            }
        raise CaptureBuildError(str(exc)) from exc
    except (DisassemblyError, CaptureBuildError) as exc:
        raise CaptureBuildError(str(exc)) from exc


def remove_existing_workload_if_overwriting(args: argparse.Namespace, workload: str) -> None:
    if not args.replace:
        return
    shutil.rmtree(args.output_dir / "workloads" / workload, ignore_errors=True)


def _build_capture_base(
    capture_dir: Path,
    workload_dir: Path,
    tools: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = read_events(capture_dir)
    code_events, launch_events = split_events(events)
    code_map = copy_code_objects(capture_dir, code_events, workload_dir)
    launches = write_launches(workload_dir, launch_events)
    extraction_report = disassemble_code_objects(workload_dir, code_map, tools)

    kernel_dirs, missing_kernels = split_launched_kernels(
        workload_dir, launches, code_map, extraction_report
    )
    extraction_report["missing_kernels"] = missing_kernels
    write_extraction_report(workload_dir, extraction_report)
    if not kernel_dirs:
        raise WorkloadSassError("no launched kernel could be mapped to disassembled SASS")

    for kernel_dir in kernel_dirs.values():
        cfg = build_cfg_for_kernel(kernel_dir)
        extract_main_loop_for_kernel(kernel_dir, cfg)
        normalize_kernel_files(kernel_dir)

    return launches, extraction_report


def _build_l0_windows_from_capture(
    capture_dir: Path,
    workload_dir: Path,
    parent_workload: str,
    args: argparse.Namespace,
    manifest_entry: dict[str, str],
    tools: dict[str, Path],
) -> int:
    (workload_dir / "dumps").mkdir(parents=True, exist_ok=True)
    (workload_dir / "kernels").mkdir(parents=True, exist_ok=True)
    write_workload_manifest(workload_dir / "manifest.json", parent_workload, manifest_entry)
    launches, extraction_report = _build_capture_base(capture_dir, workload_dir, tools)
    fragment_cache: dict[tuple[Any, ...], Any] = {}
    prepared = prepare_l0_launches(
        workload_dir,
        launches,
        short_kernel_threshold=args.short_kernel_threshold,
        content_token_budget=args.l0_config_obj.window.content_token_budget,
        fragment_cache=fragment_cache,
    )
    launches = prepared["launches"]
    extraction_report["l0_missing_launches"] = prepared["missing_launches"]
    extraction_report["l0_ready_launches"] = len(launches)
    extraction_report["l0_original_launches"] = len(read_jsonl(workload_dir / "launches.jsonl"))
    extraction_report["l0_logical_launches"] = prepared.get("logical_launch_count", len(launches))
    extraction_report["l0_unique_kernel_count"] = prepared.get("unique_kernel_count", 0)
    extraction_report["l0_unique_segment_count"] = prepared.get("unique_segment_count", 0)
    extraction_report["l0_max_bitwise_integer_ratio"] = prepared.get("max_bitwise_integer_ratio", 0.0)
    extraction_report["l0_gate_short_circuit"] = bool(prepared.get("gate_short_circuit", False))
    windows = build_l0_windows(launches, args.l0_config_obj)
    if not windows:
        extraction_report["l0_windows"] = {
            "count": 0,
            "config_path": args.l0_config_obj.config_path,
            "resolved_config": args.l0_config_obj.to_dict(),
            "missing_launches": prepared["missing_launches"],
        }
        write_extraction_report(workload_dir, extraction_report)
        return 0

    windows_dir = workload_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    signature_sources: dict[str, dict[str, Any]] = {}
    created = 0
    manifest_path = windows_dir / "manifests.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_fh:
        for window in windows:
            if args.purpose == "training" and not bool(
                window.features.get("signature_first_seen", True)
            ):
                continue
            row = _materialize_l0_window(
                workload_dir,
                windows_dir,
                safe_workload_name(f"{parent_workload}__{window.window_id}"),
                parent_workload,
                manifest_entry,
                window,
                args,
                fragment_cache,
                signature_sources,
            )
            json.dump(row, manifest_fh, sort_keys=True)
            manifest_fh.write("\n")
            created += 1

    extraction_report["l0_windows"] = {
        "count": len(windows),
        "manifest_rows": created,
        "config_path": args.l0_config_obj.config_path,
        "resolved_config": args.l0_config_obj.to_dict(),
        "compact_manifest": True,
        "first_seen_windows_only": args.purpose == "training",
    }
    write_extraction_report(workload_dir, extraction_report)
    validate_l0_workload(workload_dir)
    return created


def _materialize_l0_window(
    workload_dir: Path,
    windows_dir: Path,
    window_workload: str,
    parent_workload: str,
    manifest_entry: dict[str, str],
    window: L0Window,
    args: argparse.Namespace,
    fragment_cache: dict[tuple[Any, ...], Any] | None = None,
    signature_sources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    signature_sources = signature_sources if signature_sources is not None else {}
    sass_name = f"{window.window_id}.sass"
    signature_key = str(window.features.get("composition_signature_key") or "")
    signature_hash = signature_key_hash(signature_key)
    source_key = _signature_source_key(window)
    source = signature_sources.get(source_key)
    if source is None:
        rendered = render_workload_sass(
            workload_dir,
            window.launches,
            short_kernel_threshold=args.short_kernel_threshold,
            content_token_budget=args.l0_config_obj.window.content_token_budget,
            fragment_cache=fragment_cache,
        )
        if int(rendered["token_cost"]) > int(args.l0_config_obj.window.content_token_budget):
            raise WorkloadSassError(
                f"L0 window {window.window_id} exceeds token budget: "
                f"{rendered['token_cost']} > {args.l0_config_obj.window.content_token_budget}"
            )
        (windows_dir / sass_name).write_text(rendered["text"], encoding="utf-8")
        source = {
            "path": f"windows/{sass_name}",
            "workload": window_workload,
            "window_id": window.window_id,
            "rendered_token_cost": rendered["token_cost"],
            "front_clipped": rendered["front_clipped"],
            "clipped_token_count": rendered["clipped_token_count"],
        }
        if signature_key:
            signature_sources[source_key] = source
        signature_first_seen = True
    else:
        rendered = {
            "token_cost": source.get("rendered_token_cost", 0),
            "missing_launches": [],
            "included_launches": len(window.launches),
            "pre_clip_token_cost": window.features.get("pre_clip_token_cost", window.features.get("token_cost", 0)),
            "front_clipped": source.get("front_clipped", False),
            "clipped_token_count": source.get("clipped_token_count", 0),
        }
        signature_first_seen = False
    source_path = str(source["path"])
    source_workload = str(source["workload"])
    source_window_id = str(source["window_id"])
    window_index = _window_index(window.window_id)

    row = {
        "workload": window_workload,
        "path": source_path,
        "label": manifest_entry["label"],
        "binary_label": manifest_entry.get(
            "binary_label",
            "mining" if manifest_entry["label"] == "mining_like" else "benign",
        ),
        "family": manifest_entry["family"],
        "opt_level": manifest_entry["opt_level"],
        "program": manifest_entry.get("program"),
        "variant": manifest_entry.get("variant"),
        "capture_id": manifest_entry.get("capture_id"),
        "parent_workload": parent_workload,
        "window_id": window.window_id,
        "window_index": window_index,
        "window_type": window.window_type,
        "l0_group_kind": window.group_kind,
        "l0_group_key": window.group_key,
        "group_id": parent_workload,
        "source_capture_path": manifest_entry.get("source_capture_path"),
        "trigger_reason": window.trigger_reason,
        "packing_mode": window.packing_mode,
        "composition_signature_key": signature_hash,
        "signature_first_seen": signature_first_seen,
        "signature_occurrence_index": window.features.get("signature_occurrence_index"),
        "signature_source_window_id": source_window_id,
        "signature_source_workload": source_workload,
        "max_bitwise_integer_ratio": window.features.get("max_bitwise_integer_ratio"),
        "window_token_cost": window.features.get("token_cost"),
        "pre_clip_token_cost": window.features.get("pre_clip_token_cost"),
        "rendered_token_cost": rendered["token_cost"],
        "front_clipped": bool(rendered.get("front_clipped", False)),
    }
    row = {key: value for key, value in row.items() if value is not None}
    return row


def _materialize_no_l0_window_row(
    workload_dir: Path,
    parent_workload: str,
    manifest_entry: dict[str, str],
    args: argparse.Namespace,
) -> None:
    windows_dir = workload_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    sass_name = "no_l0_window.sass"
    meta_name = "no_l0_window.json"
    (windows_dir / sass_name).write_text("NO_L0_WINDOW\n", encoding="utf-8")

    extraction_report_path = workload_dir / "dumps" / "extraction_report.json"
    extraction_report = {}
    if extraction_report_path.exists():
        extraction_report = json.loads(extraction_report_path.read_text(encoding="utf-8"))
    max_ratio = extraction_report.get("l0_max_bitwise_integer_ratio")

    row = {
        "workload": safe_workload_name(f"{parent_workload}__no_l0_window"),
        "path": f"windows/{sass_name}",
        "label": manifest_entry["label"],
        "binary_label": manifest_entry.get(
            "binary_label",
            "mining" if manifest_entry["label"] == "mining_like" else "benign",
        ),
        "family": manifest_entry["family"],
        "opt_level": manifest_entry["opt_level"],
        "program": manifest_entry.get("program"),
        "variant": manifest_entry.get("variant"),
        "capture_id": manifest_entry.get("capture_id"),
        "parent_workload": parent_workload,
        "window_id": "no_l0_window",
        "window_type": "no_l0_window",
        "group_id": parent_workload,
        "source_capture_path": manifest_entry.get("source_capture_path"),
        "trigger_reason": ["no_l0_window_emitted"],
        "packing_mode": "none",
        "max_bitwise_integer_ratio": max_ratio,
        "window_token_cost": 0,
        "no_l0_window": True,
        "default_prediction": "benign",
        "default_prediction_reason": "no_l0_window_emitted",
    }
    row = {key: value for key, value in row.items() if value is not None}
    write_jsonl(windows_dir / "manifests.jsonl", [row])

    window_report = {
        "window_id": "no_l0_window",
        "window_type": "no_l0_window",
        "parent_workload": parent_workload,
        "workload": row["workload"],
        "sass_path": f"windows/{sass_name}",
        "config_path": args.l0_config_obj.config_path,
        "resolved_config": args.l0_config_obj.to_dict(),
        "no_l0_window": True,
        "default_prediction": "benign",
        "default_prediction_reason": "no_l0_window_emitted",
        "max_bitwise_integer_ratio": max_ratio,
    }
    write_json(windows_dir / meta_name, window_report)

    extraction_report["l0_no_window_row"] = {
        "created": True,
        "path": "windows/manifests.jsonl",
        "default_prediction": "benign",
        "default_prediction_reason": "no_l0_window_emitted",
    }
    write_extraction_report(workload_dir, extraction_report)
    validate_l0_workload(workload_dir)


def _signature_source_key(window: L0Window) -> str:
    return json.dumps(
        {
            "group_key": window.group_key,
            "composition_signature_key": window.features.get("composition_signature_key"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _window_index(window_id: str) -> int | None:
    prefix = str(window_id or "").split("_", 1)[0]
    if not prefix.startswith("w"):
        return None
    try:
        return int(prefix[1:])
    except ValueError:
        return None


def signature_key_hash(signature_key: str) -> str:
    if not signature_key:
        return ""
    return hashlib.sha256(signature_key.encode("utf-8")).hexdigest()[:24]


def validate_l0_workload(workload_dir: Path) -> None:
    required = [
        "manifest.json",
        "launches.jsonl",
        "dumps/code_map.json",
        "dumps/extraction_report.json",
        "windows/manifests.jsonl",
    ]
    missing = [path for path in required if not (workload_dir / path).exists()]
    if missing:
        raise CaptureBuildError(f"missing required files: {', '.join(missing)}")
    rows = read_jsonl(workload_dir / "windows" / "manifests.jsonl")
    if not rows:
        raise CaptureBuildError("windows/manifests.jsonl has no window rows")
    for row in rows:
        path = workload_dir / str(row["path"])
        if not path.exists():
            raise CaptureBuildError(f"missing window SASS: {path.relative_to(workload_dir)}")
        if not path.read_text(encoding="utf-8").strip():
            raise CaptureBuildError(f"empty window SASS: {path.relative_to(workload_dir)}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, sort_keys=True)
            fh.write("\n")


def load_capture_manifest_specs(
    manifest_patterns: list[str],
    captures_root: Path,
) -> list[dict[str, Any]]:
    manifest_paths = sorted(
        {
            Path(match)
            for pattern in manifest_patterns
            for match in (glob.glob(pattern) or [pattern])
        }
    )
    if not manifest_paths:
        raise ManifestError(f"no capture manifests matched: {', '.join(manifest_patterns)}")

    specs: list[dict[str, Any]] = []
    used_workloads: Counter[str] = Counter()
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise ManifestError(f"missing capture manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"{manifest_path}:{line_no}: invalid JSON: {exc}") from exc
                specs.append(capture_spec_from_manifest_row(row, manifest_path, line_no, captures_root, used_workloads))
    if not specs:
        raise ManifestError(f"no capture rows found in: {', '.join(manifest_patterns)}")
    return specs


def capture_spec_from_manifest_row(
    row: dict[str, Any],
    manifest_path: Path,
    line_no: int,
    captures_root: Path,
    used_workloads: Counter[str],
) -> dict[str, Any]:
    missing = [key for key in ("label", "family", "workload", "program", "variant") if key not in row]
    if "capture_path" not in row and "capture_dir" not in row:
        missing.append("capture_path")
    if missing:
        raise ManifestError(f"{manifest_path}:{line_no}: missing fields: {', '.join(missing)}")

    capture_path = str(row.get("capture_path") or row["capture_dir"])
    capture_dir = captures_root / capture_path
    if not capture_dir.is_dir():
        raise ManifestError(f"{manifest_path}:{line_no}: missing capture_path: {capture_dir}")

    base_workload = safe_workload_name(str(row["workload"]))
    capture_id = safe_workload_name(str(row.get("capture_id") or capture_dir.name))
    used_workloads[base_workload] += 1
    workload = base_workload
    if used_workloads[base_workload] > 1:
        workload = f"{base_workload}_{capture_id[:12]}"

    entry = {
        "family": str(row["family"]),
        "label": str(row["label"]),
        "opt_level": str(row.get("opt_level") or OPT_LEVEL_CAPTURE),
        "program": str(row["program"]),
        "variant": str(row["variant"]),
        "capture_id": str(row.get("capture_id") or capture_dir.name),
        "source_capture_path": capture_path,
    }
    if "binary_label" in row:
        entry["binary_label"] = str(row["binary_label"])
    return {
        "capture_dir": capture_dir,
        "workload": workload,
        "manifest_entry": entry,
        "event_type_counts": row.get("event_type_counts"),
    }


def capture_spec_has_no_kernel_launch(capture_spec: dict[str, Any]) -> bool:
    counts = capture_spec.get("event_type_counts")
    if not isinstance(counts, dict):
        return False
    return int(counts.get("kernel_launch") or 0) == 0


def safe_workload_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "unknown"


def new_build_report() -> dict[str, Any]:
    return {
        "captures_scanned": 0,
        "workloads_created": 0,
        "duplicates_skipped": 0,
        "empty_captures_skipped": 0,
        "unmapped_captures_skipped": 0,
        "no_l0_windows_skipped": 0,
        "failed_captures": 0,
        "windows_created": 0,
        "no_l0_window_rows_created": 0,
        "labels": Counter(),
        "opt_levels": Counter(),
        "failures": [],
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    serializable = {
        **report,
        "labels": dict(report["labels"]),
        "opt_levels": dict(report["opt_levels"]),
    }
    write_json(output_dir / "build_report.json", serializable)


def print_summary(report: dict[str, Any]) -> None:
    print("\nDataset build complete.\n")
    print(f"captures scanned: {report['captures_scanned']}")
    print(f"workloads created: {report['workloads_created']}")
    if report.get("windows_created"):
        print(f"windows created: {report['windows_created']}")
    if report.get("no_l0_window_rows_created"):
        print(f"no L0 window rows created: {report['no_l0_window_rows_created']}")
    print(f"duplicates skipped: {report['duplicates_skipped']}")
    print(f"empty captures skipped: {report['empty_captures_skipped']}")
    print(f"unmapped captures skipped: {report['unmapped_captures_skipped']}")
    print(f"no L0 windows skipped: {report['no_l0_windows_skipped']}")
    print(f"failed captures: {report['failed_captures']}")
    print("\nlabels:")
    for label, count in sorted(report["labels"].items()):
        print(f"  {label}: {count}")
    print("\nopt levels:")
    for opt_level, count in sorted(report["opt_levels"].items()):
        print(f"  {opt_level}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
