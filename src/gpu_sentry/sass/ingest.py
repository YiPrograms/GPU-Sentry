"""Capture ingestion: events, code objects, and launches."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .manifest import write_json


class IngestError(RuntimeError):
    """Raised for malformed captures."""


LAUNCH_FIELDS = (
    "sequence",
    "timestamp_ns",
    "pid",
    "tid",
    "code_id",
    "kernel_name",
    "grid_dim",
    "block_dim",
    "shared_mem_bytes",
    "stream",
    "device_pci_bus_id",
)


def read_process(capture_dir: Path) -> dict[str, Any]:
    path = capture_dir / "process.json"
    if not path.exists():
        raise IngestError("missing process.json")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid process.json: {exc}") from exc


def workload_name_from_process(process: dict[str, Any]) -> str:
    exe_path = process.get("exe_path")
    if not exe_path:
        raise IngestError("process.json missing exe_path")
    return Path(str(exe_path)).name


def read_events(capture_dir: Path) -> list[dict[str, Any]]:
    path = capture_dir / "events.jsonl"
    if not path.exists():
        raise IngestError("missing events.jsonl")
    events: list[dict[str, Any]] = []
    last_key: tuple[int, int, int] | None = None
    already_sorted = True
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestError(f"events.jsonl:{line_no}: invalid JSON: {exc}") from exc
            event["_line_index"] = line_no
            key = event_sort_key(event)
            if last_key is not None and key < last_key:
                already_sorted = False
            last_key = key
            events.append(event)
    return events if already_sorted else sorted(events, key=event_sort_key)


def event_sort_key(event: dict[str, Any]) -> tuple[int, int, int]:
    if isinstance(event.get("sequence"), int):
        return (0, int(event["sequence"]), int(event.get("_line_index", 0)))
    if isinstance(event.get("timestamp_ns"), int):
        return (1, int(event["timestamp_ns"]), int(event.get("_line_index", 0)))
    return (2, int(event.get("_line_index", 0)), 0)


def split_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code_events = [e for e in events if e.get("type") == "code"]
    launches = [e for e in events if e.get("type") == "kernel_launch"]
    if not code_events:
        raise IngestError("no code event")
    if not launches:
        raise IngestError("no kernel_launch event")
    return code_events, launches


def copy_code_objects(
    capture_dir: Path,
    code_events: list[dict[str, Any]],
    workload_dir: Path,
) -> dict[str, dict[str, Any]]:
    code_dir = capture_dir / "code"
    if not code_dir.is_dir():
        raise IngestError("missing code/ directory")

    dumps_dir = workload_dir / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    code_map: dict[str, dict[str, Any]] = {}
    copied_by_sha: dict[str, dict[str, Any]] = {}
    capture_root = capture_dir.resolve()

    for event in code_events:
        if "code_id" not in event:
            raise IngestError("code event missing code_id")
        rel = event.get("path")
        if not rel:
            raise IngestError(f"code_id {event['code_id']} missing path")

        src = (capture_dir / str(rel)).resolve()
        if not src.exists():
            raise IngestError(f"code event path does not exist: {rel}")
        if not _is_relative_to(src, capture_root):
            raise IngestError(f"code event path escapes capture directory: {rel}")

        dst = dumps_dir / src.name
        expected_sha = event.get("sha256")
        expected_size = event.get("size")
        duplicate = copied_by_sha.get(str(expected_sha).lower()) if expected_sha else None
        if duplicate is not None:
            if expected_size is not None and int(expected_size) != int(duplicate["size"]):
                raise IngestError(
                    f"size mismatch for duplicate {rel}: expected {expected_size}, got {duplicate['size']}"
                )
            digest = str(duplicate["sha256"])
            size = int(duplicate["size"])
            dump_path = str(duplicate["dump_path"])
        else:
            digest, size = copy_with_sha256(src, dst)
            dump_path = f"dumps/{src.name}"
        if expected_sha and str(expected_sha).lower() != digest:
            raise IngestError(
                f"sha256 mismatch for {rel}: expected {expected_sha}, got {digest}"
            )
        if expected_size is not None and int(expected_size) != size:
            raise IngestError(f"size mismatch for {rel}: expected {expected_size}, got {size}")

        code_id = str(event["code_id"])
        code_map[code_id] = {
            "code_id": event["code_id"],
            "code_type": event.get("code_type"),
            "source_path": str(rel),
            "dump_path": dump_path,
            "sha256": digest,
            "size": size,
        }
        if duplicate is not None:
            code_map[code_id]["deduplicated_dump_from"] = str(duplicate["code_id"])
        copied_by_sha.setdefault(
            digest,
            {
                "code_id": event["code_id"],
                "dump_abs_path": str(dst),
                "dump_path": dump_path,
                "sha256": digest,
                "size": size,
            },
        )

    write_json(dumps_dir / "code_map.json", code_map)
    return code_map


def copy_with_sha256(src: Path, dst: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total = 0
    tmp = dst.with_name(f".{dst.name}.tmp")
    with src.open("rb") as in_fh, tmp.open("wb") as out_fh:
        while True:
            chunk = in_fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            out_fh.write(chunk)
            total += len(chunk)
    tmp.replace(dst)
    shutil.copystat(src, dst, follow_symlinks=True)
    return hasher.hexdigest(), total


def write_launches(workload_dir: Path, launches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in launches:
        normalized.append({field: event.get(field) for field in LAUNCH_FIELDS})

    path = workload_dir / "launches.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for launch in normalized:
            json.dump(launch, fh, sort_keys=True)
            fh.write("\n")
    return normalized


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
