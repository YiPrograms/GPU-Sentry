from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import nvidia_smi
from .command_rewrite import rewrite_gpu_args
from .io import append_jsonl, safe_slug, write_json


@dataclass(frozen=True)
class RecordRequest:
    command: list[str]
    output_dir: Path
    gpu: int
    workload: str
    label: str
    binary_label: str | None = None
    family: str | None = None
    program: str | None = None
    variant: str | None = None
    cwd: Path | None = None
    sampling_interval_s: float = 1.0
    timeout_s: float | None = None
    extra_metadata: dict[str, Any] | None = None


def record_command(request: RecordRequest) -> dict[str, Any]:
    selected_gpu = select_gpu(request.gpu)
    started_wall_ns = time.time_ns()
    run_id = build_run_id(request.workload, started_wall_ns)
    run_dir = request.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    cwd = request.cwd or Path.cwd()
    rewritten = rewrite_gpu_args(request.command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(request.gpu)
    env.setdefault("GPU_SENTRY_DISABLE", "1")

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "workload": request.workload,
        "label": request.label,
        "binary_label": request.binary_label,
        "family": request.family,
        "program": request.program,
        "variant": request.variant,
        "command_original": request.command,
        "command_rewritten": rewritten.argv,
        "command_rewrite_changes": rewritten.changes,
        "cwd": str(cwd),
        "hostname": socket.gethostname(),
        "sampling_interval_s": request.sampling_interval_s,
        "timeout_s": request.timeout_s,
        "single_gpu": {
            "requested_physical_gpu_index": request.gpu,
            "visible_logical_gpu_index": 0,
            "cuda_visible_devices": str(request.gpu),
            "gpu_index": selected_gpu.index,
            "gpu_uuid": selected_gpu.uuid,
            "pci_bus_id": selected_gpu.pci_bus_id,
            "name": selected_gpu.name,
            "driver_version": selected_gpu.driver_version,
            "memory_total_bytes": selected_gpu.memory_total_bytes,
        },
        "backend": {
            "name": "nvidia-smi",
            "device_query": True,
            "pmon": True,
            "compute_apps": True,
        },
        "started_at_ns": started_wall_ns,
        "status": "running",
    }
    if request.extra_metadata:
        metadata.update(request.extra_metadata)
    write_json(run_dir / "metadata.json", metadata)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    device_count = 0
    process_count = 0
    timed_out = False
    exit_code: int | None = None

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            rewritten.argv,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        metadata["pid"] = proc.pid
        write_json(run_dir / "metadata.json", metadata)
        monotonic_start = time.monotonic()
        next_tick = monotonic_start
        while True:
            now = time.monotonic()
            if now >= next_tick:
                timestamp_ns = time.time_ns()
                device_count += sample_device(run_dir, run_id, timestamp_ns, selected_gpu.index)
                process_count += sample_processes(run_dir, run_id, timestamp_ns, selected_gpu)
                next_tick += request.sampling_interval_s

            exit_code = proc.poll()
            if exit_code is not None:
                break
            if request.timeout_s is not None and now - monotonic_start >= request.timeout_s:
                timed_out = True
                terminate_process_group(proc)
                exit_code = proc.wait(timeout=10)
                break
            sleep_for = max(0.02, min(0.2, next_tick - time.monotonic()))
            time.sleep(sleep_for)

    finished_ns = time.time_ns()
    metadata.update(
        {
            "finished_at_ns": finished_ns,
            "duration_s": round((finished_ns - started_wall_ns) / 1_000_000_000, 6),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "status": "timed_out" if timed_out else ("ok" if exit_code == 0 else "failed"),
            "sample_counts": {
                "device_metrics": device_count,
                "process_metrics": process_count,
            },
        }
    )
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def sample_device(run_dir: Path, run_id: str, timestamp_ns: int, gpu_index: int) -> int:
    try:
        sample = nvidia_smi.query_device_metrics(gpu_index)
    except nvidia_smi.NvidiaSmiError as exc:
        append_jsonl(run_dir / "device_metrics_errors.jsonl", {"timestamp_ns": timestamp_ns, "error": str(exc)})
        return 0
    if sample is None:
        return 0
    sample.update({"timestamp_ns": timestamp_ns, "run_id": run_id})
    append_jsonl(run_dir / "device_metrics.jsonl", sample)
    return 1


def sample_processes(run_dir: Path, run_id: str, timestamp_ns: int, gpu: nvidia_smi.GpuInfo) -> int:
    try:
        apps = nvidia_smi.query_compute_apps()
    except nvidia_smi.NvidiaSmiError as exc:
        append_jsonl(run_dir / "process_metrics_errors.jsonl", {"timestamp_ns": timestamp_ns, "source": "compute_apps", "error": str(exc)})
        apps = []
    try:
        pmon = nvidia_smi.query_pmon()
    except nvidia_smi.NvidiaSmiError as exc:
        append_jsonl(run_dir / "process_metrics_errors.jsonl", {"timestamp_ns": timestamp_ns, "source": "pmon", "error": str(exc)})
        pmon = []

    apps_by_pid = {int(app["pid"]): app for app in apps if app.get("gpu_uuid") == gpu.uuid}
    pmon_by_pid = {int(row["pid"]): row for row in pmon if row.get("gpu_index") == gpu.index}
    pids = sorted(set(apps_by_pid) | set(pmon_by_pid))
    count = 0
    for pid in pids:
        app = apps_by_pid.get(pid, {})
        util = pmon_by_pid.get(pid, {})
        used_bytes = app.get("process_gpu_memory_used_bytes")
        row = {
            "timestamp_ns": timestamp_ns,
            "run_id": run_id,
            "gpu_uuid": gpu.uuid,
            "gpu_index": gpu.index,
            "pci_bus_id": gpu.pci_bus_id,
            "pid": pid,
            "process_name": app.get("process_name") or util.get("process_name") or read_proc_comm(pid),
            "process_type": util.get("process_type"),
            "process_gpu_utilization_pct": util.get("process_gpu_utilization_pct"),
            "pmon_memory_utilization_pct": util.get("pmon_memory_utilization_pct"),
            "process_gpu_memory_used_bytes": used_bytes,
            "process_gpu_memory_pct": percentage(used_bytes, gpu.memory_total_bytes),
            "host_rss_bytes": read_proc_rss_bytes(pid),
        }
        row["host_ram_gb"] = bytes_to_gb(row["host_rss_bytes"])
        append_jsonl(run_dir / "process_metrics.jsonl", row)
        count += 1
    return count


def select_gpu(index: int) -> nvidia_smi.GpuInfo:
    gpus = nvidia_smi.query_gpus()
    for gpu in gpus:
        if gpu.index == index:
            return gpu
    available = ", ".join(str(gpu.index) for gpu in gpus) or "none"
    raise SystemExit(f"GPU index {index} is not available; available indexes: {available}")


def terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def read_proc_rss_bytes(pid: int) -> int | None:
    path = Path("/proc") / str(pid) / "status"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def read_proc_comm(pid: int) -> str | None:
    path = Path("/proc") / str(pid) / "comm"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def percentage(value: int | None, total: int | None) -> float | None:
    if value is None or not total:
        return None
    return value * 100.0 / total


def bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000_000


def build_run_id(workload: str, started_ns: int) -> str:
    return f"{safe_slug(workload)}-{started_ns}"

