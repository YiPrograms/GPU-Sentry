from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


GPU_QUERY_FIELDS = [
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "fan.speed",
    "memory.used",
    "memory.total",
    "clocks.gr",
    "clocks.mem",
]

COMPUTE_APPS_FIELDS_WITH_NAME = ["gpu_uuid", "pid", "process_name", "used_memory"]
COMPUTE_APPS_FIELDS_MINIMAL = ["gpu_uuid", "pid", "used_memory"]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    pci_bus_id: str
    name: str | None = None
    driver_version: str | None = None
    memory_total_bytes: int | None = None


class NvidiaSmiError(RuntimeError):
    pass


def run_nvidia_smi(args: list[str], *, timeout: float = 10.0) -> str:
    cmd = ["nvidia-smi", *args]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
    except FileNotFoundError as exc:
        raise NvidiaSmiError("nvidia-smi is not available on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise NvidiaSmiError(f"nvidia-smi timed out: {' '.join(cmd)}") from exc
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise NvidiaSmiError(f"nvidia-smi failed: {message}")
    return proc.stdout


def query_csv(fields: list[str]) -> list[dict[str, str]]:
    stdout = run_nvidia_smi(
        [
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_csv(stdout, fields)


def parse_csv(text: str, fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    reader = csv.reader(StringIO(text))
    for raw in reader:
        if not raw:
            continue
        values = [value.strip() for value in raw]
        if len(values) < len(fields):
            values.extend([""] * (len(fields) - len(values)))
        rows.append(dict(zip(fields, values[: len(fields)])))
    return rows


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    text = text.replace("%", "").replace("MiB", "").replace("W", "").replace("C", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def mib_to_bytes(value: Any) -> int | None:
    number = parse_number(value)
    if number is None:
        return None
    return int(number * 1024 * 1024)


def query_gpus() -> list[GpuInfo]:
    rows = query_csv(GPU_QUERY_FIELDS)
    infos: list[GpuInfo] = []
    for row in rows:
        index = parse_int(row.get("index"))
        if index is None:
            continue
        infos.append(
            GpuInfo(
                index=index,
                uuid=str(row.get("uuid") or ""),
                pci_bus_id=str(row.get("pci.bus_id") or ""),
                name=row.get("name") or None,
                driver_version=row.get("driver_version") or None,
                memory_total_bytes=mib_to_bytes(row.get("memory.total")),
            )
        )
    return infos


def query_device_metrics(selected_index: int) -> dict[str, Any] | None:
    rows = query_csv(GPU_QUERY_FIELDS)
    for row in rows:
        index = parse_int(row.get("index"))
        if index != selected_index:
            continue
        memory_used = mib_to_bytes(row.get("memory.used"))
        memory_total = mib_to_bytes(row.get("memory.total"))
        return {
            "gpu_index": index,
            "gpu_uuid": row.get("uuid") or None,
            "pci_bus_id": row.get("pci.bus_id") or None,
            "gpu_name": row.get("name") or None,
            "driver_version": row.get("driver_version") or None,
            "gpu_utilization_pct": parse_number(row.get("utilization.gpu")),
            "memory_utilization_pct": parse_number(row.get("utilization.memory")),
            "power_usage_watts": parse_number(row.get("power.draw")),
            "power_limit_watts": parse_number(row.get("power.limit")),
            "temperature_celsius": parse_number(row.get("temperature.gpu")),
            "fan_speed_pct": parse_number(row.get("fan.speed")),
            "memory_used_bytes": memory_used,
            "memory_total_bytes": memory_total,
            "graphics_clock_mhz": parse_number(row.get("clocks.gr")),
            "memory_clock_mhz": parse_number(row.get("clocks.mem")),
        }
    return None


def query_compute_apps() -> list[dict[str, Any]]:
    fields = COMPUTE_APPS_FIELDS_WITH_NAME
    try:
        rows = _query_compute_apps(fields)
    except NvidiaSmiError:
        fields = COMPUTE_APPS_FIELDS_MINIMAL
        rows = _query_compute_apps(fields)
    apps: list[dict[str, Any]] = []
    for row in rows:
        pid = parse_int(row.get("pid"))
        if pid is None:
            continue
        apps.append(
            {
                "gpu_uuid": row.get("gpu_uuid") or None,
                "pid": pid,
                "process_name": row.get("process_name") or read_proc_comm(pid),
                "process_gpu_memory_used_bytes": mib_to_bytes(row.get("used_memory")),
            }
        )
    return apps


def _query_compute_apps(fields: list[str]) -> list[dict[str, str]]:
    stdout = run_nvidia_smi(
        [
            f"--query-compute-apps={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_csv(stdout, fields)


def query_pmon() -> list[dict[str, Any]]:
    stdout = run_nvidia_smi(["pmon", "-c", "1"], timeout=15.0)
    return parse_pmon(stdout)


def parse_pmon(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        gpu_index = parse_int(parts[0])
        pid = parse_int(parts[1])
        if gpu_index is None or pid is None:
            continue
        samples.append(
            {
                "gpu_index": gpu_index,
                "pid": pid,
                "process_type": none_if_dash(parts[2]),
                "process_gpu_utilization_pct": parse_number(parts[3]),
                "pmon_memory_utilization_pct": parse_number(parts[4]),
                "encoder_utilization_pct": parse_number(parts[5]),
                "decoder_utilization_pct": parse_number(parts[6]),
                "jpg_utilization_pct": parse_number(parts[7]),
                "ofa_utilization_pct": parse_number(parts[8]),
                "process_name": none_if_dash(" ".join(parts[9:])),
            }
        )
    return samples


def none_if_dash(value: str | None) -> str | None:
    if value in {None, "", "-"}:
        return None
    return value


def read_proc_comm(pid: int) -> str | None:
    path = Path("/proc") / str(pid) / "comm"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
