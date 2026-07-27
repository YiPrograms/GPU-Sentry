"""Code object format detection and SASS disassembly."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .manifest import write_json


class DisassemblyError(RuntimeError):
    """Raised when code objects cannot be disassembled."""


def find_cuda_tools() -> dict[str, Path]:
    candidates: dict[str, list[str]] = {"nvdisasm": [], "cuobjdump": []}
    for name in candidates:
        found = shutil.which(name)
        if found:
            candidates[name].append(found)
    for pattern in (
        "/usr/local/cuda/bin/{tool}",
        "/usr/local/cuda-*/bin/{tool}",
        "/opt/cuda/bin/{tool}",
    ):
        for tool in candidates:
            candidates[tool].extend(glob.glob(pattern.format(tool=tool)))

    tools: dict[str, Path] = {}
    for name, paths in candidates.items():
        for path in paths:
            p = Path(path)
            if p.is_file() and os.access(p, os.X_OK):
                tools[name] = p
                break
    return tools


def detect_code_format(path: Path) -> str:
    data = path.read_bytes()[:256]
    if data.startswith(b"\x7fELF"):
        return "cubin"
    if data[:4] == bytes.fromhex("50ed55ba") or data[:4] == bytes.fromhex("ba55ed50"):
        return "fatbin"
    if _looks_like_ptx(data):
        return "ptx"
    return "unknown"


def disassemble_code_objects(
    workload_dir: Path,
    code_map: dict[str, dict[str, Any]],
    tools: dict[str, Path] | None = None,
) -> dict[str, Any]:
    tools = tools or find_cuda_tools()
    report: dict[str, Any] = {"code_objects": [], "deduplicated_code_objects": 0}
    by_sha: dict[str, dict[str, Any]] = {}

    for code_id in sorted(code_map, key=_code_id_sort_key):
        entry = code_map[code_id]
        dump_rel = Path(entry["dump_path"])
        dump_path = workload_dir / dump_rel
        fmt = detect_code_format(dump_path)
        out_rel = Path("dumps") / f"code_{code_id}.nvdisasm.txt"
        out_path = workload_dir / out_rel
        item = {
            "code_id": entry["code_id"],
            "path": str(dump_rel),
            "detected_format": fmt,
            "disassembly_output": str(out_rel),
            "status": "pending",
        }
        sha = str(entry.get("sha256") or "")
        if sha:
            item["sha256"] = sha
            prior = by_sha.get(sha)
            if prior and prior.get("status") == "ok":
                item["status"] = "ok"
                item["deduplicated_from_code_id"] = prior["code_id"]
                item["disassembly_output"] = prior["disassembly_output"]
                item["tool"] = prior.get("tool")
                report["deduplicated_code_objects"] += 1
                report["code_objects"].append(item)
                continue

        if fmt == "ptx":
            item["status"] = "ptx_only"
            item["error"] = "PTX-only code object; SASS disassembly not attempted"
            report["code_objects"].append(item)
            continue
        if fmt == "unknown":
            item["status"] = "error"
            item["error"] = "unknown code object format"
            report["code_objects"].append(item)
            continue

        commands = _commands_for_format(fmt, dump_path, tools)
        if not commands:
            item["status"] = "error"
            item["error"] = f"no CUDA disassembly tool found for {fmt}"
            report["code_objects"].append(item)
            continue

        errors: list[str] = []
        for command in commands:
            ok, stderr = _run_disassembler(command, out_path)
            if ok and out_path.exists() and out_path.stat().st_size > 0:
                item["status"] = "ok"
                item["tool"] = str(command[0])
                break
            errors.append(stderr.strip() or f"{command[0]} produced no output")
        else:
            item["status"] = "error"
            item["error"] = " | ".join(errors)

        if sha and item.get("status") == "ok":
            by_sha[sha] = item
        report["code_objects"].append(item)

    write_extraction_report(workload_dir, report)
    if not any(item["status"] == "ok" for item in report["code_objects"]):
        raise DisassemblyError("unable to disassemble code object")
    return report


def write_extraction_report(workload_dir: Path, report: dict[str, Any]) -> None:
    write_json(workload_dir / "dumps" / "extraction_report.json", report)


def _commands_for_format(
    fmt: str,
    dump_path: Path,
    tools: dict[str, Path],
) -> list[list[str]]:
    commands: list[list[str]] = []
    if fmt == "cubin":
        if "nvdisasm" in tools:
            commands.append([str(tools["nvdisasm"]), str(dump_path)])
        if "cuobjdump" in tools:
            commands.append([str(tools["cuobjdump"]), "--dump-sass", str(dump_path)])
    elif fmt == "fatbin":
        if "cuobjdump" in tools:
            commands.append([str(tools["cuobjdump"]), "--dump-sass", str(dump_path)])
    return commands


def _run_disassembler(command: list[str], out_path: Path) -> tuple[bool, str]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and result.stdout:
        out_path.write_text(result.stdout, encoding="utf-8")
        return True, result.stderr
    return False, result.stderr or result.stdout


def _looks_like_ptx(data: bytes) -> bool:
    if not data:
        return False
    textish = all(byte in b"\n\r\t" or 32 <= byte < 127 for byte in data)
    if not textish:
        return False
    lower = data.lower()
    return b".version" in lower or b".target" in lower or b".entry" in lower


def _code_id_sort_key(code_id: str) -> tuple[int, int | str]:
    text = str(code_id)
    if text.isdigit():
        return (0, int(text))
    return (1, text)
