"""Static normalized-SASS feature extraction for online L0 hints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .split_kernels import count_instruction_lines


BITWISE_OPS = {
    "AND",
    "BFE",
    "BFI",
    "BREV",
    "FLO",
    "IADD3",
    "IMAD",
    "ISCADD",
    "LOP",
    "LOP3",
    "POPC",
    "PRMT",
    "SHF",
    "SHL",
    "SHR",
    "VABSDIFF",
    "XOR",
}
INTEGER_OPS = {
    "I2F",
    "I2I",
    "IABS",
    "IADD",
    "IADD3",
    "ICMP",
    "IDP",
    "IMAD",
    "IMNMX",
    "IMUL",
    "ISCADD",
    "ISETP",
    "LEA",
    "LOP",
    "LOP3",
    "POPC",
    "SHF",
    "SHL",
    "SHR",
    "VADD",
}
MEMORY_OP_PREFIXES = ("LD", "ST", "ATOM", "ATOMG", "RED", "SULD", "SUST")
BRANCH_OPS = {"BRA", "BRX", "JMP", "JMX", "RET", "EXIT", "CALL"}


def kernel_static_features(kernel_dir: Path) -> dict[str, Any]:
    kernel_lines = _read_lines(kernel_dir / "kernel.normalized.sass")
    loop_lines = _read_lines(kernel_dir / "main_loop.normalized.sass")
    selected_loop = loop_lines or kernel_lines

    kernel_ops = [_opcode(line) for line in kernel_lines if _opcode(line)]
    loop_ops = [_opcode(line) for line in selected_loop if _opcode(line)]
    instruction_count = count_instruction_lines(kernel_lines)
    loop_instruction_count = count_instruction_lines(selected_loop)
    total = max(1, len(kernel_ops))

    bitwise_integer = sum(1 for op in kernel_ops if is_bitwise_integer_op(op))
    memory = sum(1 for op in kernel_ops if _is_memory_op(op))
    branches = sum(1 for op in kernel_ops if op in BRANCH_OPS)
    compute_intensity = bitwise_integer / max(1, bitwise_integer + memory)
    loop_bitwise_integer = sum(1 for op in loop_ops if is_bitwise_integer_op(op))
    loop_memory = sum(1 for op in loop_ops if _is_memory_op(op))

    return {
        "instruction_count": instruction_count,
        "main_loop_instruction_count": loop_instruction_count,
        "bitwise_integer_instruction_count": bitwise_integer,
        "memory_instruction_count": memory,
        "branch_instruction_count": branches,
        "bitwise_integer_ratio": bitwise_integer / total,
        "memory_instruction_ratio": memory / total,
        "branch_density": branches / total,
        "compute_intensity_score": compute_intensity,
        "main_loop_compute_intensity_score": loop_bitwise_integer / max(1, loop_bitwise_integer + loop_memory),
    }


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _opcode(line: str) -> str:
    return opcode_from_normalized(line)


def opcode_from_normalized(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.endswith(":") or stripped == "KERNEL_BOUNDARY":
        return ""
    token = stripped.split(maxsplit=1)[0]
    token = token.split(".", 1)[0]
    token = token.lstrip("@!PT0123456789")
    return token.upper()


def is_bitwise_integer_op(opcode: str) -> bool:
    return opcode in BITWISE_OPS or opcode in INTEGER_OPS


def bitwise_integer_instruction_ratio(lines: list[str]) -> float:
    ops = [opcode_from_normalized(line) for line in lines]
    ops = [op for op in ops if op]
    if not ops:
        return 0.0
    return sum(1 for op in ops if is_bitwise_integer_op(op)) / len(ops)


def _is_memory_op(opcode: str) -> bool:
    return any(opcode.startswith(prefix) for prefix in MEMORY_OP_PREFIXES)
