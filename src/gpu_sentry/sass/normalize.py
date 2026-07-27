"""SASS normalization for model-ready text."""

from __future__ import annotations

import re
from pathlib import Path

from .sass_tokens import KERNEL_BODY_TOKEN_BUDGET, truncate_lines_to_token_budget


def normalize_kernel_files(kernel_dir: Path, main_loop_token_budget: int = KERNEL_BODY_TOKEN_BUDGET) -> None:
    for raw_name, normalized_name in (
        ("kernel.sass", "kernel.normalized.sass"),
        ("main_loop.sass", "main_loop.normalized.sass"),
    ):
        raw_path = kernel_dir / raw_name
        if raw_path.exists():
            normalized = normalize_sass(
                raw_path.read_text(encoding="utf-8").splitlines(),
                token_budget=main_loop_token_budget if raw_name == "main_loop.sass" else None,
            )
            (kernel_dir / normalized_name).write_text(normalized, encoding="utf-8")


def normalize_sass(lines: list[str], token_budget: int | None = None) -> str:
    normalized = []
    for line in lines:
        instr = normalize_instruction(line)
        if instr:
            normalized.append(instr)
    if token_budget is not None:
        normalized = truncate_lines_to_token_budget(normalized, token_budget)
    return "\n".join(normalized).rstrip() + ("\n" if normalized else "")


def normalize_instruction(line: str) -> str | None:
    line = _strip_comments(line).strip().rstrip(";")
    if not line or line.endswith(":"):
        return None
    line = _strip_scheduler_annotations(line)

    predicate = ""
    pred_match = re.match(r"^@(?P<neg>!)?(?P<pred>[A-Za-z0-9_]+)\s+(?P<body>.*)$", line)
    if pred_match:
        predicate = "@!PRED " if pred_match.group("neg") else "@PRED "
        line = pred_match.group("body").strip()

    if not line:
        return None
    opcode, _, operands = line.partition(" ")
    opcode = opcode.split(".")[0].upper()
    if opcode == "NOP":
        return None
    operand_list = [_normalize_operand(op.strip(), opcode) for op in _split_operands(operands)]
    operand_list = [op for op in operand_list if op]
    if operand_list:
        return f"{predicate}{opcode} {', '.join(operand_list)}"
    return f"{predicate}{opcode}"


def _normalize_operand(operand: str, opcode: str = "") -> str:
    operand = operand.strip()
    if not operand:
        return ""
    if "[" in operand and "]" in operand and not operand.lower().startswith("c["):
        return "MEM"
    if re.fullmatch(r"-?c\[[^\]]+\]\[[^\]]+\](?:\s+.*)?", operand, flags=re.IGNORECASE):
        return "CONST"
    if re.fullmatch(r"\bB\d+\b", operand, flags=re.IGNORECASE):
        return "BREG"
    if re.fullmatch(r"[-+]?\b(?:R|UR)\d+(?:\.[A-Za-z0-9_]+)?\b", operand, flags=re.IGNORECASE):
        return "UREG" if operand.lstrip("+-").upper().startswith("UR") else "REG"
    if re.fullmatch(r"\b(?:RZ|URZ)\b", operand, flags=re.IGNORECASE):
        return "ZERO"
    if re.fullmatch(r"!?\b(?:P|UP)\d+\b|!?\bPT\b", operand, flags=re.IGNORECASE):
        return "PRED"
    if opcode in {
        "BRA",
        "BRX",
        "BSSY",
        "CALL",
        "JMP",
        "JMX",
        "PBK",
        "PCNT",
        "PEXIT",
        "PRET",
        "SSY",
        "SYNC",
    } and _is_symbolic_target(operand):
        return "LABEL"
    if re.fullmatch(r"L_[A-Za-z0-9_]+|\.?L[A-Za-z0-9_.$]+|TARGET\d+", operand):
        return "LABEL"

    operand = re.sub(r"\.reuse\b", "", operand)
    operand = re.sub(r"\bURZ\b", "ZERO", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bRZ\b", "ZERO", operand, flags=re.IGNORECASE)
    operand = re.sub(r"!?\bPT\b", "PRED", operand, flags=re.IGNORECASE)
    operand = re.sub(r"!?\bP\d+\b", "PRED", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bUP\d+\b", "PRED", operand, flags=re.IGNORECASE)
    operand = re.sub(r"[-+]?\bUR\d+(?:\.[A-Za-z0-9_]+)?\b", "UREG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"[-+]?\bR\d+(?:\.[A-Za-z0-9_]+)?\b", "REG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bSR_[A-Za-z0-9_.]+\b", "SREG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"\bB\d+\b", "BREG", operand, flags=re.IGNORECASE)
    operand = re.sub(r"-?\d*@(lo|hi)\([^)]+\)", "IMM", operand, flags=re.IGNORECASE)
    operand = re.sub(r"-?0x[0-9a-fA-F]+", "IMM", operand)
    operand = re.sub(
        r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?\b",
        "IMM",
        operand,
        flags=re.IGNORECASE,
    )
    operand = re.sub(r"(?<![A-Za-z_])-?\d+\b", "IMM", operand)
    operand = re.sub(r"\s+", " ", operand).strip()
    return operand


def _strip_scheduler_annotations(line: str) -> str:
    line = re.sub(r"\s+\?trans\d+\b", "", line, flags=re.IGNORECASE)
    return re.sub(r"\s+&[A-Za-z_][A-Za-z0-9_]*=(?:\{[^}]*\}|\S+)", "", line)


def _is_symbolic_target(operand: str) -> bool:
    operand = operand.strip()
    operand = operand.lstrip("+-").strip()
    if operand.startswith("`"):
        operand = operand[1:].strip()
    if operand.startswith("(") and operand.endswith(")"):
        operand = operand[1:-1].strip()
    if not operand:
        return False
    if any(char in operand for char in "$@"):
        return True
    return bool(re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.$@+-]*", operand))


def _split_operands(operands: str) -> list[str]:
    if not operands.strip():
        return []
    if "," not in operands:
        return operands.split()
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(operands):
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(operands[start:idx])
            start = idx + 1
    parts.append(operands[start:])
    return parts


def _strip_comments(line: str) -> str:
    line = re.sub(r'/\*.*?\*/', "", line)
    return re.sub(r'\(\*".*?"\*\)', "", line)
