"""Approximate SASS control-flow graph construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BRANCH_OPS = {"BRA", "BRX", "JMP", "JMX"}
TERMINAL_OPS = {"RET", "EXIT"}
CALL_OPS = {"CALL"}


@dataclass
class ParsedInstruction:
    index: int
    text: str
    labels: list[str]


def build_cfg_for_kernel(kernel_dir: Path) -> dict[str, Any]:
    lines = (kernel_dir / "kernel.sass").read_text(encoding="utf-8").splitlines()
    cfg = build_cfg(lines)
    with (kernel_dir / "cfg.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return cfg


def build_cfg(lines: list[str]) -> dict[str, Any]:
    instructions, label_to_index = parse_kernel_lines(lines)
    if not instructions:
        return {"basic_blocks": [], "edges": [], "entry": None}

    leaders = {0}
    for idx in label_to_index.values():
        leaders.add(idx)
    for instr in instructions:
        opcode = opcode_of(instr.text)
        target = branch_target(instr.text)
        if target and target in label_to_index:
            leaders.add(label_to_index[target])
        if opcode in BRANCH_OPS | TERMINAL_OPS | CALL_OPS and instr.index + 1 < len(instructions):
            leaders.add(instr.index + 1)

    sorted_leaders = sorted(leaders)
    block_ranges: list[tuple[int, int]] = []
    for pos, start in enumerate(sorted_leaders):
        end = (sorted_leaders[pos + 1] - 1) if pos + 1 < len(sorted_leaders) else len(instructions) - 1
        block_ranges.append((start, end))

    index_to_block: dict[int, str] = {}
    blocks: list[dict[str, Any]] = []
    for block_id, (start, end) in enumerate(block_ranges):
        bid = f"B{block_id}"
        for idx in range(start, end + 1):
            index_to_block[idx] = bid
        block_instrs = instructions[start : end + 1]
        labels = [label for instr in block_instrs for label in instr.labels]
        blocks.append(
            {
                "id": bid,
                "start_line": start,
                "end_line": end,
                "labels": labels,
                "instructions": [instr.text for instr in block_instrs],
                "successors": [],
                "predecessors": [],
            }
        )

    edges: list[dict[str, str]] = []
    block_by_id = {block["id"]: block for block in blocks}
    for pos, (start, end) in enumerate(block_ranges):
        src = index_to_block[start]
        last = instructions[end]
        opcode = opcode_of(last.text)
        target = branch_target(last.text)
        predicated = is_predicated(last.text)
        fallthrough = index_to_block.get(block_ranges[pos + 1][0]) if pos + 1 < len(block_ranges) else None

        if opcode in BRANCH_OPS:
            if target and target in label_to_index:
                _add_edge(edges, src, index_to_block[label_to_index[target]], "branch")
            if predicated and fallthrough:
                _add_edge(edges, src, fallthrough, "fallthrough")
        elif opcode in TERMINAL_OPS:
            continue
        elif opcode in CALL_OPS:
            if target and target in label_to_index:
                _add_edge(edges, src, index_to_block[label_to_index[target]], "call")
            if fallthrough:
                _add_edge(edges, src, fallthrough, "fallthrough")
        elif fallthrough:
            _add_edge(edges, src, fallthrough, "fallthrough")

    for edge in edges:
        block_by_id[edge["src"]]["successors"].append(edge["dst"])
        block_by_id[edge["dst"]]["predecessors"].append(edge["src"])

    return {"basic_blocks": blocks, "edges": edges, "entry": "B0"}


def parse_kernel_lines(lines: list[str]) -> tuple[list[ParsedInstruction], dict[str, int]]:
    pending_labels: list[str] = []
    instructions: list[ParsedInstruction] = []
    label_to_index: dict[str, int] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":"):
            pending_labels.append(line[:-1])
            continue
        idx = len(instructions)
        instructions.append(ParsedInstruction(idx, line, pending_labels))
        for label in pending_labels:
            label_to_index[label] = idx
        pending_labels = []
    return instructions, label_to_index


def opcode_of(instr: str) -> str:
    body = re.sub(r"^@!?\w+\s+", "", instr.strip())
    if not body:
        return ""
    return body.split()[0].split(".")[0].upper()


def branch_target(instr: str) -> str | None:
    parts = instr.replace(",", " ").split()
    for token in reversed(parts):
        token = token.strip(";")
        if re.fullmatch(r"L_[A-Za-z0-9_]+", token):
            return token
    return None


def is_predicated(instr: str) -> bool:
    return bool(re.match(r"^@!?\w+\s+", instr.strip()))


def _add_edge(edges: list[dict[str, str]], src: str, dst: str, edge_type: str) -> None:
    edge = {"src": src, "dst": dst, "type": edge_type}
    if edge not in edges:
        edges.append(edge)
