"""Main computational loop extraction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .cfg import branch_target, opcode_of, parse_kernel_lines
from .split_kernels import count_instruction_lines


SMALL_KERNEL_THRESHOLD = 256
MIN_REGION_INSTRUCTIONS = 128
TARGET_REGION_INSTRUCTIONS = 512
MAX_REGION_INSTRUCTIONS = 768

INTEGER_PREFIXES = ("IADD", "IMAD", "IMUL", "ISCADD", "LEA", "LOP")
BITWISE_PREFIXES = ("LOP3", "AND", "OR", "XOR", "BFE", "BFI", "POPC", "FLO")
SHIFT_PREFIXES = ("SHF", "SHL", "SHR")
PREDICATE_PREFIXES = ("ISETP", "PSETP", "FSETP")
MEMORY_PREFIXES = ("LD", "LDG", "LDS", "LDC", "ST", "STG", "STS", "ATOM", "RED")
CONTROL_PREFIXES = ("BRA", "JMP", "EXIT", "RET", "CALL")
NOISE_PREFIXES = ("NOP", "BRA", "JMP", "EXIT", "RET", "CALL")


def extract_main_loop_for_kernel(kernel_dir: Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    lines = (kernel_dir / "kernel.sass").read_text(encoding="utf-8").splitlines()
    main_loop, summary = extract_main_loop(lines, cfg)
    (kernel_dir / "main_loop.sass").write_text("\n".join(main_loop).rstrip() + "\n", encoding="utf-8")
    with (kernel_dir / "loop_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return summary


def extract_main_loop(lines: list[str], cfg: dict[str, Any] | None = None) -> tuple[list[str], dict[str, Any]]:
    instructions, label_to_index = parse_kernel_lines(lines)
    physical_instr_lines = [line for line in lines if line.strip() and not line.strip().endswith(":")]
    candidates: list[dict[str, Any]] = []

    for instr in instructions:
        target = branch_target(instr.text)
        if not target or target not in label_to_index:
            continue
        start_idx = label_to_index[target]
        if start_idx >= instr.index:
            continue
        body_instrs = instructions[start_idx : instr.index + 1]
        body_lines = [item.text for item in body_instrs]
        mix = opcode_mix(body_lines)
        score = score_loop(body_lines, mix, branch_backedges=1, nesting_depth=0)
        candidates.append(
            {
                "loop_id": f"L{len(candidates)}",
                "start_instruction": start_idx,
                "end_instruction": instr.index,
                "instruction_count": len(body_instrs),
                "score": score,
                "opcode_mix": dict(mix),
                "body": body_lines,
            }
        )

    _apply_nesting_depth(candidates)

    if not candidates:
        return select_no_loop_fallback(physical_instr_lines, cfg)

    selected = max(candidates, key=lambda c: (c["score"], c["instruction_count"]))
    selected_density = selected["score"] / max(1, selected["instruction_count"])
    summary = {
        "num_basic_blocks": len((cfg or {}).get("basic_blocks", [])),
        "num_loops": len(candidates),
        "selected_loop_id": selected["loop_id"],
        "selected_loop_reason": "highest_weighted_compute_score",
        "selected_loop_instruction_count": selected["instruction_count"],
        "selected_loop_score": selected["score"],
        "fallback": None,
        "selected_loop_opcode_mix": selected["opcode_mix"],
        "selected_region_start_instruction": selected["start_instruction"],
        "selected_region_end_instruction": selected["end_instruction"],
        "selected_region_density": selected_density,
        "selected_region_reason": "natural_loop_backedge",
        "loops": [
            {
                "loop_id": c["loop_id"],
                "instruction_count": c["instruction_count"],
                "score": c["score"],
                "start_instruction": c["start_instruction"],
                "end_instruction": c["end_instruction"],
            }
            for c in candidates
        ],
    }
    return selected["body"], summary


def select_no_loop_fallback(
    physical_instr_lines: list[str],
    cfg: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    instruction_count = count_instruction_lines(physical_instr_lines)
    if instruction_count <= SMALL_KERNEL_THRESHOLD:
        score, density = score_region(physical_instr_lines)
        summary = fallback_summary(
            cfg,
            fallback="full_kernel_small",
            lines=physical_instr_lines,
            score=score,
            density=density,
            start_instruction=0 if instruction_count else None,
            end_instruction=instruction_count - 1 if instruction_count else None,
            reason="kernel_instruction_count_at_or_below_small_threshold",
        )
        return physical_instr_lines, summary

    candidates = compute_region_candidates(physical_instr_lines, cfg)
    if candidates:
        selected = select_best_region(candidates)
        summary = fallback_summary(
            cfg,
            fallback="compute_region",
            lines=selected["body"],
            score=selected["score"],
            density=selected["density"],
            start_instruction=selected["start_instruction"],
            end_instruction=selected["end_instruction"],
            reason="highest_scoring_compute_heavy_basic_block_span",
            extra={
                "selected_region_num_blocks": selected["num_blocks"],
                "selected_region_met_min_instruction_count": selected["meets_min"],
                "compute_region_candidates": len(candidates),
            },
        )
        return selected["body"], summary

    score, density = score_region(physical_instr_lines)
    summary = fallback_summary(
        cfg,
        fallback="full_kernel",
        lines=physical_instr_lines,
        score=score,
        density=density,
        start_instruction=0,
        end_instruction=instruction_count - 1,
        reason="no_nontrivial_compute_region_candidate",
    )
    return physical_instr_lines, summary


def fallback_summary(
    cfg: dict[str, Any] | None,
    fallback: str,
    lines: list[str],
    score: float,
    density: float,
    start_instruction: int | None,
    end_instruction: int | None,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_basic_blocks": len((cfg or {}).get("basic_blocks", [])),
        "num_loops": 0,
        "selected_loop_id": None,
        "selected_loop_reason": None,
        "selected_loop_instruction_count": count_instruction_lines(lines),
        "selected_loop_score": score,
        "fallback": fallback,
        "selected_loop_opcode_mix": dict(opcode_mix(lines)),
        "selected_region_start_instruction": start_instruction,
        "selected_region_end_instruction": end_instruction,
        "selected_region_density": density,
        "selected_region_reason": reason,
        "loops": [],
    }
    if extra:
        summary.update(extra)
    return summary


def compute_region_candidates(
    physical_instr_lines: list[str],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    blocks = source_order_blocks(physical_instr_lines, cfg)
    target_size = min(TARGET_REGION_INSTRUCTIONS, len(physical_instr_lines))
    min_size = min(MIN_REGION_INSTRUCTIONS, len(physical_instr_lines))
    max_size = min(MAX_REGION_INSTRUCTIONS, len(physical_instr_lines))
    candidates: list[dict[str, Any]] = []

    for start_pos, start_block in enumerate(blocks):
        if is_trivial_block(start_block["instructions"]):
            continue
        body: list[str] = []
        start_instruction = int(start_block["start_line"])
        end_instruction = start_instruction
        num_blocks = 0
        for block in blocks[start_pos:]:
            block_instructions = block["instructions"]
            if not body and is_trivial_block(block_instructions):
                break
            if body and is_trivial_block(block_instructions):
                break
            if body and len(body) + len(block_instructions) > max_size:
                break
            body.extend(block_instructions)
            trimmed_body = trim_trailing_noise(body)
            end_instruction = start_instruction + len(trimmed_body) - 1
            num_blocks += 1
            if not is_nontrivial_region(trimmed_body):
                continue
            score, density = score_region(trimmed_body)
            candidates.append(
                {
                    "body": list(trimmed_body),
                    "score": score,
                    "density": density,
                    "start_instruction": start_instruction,
                    "end_instruction": end_instruction,
                    "instruction_count": len(trimmed_body),
                    "num_blocks": num_blocks,
                    "meets_min": len(trimmed_body) >= min_size,
                    "target_distance": abs(len(trimmed_body) - target_size),
                }
            )
    return candidates


def select_best_region(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if candidate["meets_min"]]
    pool = eligible or candidates
    return max(
        pool,
        key=lambda c: (
            c["score"],
            c["density"],
            -c["target_distance"],
            c["instruction_count"],
            -c["start_instruction"],
        ),
    )


def source_order_blocks(
    physical_instr_lines: list[str],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg_blocks = (cfg or {}).get("basic_blocks") or []
    if cfg_blocks:
        blocks = []
        for block in sorted(cfg_blocks, key=lambda item: int(item.get("start_line", 0))):
            instructions = [line for line in block.get("instructions", []) if line.strip()]
            if not instructions:
                continue
            start_line = int(block.get("start_line", 0))
            blocks.extend(split_oversized_block(start_line, instructions))
        if blocks:
            return blocks
    return [
        {
            "start_line": idx,
            "end_line": idx,
            "instructions": [line],
        }
        for idx, line in enumerate(physical_instr_lines)
        if line.strip()
    ]


def split_oversized_block(start_line: int, instructions: list[str]) -> list[dict[str, Any]]:
    if len(instructions) <= MAX_REGION_INSTRUCTIONS:
        return [
            {
                "start_line": start_line,
                "end_line": start_line + len(instructions) - 1,
                "instructions": instructions,
            }
        ]

    chunks: list[dict[str, Any]] = []
    chunk_size = TARGET_REGION_INSTRUCTIONS
    for offset in range(0, len(instructions), chunk_size):
        chunk = instructions[offset : offset + chunk_size]
        chunks.append(
            {
                "start_line": start_line + offset,
                "end_line": start_line + offset + len(chunk) - 1,
                "instructions": chunk,
            }
        )
    return chunks


def opcode_mix(lines: list[str]) -> Counter[str]:
    mix: Counter[str] = Counter()
    for line in lines:
        op = opcode_of(line)
        if op:
            mix[op] += 1
    return mix


def score_region(lines: list[str]) -> tuple[float, float]:
    mix = opcode_mix(lines)
    score = score_loop(lines, mix, branch_backedges=0, nesting_depth=0)
    score -= 1.5 * _count_prefixes(mix, NOISE_PREFIXES)
    score -= 2.0 * _count_prefixes(mix, CONTROL_PREFIXES)
    score -= 4.0 * trivial_control_instruction_count(lines)
    instruction_count = max(1, count_instruction_lines(lines))
    density = score / instruction_count
    return score, density


def is_nontrivial_region(lines: list[str]) -> bool:
    if not lines:
        return False
    useful = sum(1 for line in lines if is_useful_opcode(opcode_of(line)))
    return useful >= 2 and useful / len(lines) >= 0.20


def trim_trailing_noise(lines: list[str]) -> list[str]:
    trimmed = list(lines)
    while trimmed and opcode_of(trimmed[-1]).startswith(NOISE_PREFIXES):
        trimmed.pop()
    return trimmed


def is_trivial_block(lines: list[str]) -> bool:
    if not lines:
        return True
    opcodes = [opcode_of(line) for line in lines if line.strip()]
    if not opcodes:
        return True
    useful = sum(1 for opcode in opcodes if is_useful_opcode(opcode))
    if useful:
        return False
    return all(opcode.startswith(NOISE_PREFIXES) for opcode in opcodes)


def is_useful_opcode(opcode: str) -> bool:
    return opcode.startswith(
        INTEGER_PREFIXES
        + BITWISE_PREFIXES
        + SHIFT_PREFIXES
        + PREDICATE_PREFIXES
        + MEMORY_PREFIXES
    )


def trivial_control_instruction_count(lines: list[str]) -> int:
    count = 0
    for line in lines:
        opcode = opcode_of(line)
        if opcode in {"NOP", "EXIT", "RET", "BRA", "CALL"}:
            count += 1
    return count


def score_loop(
    lines: list[str],
    mix: Counter[str],
    branch_backedges: int,
    nesting_depth: int,
) -> float:
    instruction_count = len([line for line in lines if line.strip()])
    return (
        1.0 * instruction_count
        + 2.0 * _count_prefixes(mix, INTEGER_PREFIXES)
        + 2.0 * _count_prefixes(mix, BITWISE_PREFIXES)
        + 1.5 * _count_prefixes(mix, SHIFT_PREFIXES)
        + 1.5 * _count_prefixes(mix, PREDICATE_PREFIXES)
        + 1.0 * _count_prefixes(mix, MEMORY_PREFIXES)
        + 2.0 * branch_backedges
        + 1.0 * nesting_depth
    )


def _count_prefixes(mix: Counter[str], prefixes: tuple[str, ...]) -> int:
    return sum(count for opcode, count in mix.items() if opcode.startswith(prefixes))


def _apply_nesting_depth(candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        depth = 0
        for other in candidates:
            if other is candidate:
                continue
            if (
                other["start_instruction"] <= candidate["start_instruction"]
                and candidate["end_instruction"] <= other["end_instruction"]
            ):
                depth += 1
        candidate["nesting_depth"] = depth
        mix = Counter(candidate["opcode_mix"])
        candidate["score"] = score_loop(candidate["body"], mix, 1, depth)
