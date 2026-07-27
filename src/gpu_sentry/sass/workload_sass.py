"""Create workload.sass in runtime launch order."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ingest import read_jsonl
from .sass_tokens import (
    CONTENT_TOKEN_BUDGET,
    KERNEL_BOUNDARY,
    sass_token_count,
    truncate_lines_from_front_to_token_budget,
    truncate_lines_to_token_budget,
)
from .split_kernels import count_instruction_lines, load_kernel_metadata
from .static_features import bitwise_integer_instruction_ratio


class WorkloadSassError(RuntimeError):
    """Raised when workload.sass cannot be produced."""


def build_workload_sass(
    workload_dir: Path,
    max_launches: int = 16,
    short_kernel_threshold: int = 256,
) -> dict[str, Any]:
    launches = read_jsonl(workload_dir / "launches.jsonl", limit=max_launches)
    result = render_workload_sass(
        workload_dir,
        launches,
        short_kernel_threshold=short_kernel_threshold,
    )
    (workload_dir / "workload.sass").write_text(result["text"], encoding="utf-8")
    return {"included_launches": result["included_launches"], "missing_launches": result["missing_launches"]}


def render_workload_sass(
    workload_dir: Path,
    launches: list[dict[str, Any]],
    short_kernel_threshold: int = 256,
    content_token_budget: int = CONTENT_TOKEN_BUDGET,
    fragment_cache: dict[tuple[Any, ...], Any] | None = None,
    front_clip_to_budget: bool = False,
) -> dict[str, Any]:
    fragments = render_workload_sass_fragments(
        workload_dir,
        launches,
        short_kernel_threshold=short_kernel_threshold,
        content_token_budget=content_token_budget,
        fragment_cache=fragment_cache,
    )
    output_lines: list[str] = []
    token_cost = 0
    for fragment in fragments["fragments"]:
        output_lines.extend(fragment["lines"])
        token_cost += int(fragment["token_cost"])
    if not output_lines:
        raise WorkloadSassError("no launched kernel could be mapped to normalized SASS")
    pre_clip_token_cost = token_cost
    front_clipped = False
    if front_clip_to_budget and token_cost > content_token_budget:
        output_lines = truncate_lines_from_front_to_token_budget(output_lines, content_token_budget)
        token_cost = sass_token_count(output_lines)
        front_clipped = True
    if front_clip_to_budget and token_cost > content_token_budget:
        raise WorkloadSassError(
            f"rendered SASS exceeds token budget: {token_cost} > {content_token_budget}"
        )

    return {
        "text": "\n".join(output_lines).rstrip() + "\n",
        "included_launches": len(fragments["fragments"]),
        "missing_launches": fragments["missing_launches"],
        "token_cost": token_cost,
        "pre_clip_token_cost": pre_clip_token_cost,
        "front_clipped": front_clipped,
        "clipped_token_count": max(0, pre_clip_token_cost - token_cost),
    }


def render_workload_sass_fragments(
    workload_dir: Path,
    launches: list[dict[str, Any]],
    short_kernel_threshold: int = 256,
    content_token_budget: int = CONTENT_TOKEN_BUDGET,
    fragment_cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    kernel_dirs = load_kernel_metadata(workload_dir / "kernels")
    by_name = kernel_dirs_by_name(kernel_dirs)
    fragments: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for launch in launches:
        try:
            fragments.append(render_prepared_l0_fragment(launch))
            continue
        except WorkloadSassError:
            pass
        try:
            fragments.append(
                render_launch_fragment(
                    workload_dir,
                    launch,
                    short_kernel_threshold=short_kernel_threshold,
                    content_token_budget=content_token_budget,
                    kernel_dirs=kernel_dirs,
                    by_name=by_name,
                    fragment_cache=fragment_cache,
                )
            )
            continue
        except WorkloadSassError as exc:
            missing.append(
                {
                    "kernel_name": launch.get("kernel_name"),
                    "code_id": launch.get("code_id"),
                    "reason": str(exc),
                }
            )

    if not fragments:
        raise WorkloadSassError("no launched kernel could be mapped to normalized SASS")

    return {"fragments": fragments, "missing_launches": missing}


def prepare_l0_launches(
    workload_dir: Path,
    launches: list[dict[str, Any]],
    short_kernel_threshold: int = 256,
    content_token_budget: int = CONTENT_TOKEN_BUDGET,
    fragment_cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    kernel_dirs = load_kernel_metadata(workload_dir / "kernels")
    by_name = kernel_dirs_by_name(kernel_dirs)
    artifacts: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    missing_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for launch in launches:
        key = launch_identity(launch)
        if key in artifacts or key in missing_by_key:
            continue
        try:
            fragments = render_l0_launch_segments(
                workload_dir,
                launch,
                content_token_budget=content_token_budget,
                kernel_dirs=kernel_dirs,
                by_name=by_name,
                fragment_cache=fragment_cache,
            )
        except WorkloadSassError as exc:
            missing_by_key[key] = {
                "kernel_name": launch.get("kernel_name"),
                "code_id": launch.get("code_id"),
                "reason": str(exc),
            }
            continue
        artifacts[key] = [l0_artifact_from_fragment(fragment) for fragment in fragments]

    max_ratio = max(
        (
            float(artifact["l0_bitwise_integer_ratio"])
            for artifact_list in artifacts.values()
            for artifact in artifact_list
        ),
        default=0.0,
    )
    ready: list[dict[str, Any]] = []
    for launch in launches:
        artifact_list = artifacts.get(launch_identity(launch))
        if artifact_list is None:
            continue
        for artifact in artifact_list:
            ready_launch = dict(launch)
            ready_launch.update(artifact)
            ready.append(ready_launch)
    return {
        "launches": ready,
        "missing_launches": list(missing_by_key.values()),
        "unique_kernel_count": len(artifacts),
        "logical_launch_count": len(ready),
        "unique_segment_count": sum(len(value) for value in artifacts.values()),
        "max_bitwise_integer_ratio": max_ratio,
        "gate_short_circuit": False,
    }


def launch_identity(launch: dict[str, Any]) -> tuple[Any, Any]:
    return launch.get("kernel_name"), launch.get("code_id")


def render_launch_fragment(
    workload_dir: Path,
    launch: dict[str, Any],
    short_kernel_threshold: int = 256,
    content_token_budget: int = CONTENT_TOKEN_BUDGET,
    kernel_dirs: dict[tuple[str, int], Path] | None = None,
    by_name: dict[str, Path] | None = None,
    fragment_cache: dict[tuple[Any, ...], Any] | None = None,
) -> dict[str, Any]:
    kernel_dir = resolve_kernel_dir(workload_dir, launch, kernel_dirs=kernel_dirs, by_name=by_name)
    if kernel_dir is None:
        raise WorkloadSassError("normalized SASS missing")
    cache_key = (str(kernel_dir), int(short_kernel_threshold), int(content_token_budget))
    if fragment_cache is not None and cache_key in fragment_cache:
        cached = fragment_cache[cache_key]
        return {**cached, "launch": launch, "lines": list(cached["lines"])}

    kernel_lines = _read_nonempty(kernel_dir / "kernel.normalized.sass")
    main_loop_lines = _read_nonempty(kernel_dir / "main_loop.normalized.sass")
    if not kernel_lines and not main_loop_lines:
        raise WorkloadSassError("empty normalized SASS")
    body_budget = max(0, int(content_token_budget) - 1)
    chosen, mode = select_render_lines(
        kernel_lines,
        main_loop_lines or kernel_lines,
        short_kernel_threshold=short_kernel_threshold,
        body_token_budget=body_budget,
    )
    if not chosen:
        raise WorkloadSassError("empty rendered SASS")
    lines = [*chosen, KERNEL_BOUNDARY]
    token_cost = sass_token_count(lines)
    if token_cost > content_token_budget:
        chosen = truncate_lines_to_token_budget(chosen, body_budget)
        lines = [*chosen, KERNEL_BOUNDARY]
        token_cost = sass_token_count(lines)
    if token_cost > content_token_budget or not chosen:
        raise WorkloadSassError("rendered SASS exceeds token budget")
    fragment = {
        "launch": launch,
        "lines": lines,
        "token_cost": token_cost,
        "kernel_id": kernel_id_for_launch(launch, kernel_dir),
        "bitwise_integer_instruction_ratio": bitwise_integer_instruction_ratio(chosen),
        "rendered_instruction_count": count_instruction_lines(chosen),
        "render_mode": mode,
    }
    if fragment_cache is not None:
        fragment_cache[cache_key] = {**fragment, "launch": None, "lines": list(lines)}
    return fragment


def render_l0_launch_segments(
    workload_dir: Path,
    launch: dict[str, Any],
    content_token_budget: int = CONTENT_TOKEN_BUDGET,
    kernel_dirs: dict[tuple[str, int], Path] | None = None,
    by_name: dict[str, Path] | None = None,
    fragment_cache: dict[tuple[Any, ...], Any] | None = None,
) -> list[dict[str, Any]]:
    """Render scheduler-visible main-loop segments for one kernel launch."""

    kernel_dir = resolve_kernel_dir(workload_dir, launch, kernel_dirs=kernel_dirs, by_name=by_name)
    if kernel_dir is None:
        raise WorkloadSassError("normalized SASS missing")
    cache_key = ("l0_segments", str(kernel_dir), int(content_token_budget))
    if fragment_cache is not None and cache_key in fragment_cache:
        cached = fragment_cache[cache_key]
        return [
            {**fragment, "launch": launch, "lines": list(fragment["lines"])}
            for fragment in cached
        ]

    main_loop_lines = _read_nonempty(kernel_dir / "main_loop.normalized.sass")
    if not main_loop_lines:
        raise WorkloadSassError("empty main loop")
    body_budget = max(0, int(content_token_budget) - 1)
    segments = split_lines_to_token_budget(main_loop_lines, body_budget)
    parent_kernel_name = kernel_name_for_launch(launch, kernel_dir)
    parent_kernel_id = kernel_id_for_launch(launch, kernel_dir)
    segment_count = len(segments)
    fragments: list[dict[str, Any]] = []
    for offset, segment_lines in enumerate(segments, start=1):
        segment_name = (
            f"{parent_kernel_name}_seg{offset}"
            if segment_count > 1
            else parent_kernel_name
        )
        lines = [*segment_lines, KERNEL_BOUNDARY]
        token_cost = sass_token_count(lines)
        if token_cost > content_token_budget:
            raise WorkloadSassError("rendered SASS segment exceeds token budget")
        fragments.append(
            {
                "launch": launch,
                "lines": lines,
                "token_cost": token_cost,
                "kernel_id": segment_name,
                "parent_kernel_id": parent_kernel_id,
                "parent_kernel_name": parent_kernel_name,
                "subkernel_index": offset,
                "subkernel_count": segment_count,
                "subkernel_name": segment_name,
                "bitwise_integer_instruction_ratio": bitwise_integer_instruction_ratio(segment_lines),
                "rendered_instruction_count": count_instruction_lines(segment_lines),
                "render_mode": "main_loop_segment" if segment_count > 1 else "main_loop",
            }
        )
    if fragment_cache is not None:
        fragment_cache[cache_key] = [
            {**fragment, "launch": None, "lines": list(fragment["lines"])}
            for fragment in fragments
        ]
    return fragments


def l0_artifact_from_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    return {
        "l0_kernel_id": fragment["kernel_id"],
        "l0_parent_kernel_id": fragment["parent_kernel_id"],
        "l0_parent_kernel_name": fragment["parent_kernel_name"],
        "l0_subkernel_index": fragment["subkernel_index"],
        "l0_subkernel_count": fragment["subkernel_count"],
        "l0_subkernel_name": fragment["subkernel_name"],
        "l0_token_cost": fragment["token_cost"],
        "l0_bitwise_integer_ratio": fragment["bitwise_integer_instruction_ratio"],
        "l0_rendered_instruction_count": fragment["rendered_instruction_count"],
        "l0_render_mode": fragment["render_mode"],
        "l0_segment_lines": list(fragment["lines"]),
    }


def render_prepared_l0_fragment(launch: dict[str, Any]) -> dict[str, Any]:
    lines = launch.get("l0_segment_lines")
    if not isinstance(lines, list) or not lines:
        raise WorkloadSassError("prepared L0 segment lines missing")
    return {
        "launch": launch,
        "lines": [str(line) for line in lines],
        "token_cost": int(launch["l0_token_cost"]),
        "kernel_id": str(launch["l0_kernel_id"]),
        "bitwise_integer_instruction_ratio": float(launch.get("l0_bitwise_integer_ratio", 0.0)),
        "rendered_instruction_count": int(launch.get("l0_rendered_instruction_count", 0)),
        "render_mode": str(launch.get("l0_render_mode", "main_loop_segment")),
    }


def split_lines_to_token_budget(lines: list[str], body_token_budget: int) -> list[list[str]]:
    if body_token_budget <= 0:
        raise WorkloadSassError("token budget leaves no room for main-loop instructions")
    segments: list[list[str]] = []
    current: list[str] = []
    cost = 0
    for line in lines:
        stripped = str(line).strip()
        if not stripped:
            continue
        line_cost = sass_token_count(stripped)
        if line_cost > body_token_budget:
            raise WorkloadSassError("single main-loop instruction exceeds token budget")
        if current and cost + line_cost > body_token_budget:
            segments.append(current)
            current = []
            cost = 0
        current.append(stripped)
        cost += line_cost
    if current:
        segments.append(current)
    if not segments:
        raise WorkloadSassError("empty main loop")
    return segments


def select_render_lines(
    kernel_lines: list[str],
    main_loop_lines: list[str],
    short_kernel_threshold: int,
    body_token_budget: int,
) -> tuple[list[str], str]:
    if count_instruction_lines(kernel_lines) <= short_kernel_threshold:
        kernel_cost = sass_token_count(kernel_lines)
        if kernel_cost <= body_token_budget:
            return list(kernel_lines), "full_kernel"
    bounded_loop = truncate_lines_to_token_budget(main_loop_lines, body_token_budget)
    return bounded_loop, "main_loop"


def resolve_kernel_dir(
    workload_dir: Path,
    launch: dict[str, Any],
    kernel_dirs: dict[tuple[str, int], Path] | None = None,
    by_name: dict[str, Path] | None = None,
) -> Path | None:
    kernel_dirs = kernel_dirs or load_kernel_metadata(workload_dir / "kernels")
    by_name = by_name or kernel_dirs_by_name(kernel_dirs)
    key = (launch.get("kernel_name"), launch.get("code_id"))
    resolved = kernel_dirs.get(key) or by_name.get(str(launch.get("kernel_name")))
    if resolved is not None:
        return resolved
    l0_kernel_id = launch.get("l0_kernel_id")
    if l0_kernel_id is None:
        return None
    return kernel_dirs_by_l0_kernel_id(kernel_dirs).get(str(l0_kernel_id))


def kernel_dirs_by_name(kernel_dirs: dict[tuple[str, int], Path]) -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    for (kernel_name, _code_id), path in kernel_dirs.items():
        by_name.setdefault(kernel_name, path)
    return by_name


def kernel_dirs_by_l0_kernel_id(kernel_dirs: dict[tuple[str, int], Path]) -> dict[str, Path]:
    by_l0_kernel_id: dict[str, Path] = {}
    for (_kernel_name, _code_id), path in kernel_dirs.items():
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = load_kernel_metadata_value(metadata_path)
        except (OSError, ValueError):
            continue
        kernel_id = kernel_id_for_launch(metadata, path)
        by_l0_kernel_id.setdefault(kernel_id, path)
    return by_l0_kernel_id


def kernel_id_for_launch(launch: dict[str, Any], kernel_dir: Path | None = None) -> str:
    code_id = launch.get("code_id")
    kernel_name = str(launch.get("kernel_name") or "")
    if kernel_dir is not None:
        metadata_path = kernel_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = load_kernel_metadata_value(metadata_path)
                code_id = metadata.get("code_id", code_id)
                kernel_name = str(metadata.get("kernel_name") or kernel_name)
            except (OSError, ValueError):
                pass
    return f"{code_id}:{kernel_name}"


def kernel_name_for_launch(launch: dict[str, Any], kernel_dir: Path | None = None) -> str:
    kernel_name = str(launch.get("kernel_name") or "")
    if kernel_dir is not None:
        metadata_path = kernel_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = load_kernel_metadata_value(metadata_path)
                kernel_name = str(metadata.get("kernel_name") or kernel_name)
            except (OSError, ValueError):
                pass
    return kernel_name


def load_kernel_metadata_value(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("kernel metadata is not an object")
    return value


def _read_nonempty(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
