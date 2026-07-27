"""Online GPU-Sentry processor.

The processor receives collector events over a Unix-domain framed JSON stream,
analyzes code objects asynchronously, runs incremental L0 windowing, and emits
L1 detection verdicts back to the collector.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import threading
import time
from collections import Counter
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("GPU_SENTRY_DISABLE", "1")

from gpu_sentry.sass.cfg import build_cfg_for_kernel
from gpu_sentry.sass.disassemble import disassemble_code_objects, find_cuda_tools
from gpu_sentry.sass.l0_config import load_l0_config
from gpu_sentry.sass.l0_windows import L0Window, L0WindowScheduler
from gpu_sentry.sass.loop_extract import extract_main_loop_for_kernel
from gpu_sentry.sass.manifest import write_json
from gpu_sentry.sass.normalize import normalize_kernel_files
from gpu_sentry.sass.split_kernels import (
    parse_disassembly,
    render_kernel_sass,
    select_fallback_function,
    unique_safe_kernel_dir,
)
from gpu_sentry.sass.sass_tokens import sass_token_count
from gpu_sentry.sass.workload_sass import (
    l0_artifact_from_fragment,
    render_l0_launch_segments,
    render_workload_sass,
)

from gpu_sentry.model.config import load_run_config
from gpu_sentry.model.data import encode_content_ids
from gpu_sentry.model.metrics import softmax_rows
from gpu_sentry.model.tokenization import load_sass_tokenizer

from .config import DEFAULT_ONLINE_CONFIG, load_online_config, repo_path
from .framing import FrameError, recv_frame, send_frame
from .policy import DecisionPolicy, load_policy

ANSI_BOLD_CYAN = "\033[1;36m"
ANSI_BOLD_RED = "\033[1;31m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"


@dataclass
class KernelArtifact:
    code_id: int
    kernel_name: str
    kernel_id: str
    token_cost: int
    bitwise_integer_ratio: float
    rendered_instruction_count: int
    render_mode: str
    segments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: str
    session_dir: Path
    l0_scheduler: L0WindowScheduler
    policy: DecisionPolicy
    process_info: dict[str, Any] | None = None
    launches: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[tuple[int, str], KernelArtifact] = field(default_factory=dict)
    artifacts_by_code: dict[int, list[KernelArtifact]] = field(default_factory=dict)
    dropped_unready_launches: int = 0
    emitted_terminal_verdict: bool = False
    next_policy_window_index: int = 0
    pending_policy_verdicts: dict[int, dict[str, Any]] = field(default_factory=dict)
    skipped_policy_window_indices: set[int] = field(default_factory=set)
    signature_prediction_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_signature_windows: dict[str, list[L0Window]] = field(default_factory=dict)
    launch_batches_seen: int = 0
    active: bool = True
    code_futures: list[Future[Any]] = field(default_factory=list)
    inference_futures: list[Future[Any]] = field(default_factory=list)


class L1Classifier:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.available = False
        self.error: str | None = None
        self.tokenizer = None
        self.model = None
        self.torch = None
        self.run_config = None
        self.device = "cpu"
        self._load()

    def predict(self, text: str, workload: str) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.error or "L1 classifier unavailable")
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.torch is not None
        assert self.run_config is not None

        content_window = self.run_config.max_seq_length - 2
        content_ids = encode_content_ids(self.tokenizer, text)
        if len(content_ids) > content_window:
            raise RuntimeError(
                f"single-chunk L1 input exceeded budget: tokens={len(content_ids)} "
                f"content_window={content_window}"
            )
        log(
            f"L1 start workload={workload} device={self.device} chars={len(text)} "
            f"tokens={len(content_ids)}"
        )
        rows = [[
            int(self.tokenizer.cls_token_id),
            *content_ids,
            int(self.tokenizer.sep_token_id),
        ]]
        probabilities: list[list[float]] = []
        batch_size = int(self.config["l1"].get("batch_size", 1))
        with self.torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start : start + batch_size]
                encoded = self.tokenizer.pad(
                    {"input_ids": batch_rows, "attention_mask": [[1] * len(row) for row in batch_rows]},
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits.detach().cpu().numpy()
                probabilities.extend(softmax_rows(logits))
        prediction = self._aggregate(workload, probabilities)
        log(
            "L1 done "
            f"workload={workload} pred={prediction['pred_label']} "
            f"suspicious={prediction['suspicious']} "
            f"p_mining_max={prediction['mining_probability_max']:.6f} "
            f"reason={prediction['suspicious_reason']}"
        )
        return prediction

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification

            run_config = load_run_config(repo_path(self.config["l1"]["training_config_path"]))
            checkpoint = repo_path(self.config["l1"]["checkpoint_path"])
            tokenizer_dir = checkpoint if (checkpoint / "tokenizer.json").exists() else run_config.paths.tokenizer_dir
            tokenizer = load_sass_tokenizer(tokenizer_dir)
            model_config = AutoConfig.from_pretrained(checkpoint)
            model_config.reference_compile = False
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=model_config)
            requested_device = str(self.config["l1"]["device"])
            if requested_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = requested_device
            model.to(device)
            model.eval()
            self.torch = torch
            self.run_config = run_config
            self.tokenizer = tokenizer
            self.model = model
            self.device = device
            self.available = True
            log(f"L1 loaded checkpoint={checkpoint} tokenizer={tokenizer_dir} device={device}")
        except Exception as exc:  # noqa: BLE001 - fail-open service initialization
            self.available = False
            self.error = f"L1 unavailable: {type(exc).__name__}: {exc}"
            log(self.error)

    def _aggregate(self, workload: str, probabilities: list[list[float]]) -> dict[str, Any]:
        if not probabilities:
            raise RuntimeError("no L1 probabilities produced")
        assert self.run_config is not None
        num_labels = len(probabilities[0])
        mean_probs = [sum(row[idx] for row in probabilities) / len(probabilities) for idx in range(num_labels)]
        pred_id = max(range(num_labels), key=lambda idx: mean_probs[idx])
        mining_label = "mining"
        mining_id = self.run_config.label2id.get(mining_label)
        if mining_id is None:
            raise RuntimeError(f"configured mining label {mining_label!r} not present in L1 config")
        values = [row[mining_id] for row in probabilities]
        mining_mean = mean_probs[mining_id]
        mining_max = max(values)
        return {
            "workload": workload,
            "pred_label": self.run_config.id2label[pred_id],
            "pred_id": pred_id,
            "probabilities": mean_probs,
            "probabilities_by_label": {
                self.run_config.id2label[idx]: value for idx, value in enumerate(mean_probs)
            },
            "mining_probability_mean": mining_mean,
            "mining_probability_max": mining_max,
            "num_chunks": len(probabilities),
            "suspicious": False,
            "suspicious_reason": "awaiting_rolling_policy",
        }


class OnlineProcessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.frame_max_bytes = int(config["transport"]["frame_max_bytes"])
        self.work_dir = repo_path(config["storage"]["processor_work_dir"])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.l0_config = load_l0_config(repo_path(config["l0"]["config_path"]))
        self.tools = find_cuda_tools()
        self.executor = ThreadPoolExecutor(max_workers=int(config["kernel_analysis"]["workers"]))
        self.event_executor = ThreadPoolExecutor(max_workers=1)
        self.classifier = L1Classifier(config)
        self.inference_executor = ThreadPoolExecutor(max_workers=int(config["l1"]["inference_workers"]))
        self.sessions: dict[str, SessionState] = {}
        self.outbox: list[dict[str, Any]] = []
        self.lock = threading.RLock()
        tool_summary = ", ".join(f"{name}={path}" for name, path in sorted(self.tools.items())) or "none"
        log(f"processor config={config.get('_config_path')} work_dir={self.work_dir}")
        log(f"CUDA tools: {tool_summary}")
        log(f"inference workers={config['l1']['inference_workers']}")

    def serve_forever(self) -> None:
        sock_path = Path(str(self.config["transport"]["processor_socket"]))
        if sock_path.exists():
            sock_path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        sock_path.chmod(0o600)
        server.listen(8)
        log(f"GPU-Sentry online processor listening on {sock_path}")
        try:
            while True:
                conn, _addr = server.accept()
                log("collector stream connected")
                thread = threading.Thread(target=self._handle_conn, args=(conn,), daemon=True)
                thread.start()
        finally:
            server.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(int(self.config["transport"]["read_timeout_ms"]) / 1000.0)
            if not self._send_or_close(
                conn,
                {
                    "type": "processor_hello",
                    "version": 1,
                    "l1_available": self.classifier.available,
                },
            ):
                return
            while True:
                try:
                    if not self._drain_outbox(conn):
                        return
                    message = recv_frame(conn, self.frame_max_bytes)
                    if message is None:
                        log("collector stream closed")
                        return
                    if message.get("type") == "session_end":
                        self._mark_session_ended(str(message.get("session_id") or ""), str(message.get("reason") or "collector_session_end"))
                    self.event_executor.submit(self._handle_message_to_outbox, message)
                except socket.timeout:
                    continue
                except FrameError as exc:
                    self._send_or_close(conn, {"type": "processor_error", "error": str(exc)})
                    log(f"collector stream frame error: {exc}")
                    return
                except (BrokenPipeError, ConnectionResetError, OSError):
                    log("collector stream disconnected")
                    return
                except Exception as exc:  # noqa: BLE001 - fail open and keep stream alive
                    if not self._send_or_close(conn, {"type": "processor_error", "error": f"{type(exc).__name__}: {exc}"}):
                        return

    def _send_or_close(self, conn: socket.socket, message: dict[str, Any]) -> bool:
        try:
            send_frame(conn, message)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _handle_message_to_outbox(self, message: dict[str, Any]) -> None:
        try:
            responses = self.handle_message(message)
        except Exception as exc:  # noqa: BLE001 - fail open and keep stream alive
            responses = [{"type": "processor_error", "error": f"{type(exc).__name__}: {exc}"}]
        if responses:
            self._enqueue_responses(responses)

    def handle_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        msg_type = message.get("type")
        session_id = str(message.get("session_id") or "")
        if msg_type == "collector_hello":
            return []
        if msg_type == "collector_shutdown":
            return []
        if not session_id:
            return [{"type": "processor_error", "error": "missing session_id"}]
        if msg_type == "session_end":
            self._mark_session_ended(session_id, str(message.get("reason") or "collector_session_end"))
            return []
        if msg_type == "kernel_launch_batch":
            launches = [self._normalize_launch(row, session_id) for row in message.get("launches") or []]
            with self.lock:
                state = self._session(session_id)
                if not state.active:
                    return []
                state.launch_batches_seen += 1
                if should_log_launch_batch(state.launch_batches_seen):
                    log(
                        f"launch_batch session={short_session(session_id)} "
                        f"batch={state.launch_batches_seen} count={len(launches)} total_before={len(state.launches)}"
                    )
            return self._handle_launches(session_id, launches)
        with self.lock:
            state = self._session(session_id)
            if not state.active and msg_type not in {"collector_shutdown"}:
                return []
            if msg_type == "process_info":
                state.process_info = dict(message.get("process_info") or {})
                info = state.process_info
                log(f"process_info session={short_session(session_id)} pid={info.get('pid')} exe={info.get('exe_path')}")
                return []
            if msg_type == "code_object":
                log(
                    f"code_object session={short_session(session_id)} "
                    f"code_id={message.get('code_id')} size={message.get('size')} path={message.get('path')}"
                )
                self._submit_code_analysis(state, message)
                return []
            if msg_type == "stats":
                log(
                    f"stats session={short_session(session_id)} "
                    f"queued={message.get('queued_events')} dropped={message.get('dropped_events')}"
                )
                return []
            if msg_type == "collector_shutdown":
                return []
            return [{"type": "processor_error", "session_id": session_id, "error": f"unknown message type: {msg_type}"}]

    def _session(self, session_id: str) -> SessionState:
        state = self.sessions.get(session_id)
        if state is None:
            session_dir = self.work_dir / _safe_name(session_id)
            (session_dir / "dumps").mkdir(parents=True, exist_ok=True)
            (session_dir / "kernels").mkdir(parents=True, exist_ok=True)
            state = SessionState(
                session_id,
                session_dir,
                L0WindowScheduler(self.l0_config),
                load_policy(self.config["verdict"]["policy"], self.config["verdict"]),
            )
            self.sessions[session_id] = state
            log(f"new session={short_session(session_id)} dir={session_dir}")
        return state

    def _mark_session_ended(self, session_id: str, reason: str) -> None:
        if not session_id:
            return
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                return
            state.active = False
            cancelled_code = sum(1 for future in state.code_futures if future.cancel())
            cancelled_l1 = sum(1 for future in state.inference_futures if future.cancel())
            self.outbox = [row for row in self.outbox if row.get("session_id") != session_id]
        log(
            f"session ended session={short_session(session_id)} reason={reason} "
            f"cancelled_code={cancelled_code} cancelled_l1={cancelled_l1}"
        )

    def _submit_code_analysis(self, state: SessionState, message: dict[str, Any]) -> None:
        log(f"code analysis queued session={short_session(state.session_id)} code_id={message.get('code_id')}")
        future = self.executor.submit(self._analyze_code_object, state.session_id, dict(message))
        state.code_futures.append(future)
        future.add_done_callback(lambda fut: self._finish_code_analysis(state.session_id, fut))

    def _analyze_code_object(self, session_id: str, message: dict[str, Any]) -> list[KernelArtifact]:
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                log(f"code analysis skipped session={short_session(session_id)} reason=session_inactive")
                return []
        code_id = int(message["code_id"])
        source_path = Path(str(message["path"]))
        started = time.monotonic()
        log(f"code analysis start session={short_session(session_id)} code_id={code_id} source={source_path}")
        if not source_path.exists():
            raise RuntimeError(f"missing code object path: {source_path}")
        dump_name = source_path.name
        dump_path = state.session_dir / "dumps" / dump_name
        if source_path.resolve() != dump_path.resolve():
            shutil.copy2(source_path, dump_path)
        code_map = {
            str(code_id): {
                "code_id": code_id,
                "code_type": message.get("code_type"),
                "source_path": str(source_path),
                "dump_path": f"dumps/{dump_name}",
                "sha256": message.get("sha256"),
                "size": message.get("size"),
            }
        }
        write_json(state.session_dir / "dumps" / "code_map.json", code_map)
        report = disassemble_code_objects(state.session_dir, code_map, self.tools)
        artifacts: list[KernelArtifact] = []
        used_kernel_dirs: set[str] = set()
        dir_counts: Counter[str] = Counter()
        for item in report.get("code_objects", []):
            if item.get("status") != "ok":
                continue
            disasm_path = state.session_dir / str(item["disassembly_output"])
            functions = parse_disassembly(disasm_path.read_text(encoding="utf-8"))
            for function_name, instructions in functions.items():
                safe_dir = unique_safe_kernel_dir(function_name, code_id, used_kernel_dirs, dir_counts)
                used_kernel_dirs.add(safe_dir)
                kernel_dir = state.session_dir / "kernels" / safe_dir
                kernel_dir.mkdir(parents=True, exist_ok=True)
                (kernel_dir / "kernel.sass").write_text(render_kernel_sass(instructions), encoding="utf-8")
                write_json(
                    kernel_dir / "metadata.json",
                    {
                        "kernel_name": function_name,
                        "disassembly_function": function_name,
                        "kernel_match": "online_all_functions",
                        "safe_kernel_dir": safe_dir,
                        "code_id": code_id,
                        "source_code_file": f"dumps/{dump_name}",
                        "instruction_count": len(instructions),
                        "launched": False,
                    },
                )
                cfg = build_cfg_for_kernel(kernel_dir)
                extract_main_loop_for_kernel(kernel_dir, cfg)
                normalize_kernel_files(kernel_dir)
                fragments = render_l0_launch_segments(
                    state.session_dir,
                    {"code_id": code_id, "kernel_name": function_name},
                    content_token_budget=self.l0_config.window.content_token_budget,
                )
                segments = [l0_artifact_from_fragment(fragment) for fragment in fragments]
                segment_count = max((int(segment.get("l0_subkernel_count", 1)) for segment in segments), default=1)
                artifacts.append(
                    KernelArtifact(
                        code_id=code_id,
                        kernel_name=function_name,
                        kernel_id=str(segments[0]["l0_parent_kernel_id"]),
                        token_cost=max(int(segment["l0_token_cost"]) for segment in segments),
                        bitwise_integer_ratio=max(
                            float(segment["l0_bitwise_integer_ratio"]) for segment in segments
                        ),
                        rendered_instruction_count=sum(
                            int(segment["l0_rendered_instruction_count"]) for segment in segments
                        ),
                        render_mode="main_loop_segment" if segment_count > 1 else "main_loop",
                        segments=segments,
                    )
                )
        if not artifacts:
            raise RuntimeError(f"code_id {code_id} produced no SASS kernel artifacts")
        elapsed = time.monotonic() - started
        log(
            f"code analysis done session={short_session(session_id)} code_id={code_id} "
            f"artifacts={len(artifacts)} elapsed={elapsed:.3f}s"
        )
        return artifacts

    def _finish_code_analysis(self, session_id: str, future: Future[list[KernelArtifact]]) -> None:
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None:
                return
            try:
                artifacts = future.result()
            except CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - fail-open analysis
                if not state.active:
                    return
                log(f"code analysis failed session={short_session(session_id)} error={exc}")
                return
            if not artifacts:
                return
            if not state.active:
                log(f"code analysis discarded session={short_session(session_id)} reason=session_inactive")
                return
            for artifact in artifacts:
                state.artifacts[(artifact.code_id, artifact.kernel_name)] = artifact
                state.artifacts_by_code.setdefault(artifact.code_id, []).append(artifact)
                log(
                    f"kernel ready session={short_session(session_id)} code_id={artifact.code_id} "
                    f"kernel={artifact.kernel_name} tokens={artifact.token_cost} "
                    f"bitwise_integer_ratio={artifact.bitwise_integer_ratio:.6f}"
                )

    def _enqueue_responses(self, responses: list[dict[str, Any]]) -> None:
        with self.lock:
            self.outbox.extend(responses)
            log(f"queued {len(responses)} response frame(s)")

    def _drain_outbox(self, conn: socket.socket) -> bool:
        while True:
            with self.lock:
                if not self.outbox:
                    return True
                response = self.outbox[0]
            if not self._send_or_close(conn, response):
                return False
            with self.lock:
                if self.outbox and self.outbox[0] is response:
                    self.outbox.pop(0)
                else:
                    try:
                        self.outbox.remove(response)
                    except ValueError:
                        pass

    def _handle_launches(self, session_id: str, launches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ready_windows: list[L0Window] = []
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                return []
            for launch in launches:
                artifact = self._artifact_for_launch(state, launch)
                if artifact is None:
                    state.dropped_unready_launches += 1
                    if state.dropped_unready_launches <= 5 or state.dropped_unready_launches % 100 == 0:
                        log(
                            f"kernel launch session={short_session(state.session_id)} "
                            f"code_id={launch.get('code_id')} kernel={launch.get('kernel_name')}"
                        )
                    continue
                for ready_launch in launches_for_artifact(launch, artifact):
                    state.launches.append(ready_launch)
                    windows = state.l0_scheduler.add_launch(ready_launch)
                    for window in windows:
                        max_ratio = float(window.features.get("max_bitwise_integer_ratio", 0.0))
                        log(
                            f"L0 window session={short_session(state.session_id)} id={window.window_id} "
                            f"type={window.window_type} launches={len(window.launches)} "
                            f"tokens={window.features.get('token_cost')} "
                            f"max_bitwise_integer_ratio={max_ratio:.6f} "
                            f"reasons={','.join(window.trigger_reason)}"
                        )
                        ready_windows.append(window)
        return self._run_windows(session_id, ready_windows)

    def _run_windows(self, session_id: str, windows: list[L0Window], pending: bool = False) -> list[dict[str, Any]]:
        for window in windows:
            prefix = "pending window" if pending else "window"
            log(f"{prefix} ready session={short_session(session_id)} id={window.window_id}; queueing policy score")
            self._submit_l1_window(session_id, window)
        return []

    def _submit_l1_window(self, session_id: str, window: L0Window) -> None:
        ready_responses: list[dict[str, Any]] = []
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active or state.emitted_terminal_verdict:
                return
            signature_key = _online_signature_cache_key(window)
            window_index = _window_index(window.window_id)
            if signature_key and signature_key in state.signature_prediction_cache:
                verdict = self._cached_l1_verdict(state, window, state.signature_prediction_cache[signature_key])
                if window_index is None:
                    self._apply_online_verdict_policy(state, verdict)
                    ready_responses.append(verdict)
                else:
                    state.pending_policy_verdicts[int(window_index)] = verdict
                    ready_responses.extend(self._release_ordered_policy_verdicts_locked(state))
            elif signature_key and signature_key in state.pending_signature_windows:
                state.pending_signature_windows[signature_key].append(window)
                log(
                    f"L1 window cached-pending session={short_session(session_id)} "
                    f"window={window.window_id} signature_occurrence={window.features.get('signature_occurrence_index')}"
                )
            else:
                if signature_key:
                    state.pending_signature_windows[signature_key] = []
                future = self.inference_executor.submit(self._run_l1_for_session, session_id, window)
                setattr(future, "_gpu_sentry_window_index", window_index)
                setattr(future, "_gpu_sentry_signature_key", signature_key)
                state.inference_futures.append(future)
                future.add_done_callback(lambda fut: self._finish_l1(session_id, fut))
                return
        if ready_responses:
            self._enqueue_responses(ready_responses)

    def _run_l1_for_session(self, session_id: str, window: L0Window) -> dict[str, Any] | None:
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                return None
            session_dir = state.session_dir
        return self._run_l1(session_id, session_dir, window)

    def _finish_l1(self, session_id: str, future: Future[dict[str, Any] | None]) -> None:
        window_index = getattr(future, "_gpu_sentry_window_index", None)
        signature_key = getattr(future, "_gpu_sentry_signature_key", None)
        try:
            verdict = future.result()
        except Exception as exc:  # noqa: BLE001 - fail-open inference
            log(f"L1 job failed session={short_session(session_id)} error={exc}")
            self._skip_policy_window(session_id, window_index)
            self._skip_pending_signature_windows(session_id, signature_key)
            return
        if verdict is None:
            self._skip_policy_window(session_id, window_index)
            self._skip_pending_signature_windows(session_id, signature_key)
            return
        ready_responses: list[dict[str, Any]] = []
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                log(
                    f"verdict discarded session={short_session(session_id)} "
                    f"window={verdict.get('window_id')} reason=session_inactive"
                )
                return
            pending_signature_windows: list[L0Window] = []
            if signature_key:
                state.signature_prediction_cache[str(signature_key)] = dict(verdict.get("prediction") or {})
                pending_signature_windows = state.pending_signature_windows.pop(str(signature_key), [])
            if window_index is None:
                self._apply_online_verdict_policy(state, verdict)
                ready_responses.append(verdict)
            else:
                state.pending_policy_verdicts[int(window_index)] = verdict
                for cached_window in pending_signature_windows:
                    cached_index = _window_index(cached_window.window_id)
                    cached_verdict = self._cached_l1_verdict(
                        state,
                        cached_window,
                        state.signature_prediction_cache[str(signature_key)],
                    )
                    if cached_index is None:
                        self._apply_online_verdict_policy(state, cached_verdict)
                        ready_responses.append(cached_verdict)
                    else:
                        state.pending_policy_verdicts[int(cached_index)] = cached_verdict
                ready_responses.extend(self._release_ordered_policy_verdicts_locked(state))
        if ready_responses:
            self._enqueue_responses(ready_responses)

    def _skip_pending_signature_windows(self, session_id: str, signature_key: str | None) -> None:
        if not signature_key:
            return
        ready_responses: list[dict[str, Any]] = []
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                return
            pending = state.pending_signature_windows.pop(str(signature_key), [])
            for window in pending:
                window_index = _window_index(window.window_id)
                if window_index is not None:
                    state.skipped_policy_window_indices.add(int(window_index))
            ready_responses.extend(self._release_ordered_policy_verdicts_locked(state))
        if ready_responses:
            self._enqueue_responses(ready_responses)

    def _skip_policy_window(self, session_id: str, window_index: int | None) -> None:
        if window_index is None:
            return
        ready_responses: list[dict[str, Any]] = []
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or not state.active:
                return
            state.skipped_policy_window_indices.add(int(window_index))
            ready_responses.extend(self._release_ordered_policy_verdicts_locked(state))
        if ready_responses:
            self._enqueue_responses(ready_responses)

    def _release_ordered_policy_verdicts_locked(self, state: SessionState) -> list[dict[str, Any]]:
        ready_responses: list[dict[str, Any]] = []
        while True:
            while state.next_policy_window_index in state.skipped_policy_window_indices:
                state.skipped_policy_window_indices.remove(state.next_policy_window_index)
                state.next_policy_window_index += 1
            verdict = state.pending_policy_verdicts.pop(state.next_policy_window_index, None)
            if verdict is None:
                break
            if state.emitted_terminal_verdict:
                log(
                    f"verdict discarded session={short_session(state.session_id)} "
                    f"window={verdict.get('window_id')} reason=terminal_already_emitted"
                )
                state.next_policy_window_index += 1
                continue
            self._apply_online_verdict_policy(state, verdict)
            if verdict.get("suspicious"):
                state.emitted_terminal_verdict = True
                state.pending_policy_verdicts.clear()
                state.skipped_policy_window_indices.clear()
            ready_responses.append(verdict)
            state.next_policy_window_index += 1
            if state.emitted_terminal_verdict:
                break
        return ready_responses

    def _run_l1(self, session_id: str, session_dir: Path, window: L0Window) -> dict[str, Any] | None:
        try:
            text = self._render_window_text(session_dir, window)
            prediction = self.classifier.predict(
                text,
                f"{session_id}__{window.window_id}",
            )
        except Exception as exc:  # noqa: BLE001 - fail open
            log(f"L1 failed session={short_session(session_id)} window={window.window_id} error={exc}")
            return None
        action = "terminate" if prediction["suspicious"] else "log"
        log(
            f"L1 window verdict session={short_session(session_id)} window={window.window_id} "
            f"action={action} suspicious={prediction['suspicious']} reason={prediction['suspicious_reason']}"
        )
        return {
            "type": "detection_verdict",
            "session_id": session_id,
            "window_id": window.window_id,
            "action": action,
            "suspicious": bool(prediction["suspicious"]),
            "reason": prediction["suspicious_reason"],
            "message": self.config["enforcement"]["message"],
            "prediction": prediction,
            "trigger_reason": window.trigger_reason,
            "l0_features": window.features,
            "signature_cache_hit": False,
            "signature_source_window_id": window.features.get("signature_source_window_id"),
        }

    def _cached_l1_verdict(
        self,
        state: SessionState,
        window: L0Window,
        raw_prediction: dict[str, Any],
    ) -> dict[str, Any]:
        prediction = dict(raw_prediction)
        action = "terminate" if prediction.get("suspicious") else "log"
        log(
            f"L1 window verdict session={short_session(state.session_id)} window={window.window_id} "
            f"action={action} source=cached_signature "
            f"source_window={window.features.get('signature_source_window_id')}"
        )
        return {
            "type": "detection_verdict",
            "session_id": state.session_id,
            "window_id": window.window_id,
            "action": action,
            "suspicious": bool(prediction.get("suspicious", False)),
            "reason": str(prediction.get("suspicious_reason") or "signature_cache"),
            "message": self.config["enforcement"]["message"],
            "prediction": prediction,
            "trigger_reason": window.trigger_reason,
            "l0_features": window.features,
            "signature_cache_hit": True,
            "signature_source_window_id": window.features.get("signature_source_window_id"),
        }

    def _apply_online_verdict_policy(self, state: SessionState, verdict: dict[str, Any]) -> None:
        prediction = dict(verdict["prediction"])
        score = float(prediction.get("mining_probability_mean", prediction.get("mining_probability_max", 0.0)))
        decision = state.policy.update(score)
        prediction.update(
            {
                "decision": decision.details,
                "suspicious": decision.suspicious,
                "suspicious_reason": decision.reason,
            }
        )

        verdict["prediction"] = prediction
        verdict["suspicious"] = decision.suspicious
        verdict["reason"] = decision.reason
        verdict["action"] = "terminate" if decision.suspicious else "log"
        if verdict["action"] == "terminate":
            verdict["message"] = mining_termination_message(prediction)
        log(
            f"online policy session={short_session(state.session_id)} window={verdict.get('window_id')} "
            f"policy={decision.details.get('policy')} action={verdict['action']} "
            f"reason={decision.reason}"
        )

    def _render_window_text(self, session_dir: Path, window: L0Window) -> str:
        rendered = render_workload_sass(
            session_dir,
            window.launches,
            short_kernel_threshold=int(self.config["kernel_analysis"]["short_kernel_threshold"]),
            content_token_budget=self.l0_config.window.content_token_budget,
        )
        token_cost = sass_token_count(rendered["text"])
        if token_cost > self.l0_config.window.content_token_budget:
            raise RuntimeError(
                f"L0 window exceeded content budget: {token_cost} > {self.l0_config.window.content_token_budget}"
            )
        window.features["pre_clip_token_cost"] = int(rendered.get("pre_clip_token_cost", token_cost))
        window.features["rendered_token_cost"] = token_cost
        window.features["front_clipped"] = bool(rendered.get("front_clipped", False))
        window.features["clipped_token_count"] = int(rendered.get("clipped_token_count", 0))
        return str(rendered["text"])

    def _artifact_for_launch(self, state: SessionState, launch: dict[str, Any]) -> KernelArtifact | None:
        try:
            code_id = int(launch.get("code_id"))
        except (TypeError, ValueError):
            return None
        kernel_name = str(launch.get("kernel_name") or "")
        direct = state.artifacts.get((code_id, kernel_name))
        if direct is not None:
            return direct
        candidates = {
            artifact.kernel_name: artifact for artifact in state.artifacts_by_code.get(code_id, [])
        }
        fallback = select_fallback_function(kernel_name, {name: [] for name in candidates})
        if fallback is None:
            return None
        return candidates.get(fallback[0])

    def _normalize_launch(self, row: dict[str, Any], session_id: str) -> dict[str, Any]:
        return {
            "sequence": row.get("sequence"),
            "timestamp_ns": row.get("timestamp_ns"),
            "pid": row.get("pid"),
            "tid": row.get("tid"),
            "code_id": row.get("code_id"),
            "kernel_name": row.get("kernel_name"),
            "grid_dim": row.get("grid_dim"),
            "block_dim": row.get("block_dim"),
            "shared_mem_bytes": row.get("shared_mem_bytes"),
            "stream": row.get("stream"),
            "device_pci_bus_id": row.get("device_pci_bus_id"),
            "session_id": session_id,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ONLINE_CONFIG)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_online_config(args.config, args.model)
    processor = OnlineProcessor(config)
    processor.serve_forever()
    return 0


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "unknown"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _window_index(window_id: str) -> int | None:
    prefix = str(window_id or "").split("_", 1)[0]
    if not prefix.startswith("w"):
        return None
    return _int_or_none(prefix[1:])


def _online_signature_cache_key(window: L0Window) -> str:
    signature_key = str(window.features.get("composition_signature_key") or "")
    if not signature_key:
        return ""
    return json.dumps(
        {"group_key": window.group_key, "composition_signature_key": signature_key},
        sort_keys=True,
        separators=(",", ":"),
    )


def launch_for_artifact(launch: dict[str, Any], artifact: KernelArtifact) -> dict[str, Any]:
    return launches_for_artifact(launch, artifact)[0]


def launches_for_artifact(launch: dict[str, Any], artifact: KernelArtifact) -> list[dict[str, Any]]:
    segment_artifacts = artifact.segments or [
        {
            "l0_kernel_id": artifact.kernel_id,
            "l0_token_cost": artifact.token_cost,
            "l0_bitwise_integer_ratio": artifact.bitwise_integer_ratio,
            "l0_rendered_instruction_count": artifact.rendered_instruction_count,
            "l0_render_mode": artifact.render_mode,
        }
    ]
    ready_launches: list[dict[str, Any]] = []
    for segment in segment_artifacts:
        ready_launch = {
            **launch,
            "kernel_name": artifact.kernel_name,
            **segment,
        }
        captured_kernel_name = launch.get("kernel_name")
        if captured_kernel_name != artifact.kernel_name:
            ready_launch["captured_kernel_name"] = captured_kernel_name
        ready_launches.append(ready_launch)
    return ready_launches


def _read_nonempty(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mining_termination_message(prediction: dict[str, Any]) -> str:
    decision = prediction.get("decision") or {}
    max_probability = _first_float(
        decision,
        "max_mining_probability",
    )
    if not max_probability:
        max_probability = _first_float(prediction, "mining_probability_max")
    mean_probability = _first_float(
        decision,
        "mean_mining_probability",
    )
    if not mean_probability:
        mean_probability = _first_float(prediction, "mining_probability_mean")
    return (
        f"{ANSI_BOLD_CYAN}[GPU-Sentry]{ANSI_RESET} "
        f"{ANSI_BOLD_RED}Cryptomining detected:{ANSI_RESET} "
        f"Max: {ANSI_YELLOW}{max_probability:.2f}{ANSI_RESET}, "
        f"Mean: {ANSI_YELLOW}{mean_probability:.2f}{ANSI_RESET}. "
        f"{ANSI_BOLD_RED}Terminating process.{ANSI_RESET}"
    )


def _first_float(values: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            return float(values[key])
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{timestamp}] [online] {message}", file=sys.stderr, flush=True)


def short_session(session_id: str) -> str:
    return session_id[:12] if session_id else "<none>"


def should_log_launch_batch(batch_index: int) -> bool:
    return batch_index <= 3 or batch_index % 100 == 0


if __name__ == "__main__":
    raise SystemExit(main())
