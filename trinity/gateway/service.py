"""
trinity.gateway.service — bridge between FastAPI and the blocking orchestration
===============================================================================
Responsibilities:
  - build a **fresh** model pool per request (mock or real) so concurrent requests never
    share a ``LocalModel`` client or a ``MockModel`` call counter (Codex concurrency risk),
  - run the blocking ``p0.run()`` inside a worker thread (``asyncio.to_thread``) so it never
    blocks the event loop,
  - bridge the orchestration's ``on_event`` callbacks (fired on that worker thread) onto an
    ``asyncio.Queue`` via ``loop.call_soon_threadsafe`` for live SSE streaming,
  - map Trinity's multi-turn run onto an OpenAI ``chat.completion`` (the final artifact
    becomes the assistant message; the trace is offered as a non-standard ``trinity`` field).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from typing import AsyncGenerator, Optional

from trinity.config import CONFIG
from trinity.events import ERROR, FINAL
from trinity.p0 import Config, ScriptedCoordinator, _build_pool, run
from trinity.p1 import run_c2c            # torch-free import; the heavy C2CEngine is loaded lazily
from trinity.mocks import build_mock_pool

MODEL_ID = "trinity-p0"

# Headers that keep proxies/browsers from buffering the SSE stream.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_SENTINEL = object()


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def env_mock_default() -> bool:
    return _env_flag("TRINITY_GATEWAY_MOCK")


def env_c2c_default() -> bool:
    return _env_flag("TRINITY_C2C")


def resolve_mock(explicit: Optional[bool]) -> bool:
    """Explicit request flag wins; otherwise fall back to the TRINITY_GATEWAY_MOCK env var."""
    return env_mock_default() if explicit is None else bool(explicit)


def resolve_mode(trinity_mock: Optional[bool], trinity_c2c: Optional[bool]) -> str:
    """Pick the backend: ``"mock"`` | ``"text"`` | ``"c2c"``.

    Explicit ``True`` request flags win (c2c checked first). An explicit ``False`` opts that
    backend out without selecting it. Otherwise the per-backend env defaults apply, and c2c
    takes precedence over mock when both are enabled. The fallback is the real text path.
    """
    if trinity_c2c is True:
        return "c2c"
    if trinity_mock is True:
        return "mock"
    c2c = env_c2c_default() if trinity_c2c is None else bool(trinity_c2c)
    mock = env_mock_default() if trinity_mock is None else bool(trinity_mock)
    if c2c:
        return "c2c"
    if mock:
        return "mock"
    return "text"


# --- C2C engine: one cached, lock-serialized instance (the OPPOSITE of the fresh-per-request
#     pools below). Loading two torch models per request would be fatal, and a single torch model
#     is not safe under concurrent forward passes, so all C2C runs share one engine and serialize
#     on _C2C_LOCK. The lock is acquired *inside* the worker thread (see run_collect/stream_events)
#     so it never blocks the event loop. ---
_c2c_engine = None
_C2C_INIT_LOCK = threading.Lock()    # guards lazy construction (load-once)
_C2C_LOCK = threading.Lock()         # serializes torch forwards across concurrent requests


def get_c2c_engine():
    """Lazily build and cache the heterogeneous C2C engine (loads torch + two models on first use)."""
    global _c2c_engine
    if _c2c_engine is None:
        with _C2C_INIT_LOCK:
            if _c2c_engine is None:
                from trinity.c2c_edge import C2CEngine     # lazy: keeps torch out of the text/mock path
                gate_env = os.getenv("TRINITY_C2C_GATE")
                mnt = int(os.getenv("TRINITY_C2C_MAX_NEW_TOKENS", "256"))
                _c2c_engine = C2CEngine(
                    init_gate=float(gate_env) if gate_env else None, max_new_tokens=mnt)
    return _c2c_engine


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_config(max_turns: Optional[int]) -> Config:
    base = CONFIG["orchestration"]["max_turns"]
    mt = int(max_turns) if max_turns and max_turns > 0 else base
    return Config(max_turns=int(_clamp(mt, 1, 20)), verbose=False)


def build_pool(mock: bool, mock_delay: float = 0.4):
    """A fresh pool every call. Mock = offline; real = LocalModel pool from config.yml."""
    if mock:
        return build_mock_pool(_clamp(mock_delay, 0.0, 5.0))
    return _build_pool()


# ---------------------------------------------------------------------------
# running the orchestration
# ---------------------------------------------------------------------------
def _make_runnable(query: str, mode: str, cfg: Config, mock_delay: float,
                   gate: Optional[float], should_stop=None):
    """Return ``runnable(on_event) -> result`` that runs the right backend in a worker thread.

    For ``c2c`` the shared engine is fetched and ``_C2C_LOCK`` is acquired **inside** the
    returned closure, so both the (first-call) model load and the lock wait happen on the
    worker thread — never on the event loop. ``should_stop`` lets a disconnected streaming client
    abort the run between turns/steps so the lock is released promptly. For mock/text a fresh pool
    is built up front (cheap) and reused by the closure.
    """
    if mode == "c2c":
        def runnable(on_event):
            engine = get_c2c_engine()          # first call loads two torch models (in this thread)
            with _C2C_LOCK:                     # serialize torch forwards across requests
                return run_c2c(query, engine, cfg, gate=gate, should_stop=should_stop,
                               on_event=on_event)
        return runnable

    pool = build_pool(mode == "mock", mock_delay)
    def runnable(on_event):
        return run(query, ScriptedCoordinator(), pool, cfg, on_event=on_event)
    return runnable


async def run_collect(query: str, *, mode: str = "text", max_turns: Optional[int] = None,
                      mock_delay: float = 0.0,
                      gate: Optional[float] = None) -> tuple[dict, list[dict]]:
    """Non-streaming: run to completion, collecting the trace into a list."""
    cfg = make_config(max_turns)
    trace: list[dict] = []
    runnable = _make_runnable(query, mode, cfg, mock_delay, gate)
    result = await asyncio.to_thread(runnable, trace.append)
    return result, trace


async def stream_events(query: str, *, mode: str = "text", max_turns: Optional[int] = None,
                        mock_delay: float = 0.4, include_prompts: bool = True,
                        gate: Optional[float] = None) -> AsyncGenerator[dict, None]:
    """Yield trace event dicts live as the orchestration runs (used by the debug UI)."""
    cfg = make_config(max_turns)
    # On client disconnect we can't force-cancel the worker thread, but a c2c run checks this flag
    # between turns/generation steps and aborts — releasing the shared C2C lock promptly.
    stop_event = threading.Event()
    runnable = _make_runnable(query, mode, cfg, mock_delay, gate, should_stop=stop_event.is_set)
    loop = asyncio.get_running_loop()
    # Unbounded on purpose: turn-level traces are tiny and the producer is bounded by
    # max_turns. A bounded queue could raise QueueFull inside the threadsafe callback,
    # silently drop the sentinel, and hang the consumer on queue.get().
    queue: asyncio.Queue = asyncio.Queue()

    def emit(ev: dict) -> None:
        if not include_prompts and ev.get("type") == "turn_start":
            ev = {k: v for k, v in ev.items() if k not in ("system", "user")}
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def worker() -> None:
        try:
            runnable(emit)
        except Exception as e:  # surface unexpected gateway-side failures as an error event
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": ERROR, "message": f"gateway run failed: {e}", "fatal": True},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            ev = await queue.get()
            if ev is _SENTINEL:
                break
            yield ev
    finally:
        # Signal cooperative cancellation (c2c checks it between turns/steps; mock/text ignore it
        # and finish quickly). The worker thread still cannot be force-cancelled, so on a normal
        # finish the task is already done; on client disconnect we do NOT block teardown awaiting
        # the model call — detach it and swallow its (already-caught) result so asyncio logs no
        # warning. The backend run unwinds in the background; the client stream stops immediately.
        stop_event.set()
        if task.done():
            if not task.cancelled():
                task.exception()
        else:
            task.add_done_callback(lambda t: t.cancelled() or t.exception())


async def stream_debug_sse(query: str, **kwargs) -> AsyncGenerator[str, None]:
    """SSE-formatted wrapper around :func:`stream_events` for ``/debug/runs/stream``."""
    async for ev in stream_events(query, **kwargs):
        yield sse_event(ev)


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------
def sse_event(ev: dict) -> str:
    etype = ev.get("type", "message")
    return f"event: {etype}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"


def sse_data(obj) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# OpenAI mapping
# ---------------------------------------------------------------------------
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "trinity-c2c",
        }],
    }


def flatten_messages(messages) -> str:
    """Collapse OpenAI ``messages`` into a single query string for the orchestration."""
    def text_of(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # multimodal content parts
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
                else:
                    parts.append(str(p))
            return "\n".join(x for x in parts if x)
        return str(content)

    msgs = list(messages or [])
    if not msgs:
        return ""
    if len(msgs) == 1:
        return text_of(msgs[0].content)
    lines = []
    for m in msgs:
        c = text_of(m.content)
        if c:
            lines.append(f"[{str(m.role).upper()}] {c}")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)  # rough; real usage needs resp.usage from each role


def _trinity_meta(result: dict, trace: Optional[list[dict]], include_trace: bool) -> dict:
    state = result.get("state")
    meta = {
        "accepted": result.get("accepted"),
        "error": result.get("error"),
        "turns": len(state.turns) if state is not None else None,
        "usage_source": "estimated",
    }
    if include_trace and trace is not None:
        meta["trace"] = trace
    return meta


def build_completion(query: str, result: dict, trace: Optional[list[dict]],
                     *, include_trace: bool) -> dict:
    """Build a non-streaming OpenAI ``chat.completion`` from a run result."""
    content = result.get("final") or ""
    pt, ct = _estimate_tokens(query), _estimate_tokens(content)
    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            # The assistant message is complete; Trinity's accept/revise is internal, so this
            # is always "stop" (never "length", which would imply truncation to the client).
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }
    # Keep the default response pure-OpenAI; only attach the non-standard trace block on request.
    if include_trace:
        resp["trinity"] = _trinity_meta(result, trace, include_trace=True)
    return resp


def _openai_chunks(content: str):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish=None) -> dict:
        return {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    yield chunk({"role": "assistant"})
    for i in range(0, len(content), 24):
        yield chunk({"content": content[i:i + 24]})
    yield chunk({}, finish="stop")


async def stream_openai(query: str, *, mode: str = "text", max_turns: Optional[int] = None,
                        mock_delay: float = 0.0,
                        gate: Optional[float] = None) -> AsyncGenerator[str, None]:
    """``stream=true`` for ``/v1/chat/completions``: run, then stream the final content."""
    result, _ = await run_collect(query, mode=mode, max_turns=max_turns,
                                  mock_delay=mock_delay, gate=gate)
    for ch in _openai_chunks(result.get("final") or ""):
        yield sse_data(ch)
    yield "data: [DONE]\n\n"
