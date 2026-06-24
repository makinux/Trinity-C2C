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
import time
import uuid
from typing import AsyncGenerator, Optional

from trinity.config import CONFIG
from trinity.events import ERROR, FINAL
from trinity.p0 import Config, ScriptedCoordinator, _build_pool, run
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
def env_mock_default() -> bool:
    return os.getenv("TRINITY_GATEWAY_MOCK", "").strip().lower() in ("1", "true", "yes", "on")


def resolve_mock(explicit: Optional[bool]) -> bool:
    """Explicit request flag wins; otherwise fall back to the TRINITY_GATEWAY_MOCK env var."""
    return env_mock_default() if explicit is None else bool(explicit)


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
async def run_collect(query: str, *, mock: bool, max_turns: Optional[int] = None,
                      mock_delay: float = 0.0) -> tuple[dict, list[dict]]:
    """Non-streaming: run to completion, collecting the trace into a list."""
    pool = build_pool(mock, mock_delay)
    cfg = make_config(max_turns)
    trace: list[dict] = []
    result = await asyncio.to_thread(
        run, query, ScriptedCoordinator(), pool, cfg, on_event=trace.append
    )
    return result, trace


async def stream_events(query: str, *, mock: bool, max_turns: Optional[int] = None,
                        mock_delay: float = 0.4,
                        include_prompts: bool = True) -> AsyncGenerator[dict, None]:
    """Yield trace event dicts live as the orchestration runs (used by the debug UI)."""
    pool = build_pool(mock, mock_delay)
    cfg = make_config(max_turns)
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
            run(query, ScriptedCoordinator(), pool, cfg, on_event=emit)
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
        # The worker thread cannot be force-cancelled. On a normal finish the task is already
        # done; on client disconnect, do NOT block teardown awaiting an uninterruptible model
        # call — detach it and swallow its (already-caught) result so asyncio logs no warning.
        # The backend run finishes in the background; the client stream stops immediately.
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


async def stream_openai(query: str, *, mock: bool, max_turns: Optional[int] = None,
                        mock_delay: float = 0.0) -> AsyncGenerator[str, None]:
    """``stream=true`` for ``/v1/chat/completions``: run, then stream the final content."""
    result, _ = await run_collect(query, mock=mock, max_turns=max_turns, mock_delay=mock_delay)
    for ch in _openai_chunks(result.get("final") or ""):
        yield sse_data(ch)
    yield "data: [DONE]\n\n"
