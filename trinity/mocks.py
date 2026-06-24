"""
trinity/mocks.py — offline mock backend for the gateway / debug UI
==================================================================
The real role models hit local OpenAI-compatible servers (vLLM on ``localhost:8001-8003``)
that are usually **not running during development**. To exercise the orchestration, the API
gateway, and the debug ChatUI end-to-end with no GPU and no network, this module provides a
drop-in pool of deterministic mock models.

Design notes (matching the plan / Codex review):
  - We inject at the **pool boundary** — ``p0.run()`` already takes ``pool: dict[str, ...]``,
    so nothing in ``LocalModel`` needs to change.
  - :class:`MockModel` is structurally compatible with ``LocalModel`` (``name`` + ``chat``),
    captured here as the :class:`ChatModel` protocol.
  - The mock forces one ``REVISE`` -> ``ACCEPT`` cycle so the UI shows the full loop
    (Thinker -> Worker -> Verifier[REVISE] -> Worker -> Verifier[ACCEPT]).
  - A small per-turn ``delay_s`` makes the live SSE trace visibly progressive instead of
    finishing in microseconds. The sleep runs inside the worker thread (``asyncio.to_thread``)
    so it never blocks the event loop.
  - :func:`build_mock_pool` returns a **fresh** pool every call; the per-instance call
    counters must not leak between requests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from trinity.p0 import Role


@runtime_checkable
class ChatModel(Protocol):
    """The minimal interface ``p0.run()`` needs from a model. ``LocalModel`` satisfies it."""
    name: str

    def chat(self, system: str, user: str) -> str: ...


def _first_line(text: str, limit: int = 80) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line[:limit]


@dataclass
class MockModel:
    """A deterministic stand-in for one role's LLM. No network, no GPU."""
    name: str
    role: Role
    delay_s: float = 0.4
    calls: int = field(default=0, init=False)

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.role == Role.THINKER:
            return self._thinker(user)
        if self.role == Role.WORKER:
            return self._worker(user)
        if self.role == Role.VERIFIER:
            return self._verifier(user)
        raise ValueError(self.role)

    # --- per-role deterministic outputs ------------------------------------
    def _thinker(self, user: str) -> str:
        return (
            "[MOCK Thinker plan]\n"
            "1. Restate the goal and list the explicit requirements.\n"
            "2. Decompose into: input handling, core algorithm, edge cases, complexity note.\n"
            "3. Watch-outs: empty input, duplicates, and stating the time complexity.\n"
            "(This is offline mock output -- wire real models via config.yml to replace it.)"
        )

    def _worker(self, user: str) -> str:
        revised = self.calls > 1 or "REVISE" in user.upper()
        tag = "v2 (incorporates the Verifier's fix)" if revised else "v1"
        edge = (
            "    if not a:\n        return list(b)\n    if not b:\n        return list(a)\n"
            if revised else ""
        )
        return (
            f"[MOCK Worker artifact {tag}]\n"
            "```python\n"
            "def solve(a, b):\n"
            '    """Deterministic mock artifact produced by the offline backend."""\n'
            f"{edge}"
            "    out, i, j = [], 0, 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            out.append(a[i]); i += 1\n"
            "        else:\n"
            "            out.append(b[j]); j += 1\n"
            "    out.extend(a[i:]); out.extend(b[j:])\n"
            "    return out\n"
            "```\n"
            "Time complexity: O(n + m)."
        )

    def _verifier(self, user: str) -> str:
        if self.calls == 1:
            return (
                "[MOCK Verifier review]\n"
                "The core merge loop is correct, but it does not explicitly handle the "
                "empty-input edge cases mentioned in the plan. Add early returns for empty "
                "`a` or `b` before the loop.\n"
                "VERDICT: REVISE"
            )
        return (
            "[MOCK Verifier review]\n"
            "The revised artifact now handles empty inputs and states the O(n + m) "
            "complexity. It satisfies the query.\n"
            "VERDICT: ACCEPT"
        )


def build_mock_pool(delay_s: float = 0.4) -> dict[str, ChatModel]:
    """Build a **fresh** offline pool (role -> MockModel). Keys match ``p0._build_pool()``."""
    return {
        "thinker": MockModel("Mock-Thinker (GLM)", Role.THINKER, delay_s),
        "worker": MockModel("Mock-Worker (Qwen3-Coder)", Role.WORKER, delay_s),
        "verifier": MockModel("Mock-Verifier (DeepSeek-R1)", Role.VERIFIER, delay_s),
    }


@dataclass
class MockC2CEngine:
    """Model-free stand-in for :class:`trinity.c2c_edge.C2CEngine`.

    Lets :func:`trinity.p1.run_c2c` (and the gateway's ``c2c`` mode) be exercised end-to-end with
    no torch and no model downloads — it implements exactly the engine surface ``run_c2c`` uses:
    ``sharer_name`` / ``receiver_name`` / ``gate`` / ``set_gate`` / ``sharer_chat`` /
    ``receiver_chat`` / ``c2c_edge``. ``c2c_edge`` returns the same ``(text, meta_dict)`` shape as
    the real engine so the ``fusion`` event payload is identical in structure. Like
    :class:`MockModel` it forces one REVISE -> ACCEPT cycle so the full loop is visible.
    """
    sharer_name: str = "Mock-Sharer (SmolLM)"
    receiver_name: str = "Mock-Receiver (Qwen)"
    gate: float = 0.05
    delay_s: float = 0.0
    aligned_layers: int = 24
    _verifier_calls: int = field(default=0, init=False)

    def set_gate(self, value: float) -> None:
        self.gate = float(value)

    def _sleep(self) -> None:
        if self.delay_s > 0:
            time.sleep(self.delay_s)

    def sharer_chat(self, system: str, user: str) -> str:
        self._sleep()
        return ("[MOCK Thinker plan]\n1. Restate the goal and requirements.\n"
                "2. Decompose: input handling, core loop, edge cases, complexity.\n"
                "(offline mock C2C engine -- the latent plan would ride the KV channel)")

    def receiver_chat(self, system: str, user: str) -> str:
        # Used as the default Verifier: REVISE once, then ACCEPT (shows the whole loop).
        self._sleep()
        self._verifier_calls += 1
        if self._verifier_calls == 1:
            return ("[MOCK Verifier review]\nThe core is right but empty inputs are unhandled.\n"
                    "VERDICT: REVISE")
        return ("[MOCK Verifier review]\nNow handles empty inputs and states O(n+m).\n"
                "VERDICT: ACCEPT")

    def c2c_edge(self, share_text: str, recv_text: str, gen_prompt: str,
                 gate=None, should_stop=None) -> tuple[str, dict]:
        if gate is not None:
            self.set_gate(gate)
        self._sleep()
        tag = "v2 (incorporates the Verifier's fix)" if self._verifier_calls else "v1"
        artifact = (f"[MOCK Worker artifact {tag} via C2C edge]\n```python\n"
                    "def solve(a, b):\n    return sorted(a + b)\n```\nTime complexity: O(n+m).")
        approx = max(1, len(recv_text) // 4)
        meta = {
            "gate": round(float(self.gate), 4), "aligned_layers": self.aligned_layers,
            "share_len": approx, "recv_len": approx,
            "sharer_model": self.sharer_name, "receiver_model": self.receiver_name,
            "new_tokens": 24, "fuser": "mock",
        }
        return artifact, meta


def build_mock_c2c_engine(delay_s: float = 0.0) -> MockC2CEngine:
    """A **fresh** model-free C2C engine (per-instance REVISE/ACCEPT counter must not leak)."""
    return MockC2CEngine(delay_s=delay_s)
