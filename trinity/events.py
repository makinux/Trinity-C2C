"""
trinity/events.py — structured workflow-trace events
====================================================
A lightweight, dependency-free event layer so the orchestration in ``trinity/p0.py`` can
emit a typed trace of what the Coordinator and each role did — without changing its return
contract or its existing ``verbose`` print path.

Events are plain dicts (JSON-serializable) so they can be streamed straight to an SSE
client or collected into a list. :class:`EventEmitter` stamps a monotonic ``seq`` and a
wall-clock ``ts`` on every event and swallows callback exceptions, so a faulty consumer
(e.g. a disconnected debug UI) can never break a run.

Event types (the ``type`` field):

==============  ===========================  =================================================
type            when                         payload fields (besides the common ones)
==============  ===========================  =================================================
``run_start``   a run began                  query, max_turns, verbose
``decision``    Coordinator chose an action  step, role, model_key, meta, state_turns,
                (or decided to stop)         artifact_chars, reason
``turn_start``  a role is about to run       step, role, model_key, model_name, system, user
``fusion``      a C2C KV edge was applied    step, role, sharer_model, receiver_model, gate,
                (Thinker -> Worker, C2C mode) aligned_layers, share_len, recv_len, new_tokens
``turn_end``    a role finished              step, role, model_key, model_name, output,
                                             output_chars, duration_ms, artifact_chars,
                                             state_turns
``verdict``     a Verifier turn was parsed   step, verdict, accepted, artifact_chars
``error``       a turn failed / was empty    step, role, model_key, model_name, message
``final``       the run is returning         accepted, error, final, final_chars, turns
``artifact``    a body was externalized to   artifact_id, kind, content_chars, ref_type, ref_seq
                the CAS (trinity.persist)    (companion record; seq=None, carries its own art_seq)
==============  ===========================  =================================================

Every event also carries the common fields: ``{type, run_id, seq, ts}`` — except the ``artifact``
companion record, which is log bookkeeping written by :mod:`trinity.persist` (``seq`` is ``None``;
ordering uses its own ``art_seq``), not an orchestration event from :class:`EventEmitter`.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

# A trace event is just a JSON-serializable dict.
Event = dict[str, Any]
EventCallback = Callable[[Event], None]

# --- event type names (prefer these constants over bare strings) ---
RUN_START = "run_start"
DECISION = "decision"
TURN_START = "turn_start"
FUSION = "fusion"          # C2C mode only: the Thinker->Worker KV-fusion edge (gate, alignment)
TURN_END = "turn_end"
VERDICT = "verdict"
ERROR = "error"
FINAL = "final"
ARTIFACT = "artifact"     # a body was externalized to the content-addressed store (see trinity.persist)


def new_run_id() -> str:
    """A short, collision-free id for a single run."""
    return f"run-{uuid.uuid4().hex[:12]}"


class EventEmitter:
    """Stamps ``seq``/``ts`` on each event and forwards it to ``callback`` (if any), safely.

    Construct one per run. ``callback=None`` makes :meth:`emit` a cheap no-op, so the
    orchestration can call ``emitter.emit(...)`` unconditionally with zero behavioral
    change when nobody is listening.
    """

    def __init__(self, callback: Optional[EventCallback] = None, run_id: Optional[str] = None):
        self.callback = callback
        self.run_id = run_id or new_run_id()
        self._seq = 0

    def emit(self, type: str, **fields: Any) -> Event:
        self._seq += 1
        event: Event = {
            "type": type,
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": time.time(),
            **fields,
        }
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:
                # A broken/disconnected consumer must never break the run.
                pass
        return event
