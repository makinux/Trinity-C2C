"""
trinity/persist.py — canonical persistence: append-only event log + artifact CAS
=================================================================================
The orchestration in ``trinity/p0.py`` / ``trinity/p1.py`` already emits a typed trace via
:class:`trinity.events.EventEmitter`, but that trace is **callback-only** — it streams to an
SSE client or a debug UI and then evaporates when the run returns. This module turns that same
trace into the run's *canonical, replayable source of truth* on disk, following the CC + Codex
design consultation.

Two layers (the core invariant: **canonical text is state, KV latents are not**):
  - :class:`ArtifactStore` — a content-addressed store (CAS). Each artifact body (a Worker
    artifact, a Verifier critique, the final answer) is hashed and written once under its
    ``sha256``; identical bodies dedup to one blob. The id is the address.
  - :class:`RunLog` — an append-only ``events.jsonl`` (one JSON event per line). This is the
    authoritative log; nothing is ever mutated or deleted in place.

The bridge is :func:`persisting_sink`, an ``on_event`` callback you pass straight into
``p0.run(..., on_event=...)`` — no orchestration change is required. The sink moves large
bodies (``turn_end.output``, ``final.final``) into the CAS, replaces them with an
``*_artifact_id`` reference (so the log stays small), emits a companion ``artifact`` event,
and appends every event to the log.

Design notes:
  - Dependency-free (stdlib only): ``json`` / ``hashlib`` / ``os``. No torch, no network — so
    this participates in the repo's model-free ``--selftest`` discipline.
  - The sink **never mutates the caller's event dict** (it shallow-copies before trimming), so
    chaining it alongside the SSE callback is safe.
  - A broken sink must never break a run: like :class:`EventEmitter`, write failures are
    swallowed (the run is more important than its trace).

Layout (default ``runs/<run_id>/`` with a run-spanning CAS at ``runs/_cas/``):
    runs/
      _cas/<sha256>                # immutable artifact blobs (content-addressed, deduped)
      <run_id>/
        events.jsonl               # canonical append-only event log

Usage:
    from trinity.persist import open_run_sink
    sink, ctx = open_run_sink(run_id)            # ctx holds the RunLog + ArtifactStore
    result = run(query, coord, pool, on_event=sink, run_id=run_id)
    ctx.close()
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from trinity.events import ARTIFACT, FINAL, TURN_END, Event, EventCallback

# Default root for persisted runs (override with TRINITY_RUNS_DIR).
RUNS_DIR = os.getenv("TRINITY_RUNS_DIR", "runs")

# Which event field carries a large body that should live in the CAS rather than inline in the
# log. Maps event-type -> (body_field, replacement_id_field, artifact_kind).
_EXTERNALIZE: dict[str, tuple[str, str, str]] = {
    TURN_END: ("output", "output_artifact_id", "turn_output"),
    FINAL: ("final", "final_artifact_id", "final_artifact"),
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# A CAS id is exactly a sha256 hex digest. Validating before building a path stops a tampered or
# boundary-crossing log from steering a read outside the CAS dir (e.g. an absolute Windows path or
# one containing separators / "..").
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _is_valid_artifact_id(artifact_id: object) -> bool:
    return isinstance(artifact_id, str) and bool(_HEX64.match(artifact_id))


# ============================================================
# 1. Content-addressed artifact store (CAS)
# ============================================================
class ArtifactStore:
    """A tiny content-addressed blob store. ``put`` is idempotent: identical text -> one file."""

    def __init__(self, cas_dir: Optional[str] = None):
        self.cas_dir = cas_dir or os.path.join(RUNS_DIR, "_cas")
        os.makedirs(self.cas_dir, exist_ok=True)

    def _path(self, artifact_id: str) -> str:
        if not _is_valid_artifact_id(artifact_id):    # reject anything that isn't a bare sha256 hex
            raise ValueError(f"invalid artifact id {artifact_id!r} (expected a sha256 hex digest)")
        return os.path.join(self.cas_dir, artifact_id)

    def put(self, text: str, kind: Optional[str] = None) -> str:
        """Store ``text`` and return its content address (``sha256`` hex). Deduplicates."""
        artifact_id = _sha256_text(text)
        path = self._path(artifact_id)
        if not os.path.exists(path):                 # content-addressed -> write once, never rewrite
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            os.replace(tmp, path)                     # atomic publish (no torn reads)
        return artifact_id

    def has(self, artifact_id: str) -> bool:
        return _is_valid_artifact_id(artifact_id) and os.path.exists(self._path(artifact_id))

    def get(self, artifact_id: str) -> str:
        with open(self._path(artifact_id), encoding="utf-8") as f:
            return f.read()


# ============================================================
# 2. Append-only canonical event log
# ============================================================
class RunLog:
    """Append-only ``events.jsonl`` for one run. Each :meth:`write` appends one JSON line."""

    def __init__(self, run_id: str, runs_dir: Optional[str] = None):
        self.run_id = run_id
        self.run_dir = os.path.join(runs_dir or RUNS_DIR, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.path = os.path.join(self.run_dir, "events.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    def write(self, event: Event) -> None:
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._fh.flush()                              # push to the OS; survives a process crash
        # (not os.fsync'd per line: that survives a power loss too but is too slow for a hot trace —
        #  the canonical log tolerates losing the last unflushed line, never a torn earlier one).

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    # Convenience: read the whole log back (used by trinity.replay and the self-test).
    @staticmethod
    def read(run_id: str, runs_dir: Optional[str] = None) -> list[Event]:
        path = os.path.join(runs_dir or RUNS_DIR, run_id, "events.jsonl")
        events: list[Event] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events


# ============================================================
# 3. The bridge: an on_event sink that persists the trace
# ============================================================
@dataclass
class RunSink:
    """Holds the per-run log + store and exposes a ``__call__`` usable as ``on_event``."""
    log: RunLog
    store: ArtifactStore
    run_id: str
    _art_seq: int = field(default=0, init=False)   # companion-artifact counter, disjoint from run seq

    def __call__(self, event: Event) -> None:
        try:
            self._handle(event)
        except Exception:
            # A persistence failure must never break the run (mirrors EventEmitter).
            pass

    def _handle(self, event: Event) -> None:
        ext = _EXTERNALIZE.get(event.get("type", ""))
        out = event
        if ext:
            body_field, id_field, kind = ext
            body = event.get(body_field)
            if isinstance(body, str) and body:
                artifact_id = self.store.put(body, kind)
                # Shallow-copy + swap the body for its content address (never mutate the caller's dict).
                out = {k: v for k, v in event.items() if k != body_field}
                out[id_field] = artifact_id
                # Companion artifact record so the log self-describes its CAS references. It is
                # log bookkeeping, not a run event, so it carries no run ``seq`` (``seq: None``) and
                # a separate ``art_seq`` counter — this can never collide with the emitter's seq.
                self._art_seq += 1
                self.log.write({
                    "type": ARTIFACT, "run_id": self.run_id, "seq": None, "art_seq": self._art_seq,
                    "ts": event.get("ts"), "artifact_id": artifact_id, "kind": kind,
                    "content_chars": len(body), "ref_type": event.get("type"),
                    "ref_seq": event.get("seq"),
                })
        self.log.write(out)

    def close(self) -> None:
        self.log.close()


def persisting_sink(run_id: str, runs_dir: Optional[str] = None,
                    cas_dir: Optional[str] = None) -> tuple[EventCallback, RunSink]:
    """Build an ``on_event`` callback that persists the run, plus the :class:`RunSink` owning it.

    Pass the callback as ``p0.run(..., on_event=sink, run_id=run_id)`` and call ``ctx.close()``
    when the run returns. Returns ``(callback, ctx)`` — the callback *is* ``ctx`` (``RunSink`` is
    callable); both are returned for readability at the call site.
    """
    store = ArtifactStore(cas_dir or (os.path.join(runs_dir, "_cas") if runs_dir else None))
    log = RunLog(run_id, runs_dir)
    ctx = RunSink(log=log, store=store, run_id=run_id)
    return ctx, ctx


# ============================================================
# 4. Model-free self-test
# ============================================================
def _selftest() -> None:
    import shutil
    import tempfile

    from trinity.events import RUN_START
    from trinity.mocks import build_mock_pool
    from trinity.p0 import Config, ScriptedCoordinator, run

    tmp = tempfile.mkdtemp(prefix="trinity-persist-")
    try:
        run_id = "run-selftest-0001"
        sink, ctx = persisting_sink(run_id, runs_dir=tmp)
        pool = build_mock_pool(delay_s=0.0)
        result = run("merge two sorted lists", ScriptedCoordinator(), pool,
                     Config(verbose=False), on_event=sink, run_id=run_id)
        ctx.close()

        events = RunLog.read(run_id, runs_dir=tmp)
        assert events, "no events were persisted"
        assert events[0]["type"] == RUN_START, "first event is not run_start"

        # seq is monotonic AND unique within the run's own events; companion artifact events carry
        # no run seq (seq=None) and a disjoint art_seq, so they can never collide with run seqs.
        run_seqs = [e["seq"] for e in events if e["type"] != ARTIFACT]
        assert run_seqs == sorted(run_seqs), "run event seq is not monotonic (append-only broken)"
        assert len(run_seqs) == len(set(run_seqs)), "run event seq collided"
        art_events = [e for e in events if e["type"] == ARTIFACT]
        assert all(e["seq"] is None and e.get("art_seq") for e in art_events), \
            "artifact companion events must have seq=None and an art_seq"

        # Bodies were externalized: turn_end carries an output_artifact_id, not an inline output.
        turn_ends = [e for e in events if e["type"] == TURN_END]
        assert turn_ends, "no turn_end events persisted"
        for te in turn_ends:
            assert "output" not in te, "turn_end.output should have moved to the CAS"
            assert te.get("output_artifact_id"), "turn_end is missing output_artifact_id"
            assert ctx.store.has(te["output_artifact_id"]), "CAS blob missing for turn_end"

        # Companion artifact records exist and resolve back to the exact body.
        arts = [e for e in events if e["type"] == ARTIFACT]
        assert arts, "no artifact companion events"
        sample = ctx.store.get(arts[0]["artifact_id"])
        assert isinstance(sample, str) and sample, "CAS get returned empty"

        # Dedup: storing the same text twice yields one blob / one id.
        a = ctx.store.put("identical-body", "test")
        b = ctx.store.put("identical-body", "test")
        assert a == b, "CAS did not dedup identical content"

        # The accepted final is recoverable from the log (final_artifact_id -> CAS).
        finals = [e for e in events if e["type"] == FINAL]
        assert finals and result["accepted"], "run did not reach an accepted FINAL"
        final_ev = finals[-1]
        assert "final" not in final_ev and final_ev.get("final_artifact_id"), "final body not externalized"
        recovered = ctx.store.get(final_ev["final_artifact_id"])
        assert recovered == (result["final"] or ""), "recovered final != run final"

        print("[persist] selftest OK"
              f" - {len(events)} events, {len(turn_ends)} turn_end, {len(arts)} artifact records")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
