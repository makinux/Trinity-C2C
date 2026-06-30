"""
trinity/replay.py — text-only replay gate + correctness metrics
===============================================================
The core invariant from the CC + Codex design consultation is:

    *All correctness-relevant information must survive a text-only replay.*

KV / C2C latents are a disposable accelerator; if a run's result cannot be reconstructed from
the **canonical text** alone (the ``events.jsonl`` log + the artifact CAS, with no KV), then
hidden latent state has leaked into the result — which is a design bug, not a feature.

This module is that gate:
  - :func:`replay_from_log` rebuilds a ``p0.State`` (query / turns / artifact) from the
    persisted log + CAS, touching **no KV at all**.
  - :func:`assert_text_only_replay` checks that an accepted run's final artifact is recoverable
    from canonical text. It is the correctness gate referenced by the plan.
  - :func:`compute_metrics` reports the observable health metrics (the KV-namespace metrics stay
    ``None`` until ``trinity.kvstore`` lands in Phase C).

**Scope / limitation (important).** This gate proves the canonical text is *sufficient and
self-consistent*: the accepted result is fully recoverable from the log + CAS, with no body that
only ever existed in volatile memory or in a KV blob. It does **not**, on its own, prove that
generation was *independent* of latent influence — if a hidden KV effect happened to be mirrored
into the logged text, replay still matches. The complete guarantee is the design's eval-time A/B:
re-run the same query through the text-only P0 path and compare against the C2C path. This module is
the cheap, always-on half of that (it fails loudly the moment a result stops being text-recoverable);
the A/B is the expensive, model-required other half.

Dependency-free (stdlib + ``trinity.persist`` + ``trinity.p0`` dataclasses) so it runs under the
repo's model-free ``--selftest`` discipline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from trinity.events import FINAL, RUN_START, TURN_END, VERDICT
from trinity.persist import ArtifactStore, RunLog
from trinity.p0 import Role, State, Turn


def _resolve_body(event: dict, body_field: str, id_field: str,
                  store: Optional[ArtifactStore]) -> str:
    """Return an event's text body, whether inline (``body_field``) or in the CAS (``id_field``)."""
    if body_field in event and isinstance(event[body_field], str):
        return event[body_field]
    art_id = event.get(id_field)
    if art_id and store is not None and store.has(art_id):
        return store.get(art_id)
    return ""        # missing/inline-less body -> unrecoverable from text (the gate will flag it)


# ============================================================
# 1. Reconstruct a State from canonical text only (no KV)
# ============================================================
def replay_from_log(run_id: str, runs_dir: Optional[str] = None,
                    store: Optional[ArtifactStore] = None) -> State:
    """Rebuild the shared external transcript (``p0.State``) from ``events.jsonl`` + CAS only."""
    events = RunLog.read(run_id, runs_dir=runs_dir)
    store = store or ArtifactStore(None if runs_dir is None else f"{runs_dir}/_cas")

    query = ""
    for e in events:
        if e["type"] == RUN_START:
            query = e.get("query", "")
            break

    state = State(query=query)
    for e in events:
        if e["type"] != TURN_END:
            continue
        output = _resolve_body(e, "output", "output_artifact_id", store)
        role = Role(e["role"])
        state.turns.append(Turn(role, e.get("model_name", ""), output))
        if role == Role.WORKER:
            state.artifact = output            # the central product in the star type
    return state


# ============================================================
# 2. The correctness gate
# ============================================================
@dataclass
class ReplayResult:
    accepted: bool
    final_recoverable: bool
    final_matches: bool
    replayed_final: str
    logged_final: str


def replay_result(run_id: str, runs_dir: Optional[str] = None,
                  store: Optional[ArtifactStore] = None) -> ReplayResult:
    """Compare the text-only replayed final against the run's logged FINAL."""
    events = RunLog.read(run_id, runs_dir=runs_dir)
    store = store or ArtifactStore(None if runs_dir is None else f"{runs_dir}/_cas")

    finals = [e for e in events if e["type"] == FINAL]
    final_ev = finals[-1] if finals else {}
    accepted = bool(final_ev.get("accepted"))
    logged_final = _resolve_body(final_ev, "final", "final_artifact_id", store)

    state = replay_from_log(run_id, runs_dir=runs_dir, store=store)
    replayed_final = state.artifact or ""

    return ReplayResult(
        accepted=accepted,
        final_recoverable=bool(replayed_final),
        final_matches=(replayed_final == logged_final),
        replayed_final=replayed_final,
        logged_final=logged_final,
    )


def assert_text_only_replay(run_id: str, runs_dir: Optional[str] = None,
                            store: Optional[ArtifactStore] = None) -> ReplayResult:
    """Raise ``AssertionError`` if an accepted run's final can't be reconstructed from text alone."""
    r = replay_result(run_id, runs_dir=runs_dir, store=store)
    if r.accepted:
        assert r.final_recoverable, (
            f"run {run_id}: accepted but final not recoverable from canonical text "
            "(hidden KV/latent state leaked into the result)")
        assert r.final_matches, (
            f"run {run_id}: text-only replay final != logged final "
            "(canonical text is not the source of truth)")
    return r


# ============================================================
# 3. Correctness / health metrics
# ============================================================
@dataclass
class Metrics:
    runs: int = 0
    accepted_runs: int = 0
    text_only_replay_success_rate: float = 1.0
    revise_total: int = 0              # all REVISE verdicts (incl. a terminal/max-turn one)
    revise_count: int = 0             # REVISEs followed by a Worker turn (the barrier denominator)
    revise_critique_included: int = 0
    revise_critique_included_rate: Optional[float] = None
    # KV-namespace metrics are populated once trinity.kvstore (Phase C) is wired; None until then.
    kv_hit_rate_by_namespace: Optional[dict] = None
    stale_kv_rejection_count: Optional[int] = None
    branch_crossing_cache_attempts: Optional[int] = None
    fuser_abi_mismatch_count: Optional[int] = None


def compute_metrics(run_ids: list[str], runs_dir: Optional[str] = None,
                    store: Optional[ArtifactStore] = None) -> Metrics:
    """Aggregate observable correctness metrics across a set of persisted runs."""
    m = Metrics()
    replay_ok = 0
    for run_id in run_ids:
        m.runs += 1
        events = RunLog.read(run_id, runs_dir=runs_dir)
        r = replay_result(run_id, runs_dir=runs_dir, store=store)
        if r.accepted:
            m.accepted_runs += 1
        # text-only replay must hold for accepted runs; vacuously true otherwise.
        replay_ok += int((not r.accepted) or (r.final_recoverable and r.final_matches))

        # Verifier text barrier: after a REVISE verdict, the next Worker prompt must carry the
        # exact critique text (latent intent cannot substitute for it).
        rt, rc, ric = _scan_revise_barrier(events, store or ArtifactStore(
            None if runs_dir is None else f"{runs_dir}/_cas"))
        m.revise_total += rt
        m.revise_count += rc
        m.revise_critique_included += ric

    m.text_only_replay_success_rate = replay_ok / max(m.runs, 1)
    if m.revise_count:
        m.revise_critique_included_rate = m.revise_critique_included / m.revise_count
    return m


def _scan_revise_barrier(events: list[dict], store: ArtifactStore) -> tuple[int, int, int]:
    """Return ``(revise_total, revise_followed_by_worker, critique_included)``.

    Event order per Verifier turn (see ``p0.run``): ``turn_end(verifier)`` -> ``verdict``. So the
    critique is the *most recent* Verifier ``turn_end`` body (externalized to the CAS); on a REVISE
    verdict we arm it and require the next Worker ``turn_start.user`` to contain it verbatim.

    The barrier invariant only applies when a Worker turn actually follows, so the *rate* denominator
    is ``revise_followed_by_worker``. ``revise_total`` additionally counts a terminal / max-turn
    REVISE that ends the run with no subsequent Worker turn (which the followed-by-worker count would
    otherwise silently drop).
    """
    revise_total = 0
    revise_followed = 0
    included = 0
    last_verifier_critique = ""
    armed: Optional[str] = None
    for e in events:
        et = e["type"]
        if et == TURN_END and e.get("role") == Role.VERIFIER.value:
            last_verifier_critique = _resolve_body(e, "output", "output_artifact_id", store)
        elif et == VERDICT and e.get("verdict") == "REVISE":
            revise_total += 1
            armed = last_verifier_critique[:120]            # a stable slice of the critique text
        elif et == "turn_start" and e.get("role") == Role.WORKER.value and armed is not None:
            revise_followed += 1
            user = e.get("user", "") or ""
            # the critique is injected verbatim by p0.build_user_prompt -> a substring match holds
            included += int(bool(armed) and armed in user)
            armed = None
    return revise_total, revise_followed, included


# ============================================================
# 4. Model-free self-test
# ============================================================
def _selftest() -> None:
    import shutil
    import tempfile

    from trinity.mocks import build_mock_pool
    from trinity.p0 import Config, ScriptedCoordinator, run
    from trinity.persist import persisting_sink

    tmp = tempfile.mkdtemp(prefix="trinity-replay-")
    try:
        run_id = "run-replay-0001"
        sink, ctx = persisting_sink(run_id, runs_dir=tmp)
        pool = build_mock_pool(delay_s=0.0)
        result = run("merge two sorted lists", ScriptedCoordinator(), pool,
                     Config(verbose=False), on_event=sink, run_id=run_id)
        ctx.close()

        # 1. Text-only replay reconstructs the same State / final.
        r = assert_text_only_replay(run_id, runs_dir=tmp, store=ctx.store)
        assert r.accepted and r.final_recoverable and r.final_matches, "replay gate failed"
        assert r.replayed_final == (result["final"] or ""), "replayed final != live run final"

        state = replay_from_log(run_id, runs_dir=tmp, store=ctx.store)
        assert state.query == "merge two sorted lists", "query not reconstructed"
        assert any(t.role == Role.WORKER for t in state.turns), "no worker turn reconstructed"

        # 2. Tamper detection: a corrupted FINAL reference must fail the gate (proves it's real).
        bad_id = "run-replay-bad"
        events = RunLog.read(run_id, runs_dir=tmp)
        sink2, ctx2 = persisting_sink(bad_id, runs_dir=tmp)
        for e in events:
            if e["type"] == FINAL:
                e = {**e, "final_artifact_id": "deadbeef" * 8}   # point at a non-existent blob
            ctx2.log.write(e)
        ctx2.close()
        tampered = replay_result(bad_id, runs_dir=tmp, store=ctx2.store)
        assert tampered.accepted and not tampered.final_matches, "tamper not detected by gate"

        # 3. Metrics: REVISE happened once in the mock loop and the critique rode the text channel.
        m = compute_metrics([run_id], runs_dir=tmp, store=ctx.store)
        assert m.runs == 1 and m.accepted_runs == 1, "metrics run/accepted wrong"
        assert m.text_only_replay_success_rate == 1.0, "replay success rate wrong"
        assert m.revise_total >= 1, "expected at least one REVISE verdict in the mock loop"
        assert m.revise_count >= 1, "expected the REVISE to be followed by a Worker turn"
        assert m.revise_critique_included_rate == 1.0, "verifier text barrier not satisfied in replay"

        print(f"[replay] selftest OK - replay matches, tamper caught, "
              f"revise_barrier={m.revise_critique_included}/{m.revise_count}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
