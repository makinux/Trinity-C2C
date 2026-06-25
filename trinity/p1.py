"""
P1: C2C orchestration — replace P0's Thinker->Worker *text* hand-off with a KV-fusion edge
==========================================================================================
Same star topology and the same event contract as :func:`trinity.p0.run`, but the
Thinker(Sharer) -> Worker(Receiver) edge goes through :meth:`trinity.c2c_edge.C2CEngine.c2c_edge`
(heterogeneous KV fusion) instead of plain text. This is the orchestration the gateway runs in
``c2c`` mode; the debug UI renders the extra ``fusion`` event it emits.

Reuses ``State`` / ``Turn`` / ``Role`` / ``Verdict`` / ``parse_verdict`` / ``build_user_prompt`` /
``ScriptedCoordinator`` / ``SYS`` / ``Config`` from :mod:`trinity.p0` verbatim — only the
per-role *execution* differs (sharer text plan / fused Worker generation / text Verifier).

Dual channel (per the design's abstract):
  - **text channel** : the Thinker's plan is a normal text turn in the transcript (so it is also
    visible to the Worker as text) -> at gate~0 the Worker is its plain self (no latent added).
  - **KV channel**   : the Thinker(SmolLM) KV of the same Worker context is fused into the
    Worker(Qwen) KV -> at gate>0 a different-lineage latent "second opinion" is injected.

REVISE = re-fuse from a fresh canonical text state (no continued generation): every Worker turn
re-encodes ``recv_text = state.transcript()`` (which now contains the prior artifact + the
Verifier critique), so the "history branching trap" is avoided structurally (docs/design.md).

The Verifier stays text-based; by default it is the in-process Receiver (Qwen) run as a plain
chat model, so the ``c2c`` profile needs no extra service.

Run (real, small cached models; CPU):  python -m trinity.p1
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from trinity.events import (
    DECISION, ERROR, FINAL, FUSION, RUN_START, TURN_END, TURN_START, VERDICT,
    EventCallback, EventEmitter,
)
from trinity.p0 import (
    Config, Role, ScriptedCoordinator, State, Turn, Verdict, Coordinator,
    SYS, build_user_prompt, parse_verdict,
)

# A text verifier is any (system, user) -> str callable.
VerifierFn = Callable[[str, str], str]


def _worker_gen_prompt(state: State) -> str:
    """The Worker's generation instruction (the live continuation appended after the fused KV).

    Mirrors the ``[TASK]`` tail of ``trinity.p0.build_user_prompt(Role.WORKER, ...)`` and surfaces
    the latest Verifier critique explicitly so a REVISE turn re-conditions on it as text too."""
    critique = state.latest_critique()
    extra = f"\n\n[Latest feedback to incorporate]\n{critique}" if critique else ""
    return (f"{extra}\n\n[TASK] Building on the plan and any feedback, produce the final, complete "
            f"solution (full code / derivation). Improve on the prior artifact if one exists.\n\n"
            f"[Solution]\n")


def run_c2c(query: str, engine, cfg: Config = Config(), *,
            coordinator: Optional[Coordinator] = None,
            verifier: Optional[VerifierFn] = None,
            gate: Optional[float] = None,
            should_stop=None,
            on_event: Optional[EventCallback] = None,
            run_id: Optional[str] = None) -> dict:
    """C2C orchestration loop. Returns the same result dict shape as :func:`trinity.p0.run`.

    ``engine`` is a :class:`trinity.c2c_edge.C2CEngine` (or a structurally-compatible mock):
    it must expose ``sharer_name`` / ``receiver_name`` / ``gate`` / ``set_gate`` /
    ``sharer_chat`` / ``receiver_chat`` / ``c2c_edge``. ``on_event=None`` => no tracing.
    ``should_stop`` (optional predicate) is checked between turns and during Worker generation so a
    disconnected streaming client releases the shared C2C engine lock promptly.
    """
    em = EventEmitter(on_event, run_id)
    coordinator = coordinator or ScriptedCoordinator()
    verifier = verifier or (lambda system, user: engine.receiver_chat(system, user))
    state = State(query=query)
    final: Optional[str] = None
    error: Optional[str] = None

    # A run-level gate is applied per Worker turn as a TEMPORARY override (c2c_edge save/restores it),
    # so it never persists onto this shared engine; gate=None keeps the engine's gates (e.g. learned).
    eff_gate = gate if gate is not None else engine.gate
    em.emit(RUN_START, query=query, max_turns=cfg.max_turns, verbose=cfg.verbose,
            mode="c2c", gate=eff_gate,
            sharer_model=engine.sharer_name, receiver_model=engine.receiver_name)

    try:
        for step in range(cfg.max_turns):
            step_no = step + 1
            if should_stop is not None and should_stop():
                error = "run cancelled"
                em.emit(ERROR, step=step_no, role=None, model_key=None, model_name=None,
                        message=error)
                break
            action = coordinator.decide(state)
            if action is None:
                em.emit(DECISION, step=step_no, role=None, model_key=None, meta={},
                        state_turns=len(state.turns), artifact_chars=len(state.artifact or ""),
                        reason="coordinator_stop")
                break
            em.emit(DECISION, step=step_no, role=action.role.value, model_key=action.model_key,
                    meta=action.meta, state_turns=len(state.turns),
                    artifact_chars=len(state.artifact or ""), reason="action")

            role = action.role
            t0 = time.perf_counter()
            try:
                if role == Role.THINKER:
                    out, model_name = _run_thinker(engine, state, em, step_no)
                elif role == Role.WORKER:
                    out, model_name = _run_worker(engine, state, em, step_no, should_stop, gate)
                elif role == Role.VERIFIER:
                    out, model_name = _run_verifier(verifier, engine, state, em, step_no)
                else:
                    raise ValueError(role)
            except Exception as e:                       # a model/edge failure aborts (like P0)
                error = f"{role.value} step failed: {e}"
                em.emit(ERROR, step=step_no, role=role.value, model_key=action.model_key,
                        model_name=None, message=error)
                break
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Validate BEFORE emitting turn_end (parity with p0.run: an empty turn is an error,
            # not a completed turn).
            if not out:
                error = f"{role.value} returned empty output"
                em.emit(ERROR, step=step_no, role=role.value, model_key=action.model_key,
                        model_name=model_name, message=error)
                break

            state.turns.append(Turn(role, model_name, out))
            if role == Role.WORKER:
                state.artifact = out
            em.emit(TURN_END, step=step_no, role=role.value, model_key=action.model_key,
                    model_name=model_name, output=out, output_chars=len(out),
                    duration_ms=duration_ms, artifact_chars=len(state.artifact or ""),
                    state_turns=len(state.turns))

            if role == Role.VERIFIER:
                verdict = parse_verdict(out)
                accepted = bool(state.artifact) and verdict == Verdict.ACCEPT
                em.emit(VERDICT, step=step_no, verdict=verdict.value, accepted=accepted,
                        artifact_chars=len(state.artifact or ""))
                if accepted:
                    final = state.artifact
                    break
    except Exception as e:
        error = f"unexpected orchestration error: {e}"
        em.emit(ERROR, step=None, role=None, model_key=None, model_name=None, message=error)
        em.emit(FINAL, accepted=False, error=error, final=state.artifact,
                final_chars=len(state.artifact or ""), turns=len(state.turns))
        raise

    result = {
        "final": final if final is not None else state.artifact,
        "accepted": final is not None,
        "error": error,
        "state": state,
    }
    em.emit(FINAL, accepted=result["accepted"], error=error, final=result["final"],
            final_chars=len(result["final"] or ""), turns=len(state.turns))
    return result


# ---------------------------------------------------------------------------
# per-role execution: each emits TURN_START (+ FUSION for the Worker) and returns (output,
# model_name). TURN_END is emitted by the loop AFTER the empty-output check (p0 parity).
# ---------------------------------------------------------------------------
def _run_thinker(engine, state: State, em: EventEmitter, step_no: int) -> tuple[str, str]:
    system, user = SYS[Role.THINKER], build_user_prompt(Role.THINKER, state)
    em.emit(TURN_START, step=step_no, role=Role.THINKER.value, model_key="thinker",
            model_name=engine.sharer_name, system=system, user=user)
    out = engine.sharer_chat(system, user)               # text plan -> transcript (text channel)
    return out, engine.sharer_name


def _run_worker(engine, state: State, em: EventEmitter, step_no: int,
                should_stop=None, gate=None) -> tuple[str, str]:
    # Both models condition on the SAME Worker context; the Thinker's latent (KV) is what gets
    # injected. recv == share so char-span alignment over identical text is near-identity.
    worker_context = state.transcript()
    task = _worker_gen_prompt(state)
    system = SYS[Role.WORKER]
    # C2C generation is a raw continuation (no chat system role), so fold the Worker system prompt
    # into what is actually generated — and report it in the trace, so system/user reflect reality.
    gen_prompt = f"{system}\n{task}"
    em.emit(TURN_START, step=step_no, role=Role.WORKER.value, model_key="worker",
            model_name=engine.receiver_name, system=system, user=worker_context + task)
    artifact, meta = engine.c2c_edge(share_text=worker_context, recv_text=worker_context,
                                     gen_prompt=gen_prompt, gate=gate, should_stop=should_stop)
    em.emit(FUSION, step=step_no, role=Role.WORKER.value, **meta)
    return artifact, engine.receiver_name


def _run_verifier(verifier: VerifierFn, engine, state: State, em: EventEmitter,
                  step_no: int) -> tuple[str, str]:
    system, user = SYS[Role.VERIFIER], build_user_prompt(Role.VERIFIER, state)
    em.emit(TURN_START, step=step_no, role=Role.VERIFIER.value, model_key="verifier",
            model_name=engine.receiver_name, system=system, user=user)
    out = verifier(system, user)
    return out, engine.receiver_name


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from trinity.c2c_edge import C2CEngine

    q = "Write a Python function that merges two sorted lists. State its time complexity."
    print("[p1] loading C2C engine (SmolLM2-135M -> Qwen2.5-0.5B, CPU) ...")
    eng = C2CEngine(max_new_tokens=128)
    eng.set_gate(0.0)              # gate=0: Worker is its plain self; the plan rides the text channel

    def show(ev: dict) -> None:
        t = ev.get("type")
        if t == FUSION:
            print(f"  [fusion] gate={ev['gate']} aligned_layers={ev['aligned_layers']} "
                  f"share_len={ev['share_len']} recv_len={ev['recv_len']} new_tokens={ev['new_tokens']}")
        elif t in (DECISION, VERDICT, FINAL):
            print(f"  [{t}] " + ", ".join(f"{k}={ev[k]}" for k in ev
                                          if k not in ("type", "run_id", "seq", "ts", "meta")))

    result = run_c2c(q, eng, Config(max_turns=5), on_event=show)
    print("\n==================== RESULT ====================")
    print("ACCEPTED:", result["accepted"], "| ERROR:", result["error"])
    print("\n--- FINAL ARTIFACT ---\n", (result["final"] or "")[:800])
