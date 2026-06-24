"""
P0: minimal text-only local Trinity (no C2C)
=================================================
Goal: get the Coordinator, the 3 roles (Thinker/Worker/Verifier), and loop control running
      on just a "shared text transcript" first. C2C (latent fusion) is added later in P1/P2.

Design mapping:
  - star type: the Worker (=Receiver) is the central integration point. Thinker/Verifier contribute via the transcript.
  - error decorrelation: a different model lineage per role (Thinker=GLM / Worker=Qwen3-Coder / Verifier=DeepSeek-R1).
  - continuity: depends on no model's internal state; context lives in the external text (transcript) (the heart of Trinity).

Local serving example (stand up each model on its own port, OpenAI-compatible):
  vllm serve <glm-path>          --port 8001 --served-model-name glm-4
  vllm serve <qwen3-coder-path>  --port 8002 --served-model-name qwen3-coder
  vllm serve <deepseek-r1-path>  --port 8003 --served-model-name deepseek-r1-distill

Depends: pip install openai>=1.0
Run: python -m trinity.p0
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from trinity.config import CONFIG
from trinity.events import (
    DECISION, ERROR, FINAL, RUN_START, TURN_END, TURN_START, VERDICT,
    EventCallback, EventEmitter,
)


# ============================================================
# 1. Model pool (local OpenAI-compatible endpoints)
# ============================================================
@dataclass
class LocalModel:
    name: str
    base_url: str
    model_id: str
    api_key: str = "EMPTY"
    temperature: float = 0.6
    max_tokens: int = 4096
    _client: object = field(default=None, init=False, repr=False)

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI          # lazy import: scorer/tests need no SDK
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def chat(self, system: str, user: str) -> str:
        resp = self._ensure_client().chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return strip_think((resp.choices[0].message.content or "").strip())


def _build_pool() -> dict[str, "LocalModel"]:
    """Build role -> model from the models section of config.yml."""
    pool = {}
    for role, m in CONFIG["models"].items():
        pool[role] = LocalModel(m.get("name", role), m["base_url"], m["model_id"],
                                api_key=m.get("api_key", "EMPTY"),
                                temperature=m.get("temperature", 0.6),
                                max_tokens=m.get("max_tokens", 4096))
    return pool


POOL: dict[str, "LocalModel"] = _build_pool()


# ============================================================
# 2. Role definitions and system prompts
# ============================================================
class Role(str, Enum):
    THINKER = "thinker"
    WORKER = "worker"
    VERIFIER = "verifier"


def _prompt_for(role: "Role") -> str:
    """Prefer models.<role>.system_prompt if present, else prompts.<role>."""
    return CONFIG["models"].get(role.value, {}).get("system_prompt") or CONFIG["prompts"][role.value]


SYS: dict[Role, str] = {r: _prompt_for(r) for r in Role}


# ============================================================
# 3. State (shared external transcript)
# ============================================================
@dataclass
class Turn:
    role: Role
    model: str
    output: str


@dataclass
class State:
    query: str
    turns: list[Turn] = field(default_factory=list)
    artifact: Optional[str] = None     # the latest Worker artifact (the central product in the star type)

    def latest_critique(self) -> str:
        for t in reversed(self.turns):
            if t.role == Role.VERIFIER:
                return t.output
        return ""

    def transcript(self, max_chars: int = 12000) -> str:
        """Shared external transcript. If it grows too long, compress to QUERY + the last few turns."""
        head = f"[QUERY]\n{self.query}"
        blocks = [f"[{t.role.value.upper()} #{i} ({t.model})]\n{t.output}"
                  for i, t in enumerate(self.turns, 1)]
        full = "\n\n".join([head] + blocks)
        if len(full) <= max_chars:
            return full
        kept: list[str] = []
        budget = max_chars - len(head) - 64
        for b in reversed(blocks):
            if budget - len(b) < 0:
                break
            kept.insert(0, b)
            budget -= len(b)
        return "\n\n".join([head, "...(omitted)...", *kept])


# ============================================================
# 4. Parsing utilities
# ============================================================
class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


def parse_verdict(text: str) -> Verdict:
    """Use the last VERDICT line (so quotes in the body don't mislead it). Default to the safe side (REVISE)."""
    matches = re.findall(r"(?im)^\s*VERDICT:\s*(ACCEPT|REVISE)\s*$", text)
    if not matches:
        matches = re.findall(r"(?i)VERDICT:\s*(ACCEPT|REVISE)", text)
    return Verdict(matches[-1].upper()) if matches else Verdict.REVISE


def strip_think(text: str) -> str:
    """Strip <think>/<thinking> from reasoning models (DeepSeek/Qwen) (including an unclosed tag to the end)."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.S | re.I)
    return text.strip()


# ============================================================
# 5. Coordinator (P0 is hard-coded; later replaced by sep-CMA-ES + a small SLM head)
# ============================================================
@dataclass
class Action:
    role: Role
    model_key: str
    meta: dict = field(default_factory=dict)   # future: logits/scores etc. (for sep-CMA-ES)


class Coordinator:
    """Extension point: replace this with a learned coordinator (Qwen3-0.6B + head, sep-CMA-ES).
    In the learned version, decide() is designed to receive "compressed features rather than the raw transcript"."""
    def decide(self, state: State) -> Optional[Action]:
        raise NotImplementedError


class ScriptedCoordinator(Coordinator):
    """For P0: the fixed flow Thinker -> Worker -> Verifier -> (if REVISE, Worker -> Verifier ...)."""
    def decide(self, state: State) -> Optional[Action]:
        if not state.turns:
            return Action(Role.THINKER, "thinker")
        last = state.turns[-1].role
        if last == Role.THINKER:
            return Action(Role.WORKER, "worker")
        if last == Role.WORKER:
            return Action(Role.VERIFIER, "verifier")
        if last == Role.VERIFIER:
            return Action(Role.WORKER, "worker")   # continue on REVISE (ACCEPT termination is decided by the loop)
        return None


# ============================================================
# 6. Per-role user-prompt construction (explicitly inject artifact/critique for Worker/Verifier)
# ============================================================
def build_user_prompt(role: Role, state: State) -> str:
    if role == Role.THINKER:
        return f"{state.transcript()}\n\n[TASK] Concisely give a plan, decomposition, and key points for the query above. Do not write code."
    if role == Role.WORKER:
        critique = state.latest_critique()
        extra = f"\n\n[Latest feedback (incorporate if any)]\n{critique}" if critique else ""
        return (f"{state.transcript()}{extra}\n\n"
                f"[TASK] Building on the plan/critique, produce the final solution (complete code/derivation). If a prior artifact exists, improve on it as the base.")
    if role == Role.VERIFIER:
        art = state.artifact or "(no artifact yet)"
        return (f"[QUERY]\n{state.query}\n\n[Under review = latest Worker artifact]\n{art}\n\n"
                f"[TASK] Check whether this artifact correctly and completely satisfies the query, and "
                f"put 'VERDICT: ACCEPT' or 'VERDICT: REVISE' on the last line. If REVISE, list the fixes.")
    raise ValueError(role)


# ============================================================
# 7. Orchestration loop
# ============================================================
@dataclass
class Config:
    max_turns: int = CONFIG["orchestration"]["max_turns"]
    verbose: bool = CONFIG["orchestration"]["verbose"]


def run(query: str, coordinator: Coordinator, pool: dict[str, LocalModel],
        cfg: Config = Config(), *,
        on_event: Optional[EventCallback] = None,
        run_id: Optional[str] = None) -> dict:
    """Run the orchestration loop.

    ``on_event`` (optional) receives structured trace events (see ``trinity.events``) for
    every Coordinator decision and role turn — this is what the gateway / debug UI consume.
    It is purely observational: with ``on_event=None`` behavior is byte-for-byte unchanged.
    ``pool`` only needs the ``ChatModel`` interface (``name`` + ``chat``); ``LocalModel`` and
    ``trinity.mocks.MockModel`` both satisfy it.
    """
    em = EventEmitter(on_event, run_id)
    state = State(query=query)
    final: Optional[str] = None
    error: Optional[str] = None

    em.emit(RUN_START, query=query, max_turns=cfg.max_turns, verbose=cfg.verbose)

    try:
        for step in range(cfg.max_turns):
            step_no = step + 1
            action = coordinator.decide(state)
            if action is None:
                em.emit(DECISION, step=step_no, role=None, model_key=None, meta={},
                        state_turns=len(state.turns), artifact_chars=len(state.artifact or ""),
                        reason="coordinator_stop")
                break
            em.emit(DECISION, step=step_no, role=action.role.value, model_key=action.model_key,
                    meta=action.meta, state_turns=len(state.turns),
                    artifact_chars=len(state.artifact or ""), reason="action")

            model = pool[action.model_key]
            user = build_user_prompt(action.role, state)
            system = SYS[action.role]
            em.emit(TURN_START, step=step_no, role=action.role.value, model_key=action.model_key,
                    model_name=model.name, system=system, user=user)

            t0 = time.perf_counter()
            try:
                out = model.chat(system, user)
            except Exception as e:                   # a call failure aborts (P0 keeps it simple)
                error = f"{model.name} call failed: {e}"
                em.emit(ERROR, step=step_no, role=action.role.value, model_key=action.model_key,
                        model_name=model.name, message=error)
                break
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            if not out:                              # don't treat an empty response as an artifact
                error = f"{model.name} returned empty output"
                em.emit(ERROR, step=step_no, role=action.role.value, model_key=action.model_key,
                        model_name=model.name, message=error)
                break

            state.turns.append(Turn(action.role, model.name, out))
            if action.role == Role.WORKER:
                state.artifact = out
            em.emit(TURN_END, step=step_no, role=action.role.value, model_key=action.model_key,
                    model_name=model.name, output=out, output_chars=len(out),
                    duration_ms=duration_ms, artifact_chars=len(state.artifact or ""),
                    state_turns=len(state.turns))
            if cfg.verbose:
                print(f"\n=== turn {step_no}: {action.role.value} ({model.name}) ===\n{out}")

            if action.role == Role.VERIFIER:
                verdict = parse_verdict(out)
                accepted = bool(state.artifact) and verdict == Verdict.ACCEPT
                em.emit(VERDICT, step=step_no, verdict=verdict.value, accepted=accepted,
                        artifact_chars=len(state.artifact or ""))
                if accepted:
                    final = state.artifact
                    break
    except Exception as e:
        # Unexpected orchestration error (e.g. coordinator/prompt bug). Preserve the original
        # propagation behavior, but always terminate the event stream with error + final first.
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


# ============================================================
# 8. Evaluation harness template (the key piece that makes P0 the training/comparison baseline)
# ============================================================
@dataclass
class Task:
    query: str
    scorer: Callable[[str], bool]    # artifact text -> pass/fail (external scoring: unit tests / numeric match, etc.)


def evaluate(tasks: list[Task], coordinator: Coordinator, pool: dict[str, LocalModel] = POOL) -> float:
    """Pass rate over a fixed task set. Used to compare policies (coordinators) = the concrete reward()."""
    ok = 0
    for task in tasks:
        res = run(task.query, coordinator, pool, Config(verbose=False))
        art = res["final"] or ""
        ok += int(bool(art) and task.scorer(art))
    return ok / max(len(tasks), 1)


def reward(query: str, result: dict) -> float:
    """Terminal binary reward (0/1). The objective once sep-CMA-ES is introduced. For now, a placeholder based on whether it was accepted."""
    return 1.0 if result.get("accepted") else 0.0


# ============================================================
if __name__ == "__main__":
    q = "Write a Python function that merges two sorted lists. Also state its time complexity."
    result = run(q, ScriptedCoordinator(), POOL)
    print("\n==================== RESULT ====================")
    print("ACCEPTED:", result["accepted"], "| ERROR:", result["error"])
    print("\n--- FINAL ARTIFACT ---\n", result["final"])
