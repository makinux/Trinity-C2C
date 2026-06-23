"""
P0 evaluation harness (real scorer) + learned-coordinator integration
=======================================================
- scorer: extract code from the output -> run unit tests in an isolated subprocess -> pass/fail (deterministic, external scoring)
- comparison: Trinity(P0 scripted) / Trinity(learned) / Worker x5 self-refine / each model 1-shot
  Note: the Trinity variants and Worker x5 are a compute-matched fair comparison. 1-shot is a reference value.

Run:
  python -m trinity.eval --selftest                              # no model. wiring check
  python -m trinity.eval --trials 3                              # baseline of 4 policies
  python -m trinity.eval --trials 3 --learned coordinator_theta.npy   # also add the learned column (qwen3 features)
  python -m trinity.eval --learned theta.npy --featurizer mock   # load the learned version with mock features

[!] Security: this runs generated code. It isolates with -I/-S, timeout, a temp cwd, and a minimal env, but is "not a true sandbox".
"""
from __future__ import annotations

import os
import re
import sys
import secrets
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional

from trinity.p0 import (
    POOL, Role, SYS, State, Config, Coordinator, ScriptedCoordinator,
    run, build_user_prompt,
)
from trinity.config import get


# ============================================================
# 1. Coding tasks (a minimal bench)
# ============================================================
@dataclass
class CodingTask:
    name: str
    query: str
    test: str


TASKS: list[CodingTask] = [
    CodingTask(
        "merge_sorted",
        "Write a Python function `merge(a, b)` that takes two sorted integer lists and returns a new list merged in sorted order.",
        textwrap.dedent("""
            assert merge([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
            assert merge([], [1]) == [1]
            assert merge([1,2,2],[2,3]) == [1,2,2,2,3]
        """),
    ),
    CodingTask(
        "two_sum",
        "Write a function `two_sum(nums, target)` that takes an integer list `nums` and a `target`, and returns the index pair of the two elements whose sum equals `target`.",
        textwrap.dedent("""
            assert sorted(two_sum([2,7,11,15],9)) == [0,1]
            assert sorted(two_sum([3,2,4],6)) == [1,2]
        """),
    ),
    CodingTask(
        "is_palindrome",
        "Write a function `is_palindrome(s)` that decides whether a string is a palindrome, considering only alphanumeric characters and ignoring case.",
        textwrap.dedent("""
            assert is_palindrome("A man, a plan, a canal: Panama") is True
            assert is_palindrome("race a car") is False
        """),
    ),
    CodingTask(
        "fib",
        "Write a function `fib(n)` that returns the n-th Fibonacci number (F(0)=0, F(1)=1).",
        textwrap.dedent("""
            assert fib(0) == 0 and fib(1) == 1
            assert fib(10) == 55
        """),
    ),
]


# ============================================================
# 2. Code extraction + execution scorer
# ============================================================
def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if blocks:
        defed = [b for b in blocks if re.search(r"^\s*(def|class)\s", b, re.M)]
        return max(defed or blocks, key=len).strip()
    return text.strip()


def run_unit_test(code: str, test: str, timeout: float = 10.0) -> tuple[bool, str]:
    if not code.strip():
        return False, "empty code"
    token = "PASS_" + secrets.token_hex(8)
    program = f"{code}\n\n{test}\nprint({token!r})\n"
    try:
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", program],
                capture_output=True, text=True, timeout=timeout,
                cwd=td, env={"PATH": os.environ.get("PATH", "")},
            )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    ok = proc.returncode == 0 and token in proc.stdout
    err = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
    return ok, err


def make_scorer(task: CodingTask, timeout: float | None = None) -> Callable[[str], bool]:
    to = get("eval", "timeout", 10.0) if timeout is None else timeout      # config.eval.timeout
    def scorer(artifact: str) -> bool:
        ok, _ = run_unit_test(extract_code(artifact), task.test, to)
        return ok
    return scorer


# ============================================================
# 3. Policies (generator factory). pool is injectable (swap in a mock for tests)
# ============================================================
def gen_with_coordinator(coordinator: Coordinator, pool: dict = POOL) -> Callable[[CodingTask], str]:
    """A policy that runs Trinity with any Coordinator (Scripted / Learned)."""
    def gen(task: CodingTask) -> str:
        return run(task.query, coordinator, pool, Config(verbose=False))["final"] or ""
    return gen


def gen_single(model_key: str, pool: dict = POOL) -> Callable[[CodingTask], str]:
    def gen(task: CodingTask) -> str:
        try:
            return pool[model_key].chat(SYS[Role.WORKER], build_user_prompt(Role.WORKER, State(query=task.query)))
        except Exception as e:
            return f"(error: {e})"
    return gen


def gen_worker_self_refine(turns: int = 5, pool: dict = POOL) -> Callable[[CodingTask], str]:
    """Worker xN self-refinement (a fair baseline compute-matched to Trinity; it does not peek at the tests)."""
    def gen(task: CodingTask) -> str:
        try:
            ans = pool["worker"].chat(SYS[Role.WORKER], f"[QUERY]\n{task.query}\n\n[TASK] Write the complete solution (code).")
            for _ in range(max(turns - 1, 0)):
                ans = pool["worker"].chat(
                    SYS[Role.WORKER],
                    f"[QUERY]\n{task.query}\n\n[Current solution]\n{ans}\n\n"
                    f"[TASK] Critically check your solution for errors/omissions and, if needed, output an improved complete solution. If it is fine, restate the same solution.",
                )
            return ans
        except Exception as e:
            return f"(error: {e})"
    return gen


# ============================================================
# 4. Load a learned coordinator (theta -> LearnedCoordinator)
# ============================================================
def build_learned_coordinator(theta_path: str, featurizer_kind: str = "qwen3") -> Coordinator:
    import numpy as np
    from trinity.coordinator import (
        MockFeaturizer, Qwen3HiddenStateFeaturizer, LinearHead, LearnedCoordinator,
    )
    theta = np.load(theta_path)
    if featurizer_kind == "mock":
        dim = (len(theta) - 3) // 3                 # back out from n_params = dim*3 + 3
        feat = MockFeaturizer(dim)
    else:
        feat = Qwen3HiddenStateFeaturizer()         # GPU + transformers
    head = LinearHead(dim=feat.dim)
    assert len(theta) == head.n_params, f"theta size {len(theta)} != head {head.n_params}"
    return LearnedCoordinator(feat, head, theta)


# ============================================================
# 5. Run the bench (4 policies + an optional extra policy = the learned version)
# ============================================================
def bench(tasks: list[CodingTask] = TASKS, trials: int = 1, pool: dict = POOL,
          extra_policies: Optional[dict[str, Callable[[CodingTask], str]]] = None) -> dict[str, float]:
    policies: dict[str, Callable[[CodingTask], str]] = {
        "Trinity(P0)":       gen_with_coordinator(ScriptedCoordinator(), pool),  # up to 5 calls
        "Worker x5 selfref": gen_worker_self_refine(5, pool),                    # 5 calls (fair)
        "Worker 1-shot":     gen_single("worker", pool),                         # reference
        "Thinker 1-shot":    gen_single("thinker", pool),                        # reference
        "Verifier 1-shot":   gen_single("verifier", pool),                       # reference
    }
    if extra_policies:
        policies.update(extra_policies)             # e.g. {"Trinity(learned)": gen_with_coordinator(learned, pool)}

    print(f"\n[pass@1 over {trials} seed(s)]  Note: Trinity variants / Worker x5 are compute-matched. 1-shot is a reference.")
    print(f"{'policy':<19}" + "".join(f"{t.name:<14}" for t in tasks) + "mean")
    print("-" * (19 + 14 * len(tasks) + 6))
    results: dict[str, float] = {}
    for name, gen in policies.items():
        cells, tot = [], 0
        for t in tasks:
            p = sum(int(make_scorer(t)(gen(t))) for _ in range(trials))
            cells.append(f"{p}/{trials}")
            tot += p
        rate = tot / (len(tasks) * trials)
        results[name] = rate
        print(f"{name:<19}" + "".join(f"{c:<14}" for c in cells) + f"{rate:.0%}")
    return results


# ============================================================
# 6. Self-test (no model needed)
# ============================================================
class MockModel:
    def __init__(self, name: str, fn: Callable[[str, str], str]):
        self.name = name
        self._fn = fn

    def chat(self, system: str, user: str) -> str:
        return self._fn(system, user)


def _mock_pool() -> dict:
    def worker_fn(s, u):
        return ("```python\nimport heapq\ndef merge(a,b):\n    return list(heapq.merge(a,b))\n```"
                if "merge" in u else "```python\ndef solve():\n    return None\n```")
    return {
        "thinker":  MockModel("t", lambda s, u: "plan"),
        "worker":   MockModel("w", worker_fn),
        "verifier": MockModel("v", lambda s, u: "VERDICT: ACCEPT"),
    }


def selftest() -> None:
    good = "```python\nimport heapq\ndef merge(a,b):\n    return list(heapq.merge(a,b))\n```"
    bad = "```python\ndef merge(a,b):\n    return a + b  # not merged\n```"
    s = make_scorer(TASKS[0])
    assert s(good) is True and s(bad) is False
    print("[selftest] scorer OK : correct=PASS / buggy=fail")

    spoof = "```python\ndef merge(a,b):\n    print('PASS_anything'); return a\n```"
    assert s(spoof) is False
    print("[selftest] anti-spoof OK : fake marker rejected")

    mock = _mock_pool()
    res = run(TASKS[0].query, ScriptedCoordinator(), mock, Config(verbose=False))
    assert res["accepted"] is True and make_scorer(TASKS[0])(res["final"]) is True
    print("[selftest] pipeline OK : orchestrate->extract->exec->score")

    # the learned version can be integrated into the bench (mock features + mock model)
    import numpy as np
    from trinity.coordinator import MockFeaturizer, LinearHead, LearnedCoordinator
    feat = MockFeaturizer(16)
    head = LinearHead(dim=feat.dim)
    learned = LearnedCoordinator(feat, head, np.zeros(head.n_params))
    table = bench(tasks=TASKS[:1], trials=1, pool=mock,
                  extra_policies={"Trinity(learned)": gen_with_coordinator(learned, mock)})
    assert "Trinity(learned)" in table and isinstance(table["Trinity(learned)"], float)
    print("[selftest] learned-in-bench OK : learned version integrated into the comparison table")
    print("[selftest] ALL PASSED")


# ============================================================
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        trials = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else get("eval", "trials", 1)
        tasks = TASKS
        if "--dataset" in sys.argv:                 # use HumanEval+/MBPP+
            ds = sys.argv[sys.argv.index("--dataset") + 1]
            limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 20
            from trinity.datasets import load_humaneval_plus, load_mbpp_plus
            tasks = (load_humaneval_plus(limit) if ds == "humaneval" else load_mbpp_plus(limit))
            print(f"[dataset] {ds}+ : {len(tasks)} tasks")
        extra = {}
        if "--learned" in sys.argv:
            path = sys.argv[sys.argv.index("--learned") + 1]
            kind = sys.argv[sys.argv.index("--featurizer") + 1] if "--featurizer" in sys.argv else "qwen3"
            extra["Trinity(learned)"] = gen_with_coordinator(build_learned_coordinator(path, kind))
        bench(tasks=tasks, trials=trials, extra_policies=extra)
