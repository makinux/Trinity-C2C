"""
trinity/datasets.py — HumanEval+ / MBPP+ loader
================================================
Converts EvalPlus (HumanEval+/MBPP+) problems into trinity_eval.CodingTask.
Scoring turns "differential testing of the candidate vs the reference (canonical) solution"
into a self-contained sequence of asserts, run via the existing run_unit_test(code, test)
(EvalPlus-style input-based checking: confirm output agreement over all of base_input + plus_input).

Depends: pip install evalplus
Usage (from code):
    from trinity.datasets import load_humaneval_plus
    from trinity.eval import bench
    tasks = load_humaneval_plus(limit=20)     # 20 problems
    bench(tasks=tasks, trials=1, ...)
CLI:
    python -m trinity.datasets --selftest                      # no evalplus. checks the differential-test machinery
    python -m trinity.datasets --source humaneval --limit 10   # confirm loading of real data
"""
from __future__ import annotations

import ast
import sys
from typing import Optional

from trinity.eval import CodingTask


# ============================================================
# problem(dict) -> CodingTask
# ============================================================
def _ref_src(problem: dict) -> str:
    """Full source of the reference solution. If canonical already defines entry_point at top level, use it;
    otherwise prompt+canonical. (HumanEval+'s canonical_solution is usually a "completion", so the prompt must be
    prepended. Decided strictly via AST.)"""
    cs = problem["canonical_solution"]
    ep = problem["entry_point"]
    try:
        defines = any(isinstance(n, ast.FunctionDef) and n.name == ep for n in ast.parse(cs).body)
    except SyntaxError:
        defines = False
    return cs if defines else problem.get("prompt", "") + cs


def _get_inputs(problem: dict, max_inputs: Optional[int]):
    base = problem.get("base_input") or problem.get("base_inputs") or []
    plus = problem.get("plus_input") or problem.get("plus_inputs") or []
    inputs = list(base) + list(plus)
    capped = bool(max_inputs and len(inputs) > max_inputs)
    if capped:
        inputs = inputs[:max_inputs]
    return inputs, capped, len(base), len(plus)


def _diff_test(entry_point: str, ref_src: str, inputs: list, atol: float = 0.0) -> str:
    """A sequence of asserts checking output agreement between the candidate (entry_point defined on the code side)
    and the reference (in a separate namespace) over all inputs."""
    return (
        "import copy as _c\n"
        "_ns = {}\n"
        f"exec({ref_src!r}, _ns)\n"
        f"_ref = _ns[{entry_point!r}]\n"
        f"_atol = {atol!r}\n"
        "def _eq(a, b):\n"
        "    if _atol and isinstance(a, (int, float)) and isinstance(b, (int, float)):\n"
        "        return abs(a - b) <= _atol\n"
        "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
        "        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))\n"
        "    return a == b\n"
        f"for _inp in {inputs!r}:\n"
        f"    assert _eq({entry_point}(*_c.deepcopy(_inp)), _ref(*_c.deepcopy(_inp))), _inp\n"
    )


def _problem_to_task(problem: dict, source: str, max_inputs: Optional[int]) -> CodingTask:
    ep = problem["entry_point"]
    inputs, capped, n_base, n_plus = _get_inputs(problem, max_inputs)
    if source == "humaneval":
        query = f"Complete the following function and return the full implementation as Python code.\n\n{problem['prompt']}"
    else:  # mbpp
        query = f"Implement a Python function `{ep}` that satisfies the following spec.\n\n{problem['prompt']}"
    test = _diff_test(ep, _ref_src(problem), inputs, float(problem.get("atol", 0) or 0))
    name = str(problem["task_id"]).replace("/", "_")
    if capped:
        print(f"[load] {name}: inputs capped to {max_inputs} (base={n_base}, plus={n_plus})  (reduced coverage)")
    return CodingTask(name, query, test)


# ============================================================
# Loaders (evalplus required)
# ============================================================
# max_inputs=None: score over all inputs (EvalPlus-faithful). Set it only to reduce for speed -> logs a cap warning.
def load_humaneval_plus(limit: Optional[int] = None, max_inputs: Optional[int] = None) -> list[CodingTask]:
    from evalplus.data import get_human_eval_plus
    probs = list(get_human_eval_plus().values())
    if limit:
        probs = probs[:limit]
    return [_problem_to_task(p, "humaneval", max_inputs) for p in probs]


def load_mbpp_plus(limit: Optional[int] = None, max_inputs: Optional[int] = None) -> list[CodingTask]:
    from evalplus.data import get_mbpp_plus
    probs = list(get_mbpp_plus().values())
    if limit:
        probs = probs[:limit]
    return [_problem_to_task(p, "mbpp", max_inputs) for p in probs]


# ============================================================
# Self-test (no evalplus: validate the differential-test machinery with a synthetic problem)
# ============================================================
def selftest() -> None:
    from trinity.eval import make_scorer
    prob = {
        "task_id": "Synthetic/0",
        "entry_point": "add_one",
        "prompt": "def add_one(x):\n    \"\"\"add 1 to x\"\"\"\n",
        "canonical_solution": "def add_one(x):\n    return x + 1\n",
        "base_input": [[0], [5]],
        "plus_input": [[-3], [100]],
        "atol": 0,
    }
    task = _problem_to_task(prob, "humaneval", max_inputs=100)
    sc = make_scorer(task)

    good = "```python\ndef add_one(x):\n    return x + 1\n```"
    bad = "```python\ndef add_one(x):\n    return x + 2\n```"
    assert sc(good) is True, "correct solution does not PASS"
    assert sc(bad) is False, "buggy solution does not fail"
    print("[selftest] differential test OK : correct=PASS / buggy=fail")

    assert sc("```python\n" + prob["canonical_solution"] + "```") is True
    print("[selftest] canonical passes its own oracle")

    capped_task = _problem_to_task({**prob, "plus_input": [[i] for i in range(500)]}, "humaneval", max_inputs=50)
    assert make_scorer(capped_task)(good) is True
    print("[selftest] input capping OK")
    print("[selftest] ALL PASSED")


# ============================================================
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        source = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else "humaneval"
        limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10
        loader = load_humaneval_plus if source == "humaneval" else load_mbpp_plus
        tasks = loader(limit=limit)
        print(f"loaded {len(tasks)} tasks from {source}+ (limit={limit})")
        print("e.g.:", tasks[0].name if tasks else "(none)")
