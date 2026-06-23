"""
trinity/datasets.py — HumanEval+ / MBPP+ ローダ
================================================
EvalPlus(HumanEval+/MBPP+) の問題を trinity_eval.CodingTask に変換する。
採点は「候補解 vs 参照解(canonical) の差分テスト(differential testing)」を
自己完結した assert 列に変換し、既存の run_unit_test(code, test) で実行する
（EvalPlus流の入力ベース検証：base_input + plus_input の全入力で出力一致を確認）。

依存: pip install evalplus
使い方(コードから):
    from trinity.datasets import load_humaneval_plus
    from trinity.eval import bench
    tasks = load_humaneval_plus(limit=20)     # 20問
    bench(tasks=tasks, trials=1, ...)
CLI:
    python -m trinity.datasets --selftest                      # evalplus不要。差分テスト機構の検証
    python -m trinity.datasets --source humaneval --limit 10   # 実データの読み込み確認
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
    """参照解の完全ソース。canonicalがentry_pointをトップレベル定義済みならそれ、無ければ prompt+canonical。
    （HumanEval+ の canonical_solution は通常「補完」なので prompt 前置が必要。AST で厳密判定）。"""
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
    """候補(code側でentry_pointを定義) と 参照(別名前空間) の出力一致を全入力で検証するassert列。"""
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
        query = f"次の関数を完成させ、完全な実装をPythonコードで返せ。\n\n{problem['prompt']}"
    else:  # mbpp
        query = f"次の仕様を満たす関数 `{ep}` をPythonで実装せよ。\n\n{problem['prompt']}"
    test = _diff_test(ep, _ref_src(problem), inputs, float(problem.get("atol", 0) or 0))
    name = str(problem["task_id"]).replace("/", "_")
    if capped:
        print(f"[load] {name}: inputs capped to {max_inputs} (base={n_base}, plus={n_plus})  ※coverage削減")
    return CodingTask(name, query, test)


# ============================================================
# ローダ（evalplus 必須）
# ============================================================
# max_inputs=None: 全入力で採点（EvalPlus忠実）。速度のため減らす時のみ指定→capをログ警告。
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
# セルフテスト（evalplus不要：合成problemで差分テスト機構を検証）
# ============================================================
def selftest() -> None:
    from trinity.eval import make_scorer
    prob = {
        "task_id": "Synthetic/0",
        "entry_point": "add_one",
        "prompt": "def add_one(x):\n    \"\"\"x に 1 を足す\"\"\"\n",
        "canonical_solution": "def add_one(x):\n    return x + 1\n",
        "base_input": [[0], [5]],
        "plus_input": [[-3], [100]],
        "atol": 0,
    }
    task = _problem_to_task(prob, "humaneval", max_inputs=100)
    sc = make_scorer(task)

    good = "```python\ndef add_one(x):\n    return x + 1\n```"
    bad = "```python\ndef add_one(x):\n    return x + 2\n```"
    assert sc(good) is True, "正解がPASSしない"
    assert sc(bad) is False, "バグありがfailしない"
    print("[selftest] differential test OK : 正解=PASS / バグあり=fail")

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
        print("例:", tasks[0].name if tasks else "(none)")
