"""
P0 評価ハーネス（実スコアラー）＋ 学習コーディネータ統合
=======================================================
- スコアラー: 生成物からコード抽出 → 隔離サブプロセスで単体テスト実行 → 合否（決定的・外部採点）
- 比較: Trinity(P0 scripted) / Trinity(learned) / Worker×5自己改善 / 各モデル1-shot
  ※ Trinity系 と Worker×5 は計算量を揃えた公平比較。1-shotは参考値。

実行:
  python -m trinity.eval --selftest                              # モデル不要。配線検証
  python -m trinity.eval --trials 3                              # 4方策のベースライン
  python -m trinity.eval --trials 3 --learned coordinator_theta.npy   # 学習版も列に追加(qwen3特徴量)
  python -m trinity.eval --learned theta.npy --featurizer mock   # mock特徴量で学習版を載せる

⚠️ セキュリティ: 生成コードを実行する。-I/-S・timeout・一時cwd・最小envで隔離するが「真のサンドボックスではない」。
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
# 1. コーディングタスク（最小ベンチ）
# ============================================================
@dataclass
class CodingTask:
    name: str
    query: str
    test: str


TASKS: list[CodingTask] = [
    CodingTask(
        "merge_sorted",
        "2つのソート済み整数リストを受け取り、ソート済みにマージした新リストを返す関数 `merge(a, b)` を Python で書け。",
        textwrap.dedent("""
            assert merge([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
            assert merge([], [1]) == [1]
            assert merge([1,2,2],[2,3]) == [1,2,2,2,3]
        """),
    ),
    CodingTask(
        "two_sum",
        "整数リスト nums と target を受け取り、和が target になる2要素のインデックス対を返す関数 `two_sum(nums, target)` を書け。",
        textwrap.dedent("""
            assert sorted(two_sum([2,7,11,15],9)) == [0,1]
            assert sorted(two_sum([3,2,4],6)) == [1,2]
        """),
    ),
    CodingTask(
        "is_palindrome",
        "文字列を英数字のみ・大文字小文字無視で回文判定する関数 `is_palindrome(s)` を書け。",
        textwrap.dedent("""
            assert is_palindrome("A man, a plan, a canal: Panama") is True
            assert is_palindrome("race a car") is False
        """),
    ),
    CodingTask(
        "fib",
        "n番目のフィボナッチ数 (F(0)=0, F(1)=1) を返す関数 `fib(n)` を書け。",
        textwrap.dedent("""
            assert fib(0) == 0 and fib(1) == 1
            assert fib(10) == 55
        """),
    ),
]


# ============================================================
# 2. コード抽出 + 実行スコアラー
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
# 3. 方策（generator factory）。pool を注入可能（テスト用にmock差し替え）
# ============================================================
def gen_with_coordinator(coordinator: Coordinator, pool: dict = POOL) -> Callable[[CodingTask], str]:
    """任意の Coordinator（Scripted / Learned）で Trinity を回す方策。"""
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
    """Worker×N自己改善（計算量をTrinityと同等にした公平ベースライン。テストは覗かない）。"""
    def gen(task: CodingTask) -> str:
        try:
            ans = pool["worker"].chat(SYS[Role.WORKER], f"[QUERY]\n{task.query}\n\n[TASK] 完全な解（コード）を書け。")
            for _ in range(max(turns - 1, 0)):
                ans = pool["worker"].chat(
                    SYS[Role.WORKER],
                    f"[QUERY]\n{task.query}\n\n[現在の解]\n{ans}\n\n"
                    f"[TASK] 自分の解の誤り・抜けを批判的に点検し、必要なら改善した完全な解を出せ。問題なければ同じ解を再掲。",
                )
            return ans
        except Exception as e:
            return f"(error: {e})"
    return gen


# ============================================================
# 4. 学習済みコーディネータのロード（θ → LearnedCoordinator）
# ============================================================
def build_learned_coordinator(theta_path: str, featurizer_kind: str = "qwen3") -> Coordinator:
    import numpy as np
    from trinity.coordinator import (
        MockFeaturizer, Qwen3HiddenStateFeaturizer, LinearHead, LearnedCoordinator,
    )
    theta = np.load(theta_path)
    if featurizer_kind == "mock":
        dim = (len(theta) - 3) // 3                 # n_params = dim*3 + 3 から逆算
        feat = MockFeaturizer(dim)
    else:
        feat = Qwen3HiddenStateFeaturizer()         # GPU + transformers
    head = LinearHead(dim=feat.dim)
    assert len(theta) == head.n_params, f"theta size {len(theta)} != head {head.n_params}"
    return LearnedCoordinator(feat, head, theta)


# ============================================================
# 5. ベンチ実行（4方策＋任意の追加方策＝学習版）
# ============================================================
def bench(tasks: list[CodingTask] = TASKS, trials: int = 1, pool: dict = POOL,
          extra_policies: Optional[dict[str, Callable[[CodingTask], str]]] = None) -> dict[str, float]:
    policies: dict[str, Callable[[CodingTask], str]] = {
        "Trinity(P0)":       gen_with_coordinator(ScriptedCoordinator(), pool),  # 最大5呼び出し
        "Worker x5 selfref": gen_worker_self_refine(5, pool),                    # 5呼び出し(公平)
        "Worker 1-shot":     gen_single("worker", pool),                         # 参考
        "Thinker 1-shot":    gen_single("thinker", pool),                        # 参考
        "Verifier 1-shot":   gen_single("verifier", pool),                       # 参考
    }
    if extra_policies:
        policies.update(extra_policies)             # 例: {"Trinity(learned)": gen_with_coordinator(learned, pool)}

    print(f"\n[pass@1 over {trials} seed(s)]  ※ Trinity系/Worker×5は計算量同等。1-shotは参考。")
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
# 6. セルフテスト（モデル不要）
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
                if "マージ" in u else "```python\ndef solve():\n    return None\n```")
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

    # 学習版を bench に統合できること（mock特徴量＋mockモデル）
    import numpy as np
    from trinity.coordinator import MockFeaturizer, LinearHead, LearnedCoordinator
    feat = MockFeaturizer(16)
    head = LinearHead(dim=feat.dim)
    learned = LearnedCoordinator(feat, head, np.zeros(head.n_params))
    table = bench(tasks=TASKS[:1], trials=1, pool=mock,
                  extra_policies={"Trinity(learned)": gen_with_coordinator(learned, mock)})
    assert "Trinity(learned)" in table and isinstance(table["Trinity(learned)"], float)
    print("[selftest] learned-in-bench OK : 学習版を比較表に統合")
    print("[selftest] ALL PASSED")


# ============================================================
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        trials = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else get("eval", "trials", 1)
        tasks = TASKS
        if "--dataset" in sys.argv:                 # HumanEval+/MBPP+ を使う
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
