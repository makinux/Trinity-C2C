"""
P0: テキスト版ローカルTrinity（C2C無し）最小骨子
=================================================
目的: Coordinator・3役割（Thinker/Worker/Verifier）・ループ制御を、
      まず "テキストの共有トランスクリプト" だけで動かす。C2C(潜在融合)は後のP1/P2で追加する。

設計対応:
  - star型: Worker(=Receiver) が中心の統合点。Thinker/Verifier はトランスクリプト経由で寄与。
  - 誤りの非相関: 役割ごとに別系統モデル（Thinker=GLM / Worker=Qwen3-Coder / Verifier=DeepSeek-R1）。
  - 連続性: どのモデルの内部状態にも依存せず、文脈は外部テキスト(transcript)に宿す（Trinityの肝）。

ローカルサービング例（各モデルをOpenAI互換で個別ポートに立てる）:
  vllm serve <glm-path>          --port 8001 --served-model-name glm-4
  vllm serve <qwen3-coder-path>  --port 8002 --served-model-name qwen3-coder
  vllm serve <deepseek-r1-path>  --port 8003 --served-model-name deepseek-r1-distill

依存: pip install openai>=1.0
実行: python -m trinity.p0
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from trinity.config import CONFIG


# ============================================================
# 1. モデルプール（ローカル OpenAI 互換エンドポイント）
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
            from openai import OpenAI          # 遅延import: スコアラー/テストはSDK不要
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
    """config.yml の models セクションから「役割→モデル」を構築。"""
    pool = {}
    for role, m in CONFIG["models"].items():
        pool[role] = LocalModel(m.get("name", role), m["base_url"], m["model_id"],
                                api_key=m.get("api_key", "EMPTY"),
                                temperature=m.get("temperature", 0.6),
                                max_tokens=m.get("max_tokens", 4096))
    return pool


POOL: dict[str, "LocalModel"] = _build_pool()


# ============================================================
# 2. 役割定義とシステムプロンプト
# ============================================================
class Role(str, Enum):
    THINKER = "thinker"
    WORKER = "worker"
    VERIFIER = "verifier"


def _prompt_for(role: "Role") -> str:
    """models.<role>.system_prompt があれば優先、無ければ prompts.<role>。"""
    return CONFIG["models"].get(role.value, {}).get("system_prompt") or CONFIG["prompts"][role.value]


SYS: dict[Role, str] = {r: _prompt_for(r) for r in Role}


# ============================================================
# 3. 状態（共有外部トランスクリプト）
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
    artifact: Optional[str] = None     # 最新の Worker 成果物（star型の中心生成物）

    def latest_critique(self) -> str:
        for t in reversed(self.turns):
            if t.role == Role.VERIFIER:
                return t.output
        return ""

    def transcript(self, max_chars: int = 12000) -> str:
        """共有外部トランスクリプト。長くなりすぎたら QUERY + 末尾の数ターンに圧縮。"""
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
        return "\n\n".join([head, "...(中略)...", *kept])


# ============================================================
# 4. パース系ユーティリティ
# ============================================================
class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


def parse_verdict(text: str) -> Verdict:
    """最後の VERDICT 行を採用（本文中の引用に引っ張られないように）。既定は安全側(REVISE)。"""
    matches = re.findall(r"(?im)^\s*VERDICT:\s*(ACCEPT|REVISE)\s*$", text)
    if not matches:
        matches = re.findall(r"(?i)VERDICT:\s*(ACCEPT|REVISE)", text)
    return Verdict(matches[-1].upper()) if matches else Verdict.REVISE


def strip_think(text: str) -> str:
    """reasoning系(DeepSeek/Qwen)の <think>/<thinking> を除去（未閉じも末尾まで）。"""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think(?:ing)?>.*\Z", "", text, flags=re.S | re.I)
    return text.strip()


# ============================================================
# 5. Coordinator（P0は決め打ち。将来 sep-CMA-ES + 小型SLMヘッドに差し替え）
# ============================================================
@dataclass
class Action:
    role: Role
    model_key: str
    meta: dict = field(default_factory=dict)   # 将来: ロジット/スコア等(sep-CMA-ES用)


class Coordinator:
    """拡張点: ここを学習済みコーディネータ(Qwen3-0.6B + head, sep-CMA-ES)に差し替える。
    学習版では decide() に「生transcriptではなく圧縮特徴量」を渡す設計にする。"""
    def decide(self, state: State) -> Optional[Action]:
        raise NotImplementedError


class ScriptedCoordinator(Coordinator):
    """P0用: Thinker -> Worker -> Verifier -> (REVISEなら Worker -> Verifier ...) の固定フロー。"""
    def decide(self, state: State) -> Optional[Action]:
        if not state.turns:
            return Action(Role.THINKER, "thinker")
        last = state.turns[-1].role
        if last == Role.THINKER:
            return Action(Role.WORKER, "worker")
        if last == Role.WORKER:
            return Action(Role.VERIFIER, "verifier")
        if last == Role.VERIFIER:
            return Action(Role.WORKER, "worker")   # REVISE継続（ACCEPT終了はループ側で判定）
        return None


# ============================================================
# 6. 役割別ユーザープロンプト生成（Worker/Verifierには成果物・批評を明示注入）
# ============================================================
def build_user_prompt(role: Role, state: State) -> str:
    if role == Role.THINKER:
        return f"{state.transcript()}\n\n[TASK] 上記クエリへの計画・分解・要点を簡潔に。コードは書かない。"
    if role == Role.WORKER:
        critique = state.latest_critique()
        extra = f"\n\n[直近の指摘(あれば反映)]\n{critique}" if critique else ""
        return (f"{state.transcript()}{extra}\n\n"
                f"[TASK] 計画/批評を踏まえ最終解（完全なコード/導出）を作れ。既存成果物があれば土台に改善する。")
    if role == Role.VERIFIER:
        art = state.artifact or "(まだ成果物なし)"
        return (f"[QUERY]\n{state.query}\n\n[点検対象=最新のWorker成果物]\n{art}\n\n"
                f"[TASK] この成果物がクエリを正しく・完全に満たすか点検し、"
                f"最終行に 'VERDICT: ACCEPT' か 'VERDICT: REVISE'。REVISEなら修正点も。")
    raise ValueError(role)


# ============================================================
# 7. オーケストレーション・ループ
# ============================================================
@dataclass
class Config:
    max_turns: int = CONFIG["orchestration"]["max_turns"]
    verbose: bool = CONFIG["orchestration"]["verbose"]


def run(query: str, coordinator: Coordinator, pool: dict[str, LocalModel],
        cfg: Config = Config()) -> dict:
    state = State(query=query)
    final: Optional[str] = None
    error: Optional[str] = None

    for step in range(cfg.max_turns):
        action = coordinator.decide(state)
        if action is None:
            break

        model = pool[action.model_key]
        user = build_user_prompt(action.role, state)
        try:
            out = model.chat(SYS[action.role], user)
        except Exception as e:                       # 呼び出し失敗は中断（P0は単純に）
            error = f"{model.name} call failed: {e}"
            break
        if not out:                                  # 空応答は成果物にしない
            error = f"{model.name} returned empty output"
            break

        state.turns.append(Turn(action.role, model.name, out))
        if cfg.verbose:
            print(f"\n=== turn {step+1}: {action.role.value} ({model.name}) ===\n{out}")

        if action.role == Role.WORKER:
            state.artifact = out
        if (action.role == Role.VERIFIER
                and state.artifact                      # 成果物がある時のみ…
                and parse_verdict(out) == Verdict.ACCEPT):
            final = state.artifact
            break

    return {
        "final": final if final is not None else state.artifact,
        "accepted": final is not None,
        "error": error,
        "state": state,
    }


# ============================================================
# 8. 評価ハーネス雛形（P0を学習/比較のベースラインにする最重要ピース）
# ============================================================
@dataclass
class Task:
    query: str
    scorer: Callable[[str], bool]    # 成果物テキスト -> 合否（外部採点：単体テスト/数値一致 等）


def evaluate(tasks: list[Task], coordinator: Coordinator, pool: dict[str, LocalModel] = POOL) -> float:
    """固定タスク集合での合格率。方策(coordinator)の比較に使う＝reward()の実体。"""
    ok = 0
    for task in tasks:
        res = run(task.query, coordinator, pool, Config(verbose=False))
        art = res["final"] or ""
        ok += int(bool(art) and task.scorer(art))
    return ok / max(len(tasks), 1)


def reward(query: str, result: dict) -> float:
    """終端の二値報酬(0/1)。sep-CMA-ES導入時の目的関数。今はaccept有無の暫定。"""
    return 1.0 if result.get("accepted") else 0.0


# ============================================================
if __name__ == "__main__":
    q = "2つのソート済みリストをマージする関数をPythonで書け。計算量も示せ。"
    result = run(q, ScriptedCoordinator(), POOL)
    print("\n==================== RESULT ====================")
    print("ACCEPTED:", result["accepted"], "| ERROR:", result["error"])
    print("\n--- FINAL ARTIFACT ---\n", result["final"])
