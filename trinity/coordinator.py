"""
学習コーディネータ骨子 (1/2): 特徴量化 + 線形ヘッド + LearnedCoordinator
=====================================================================
Trinity準拠:
  - 小型SLM(Qwen3-0.6B, 凍結)が state(transcript) を符号化 → penultimateトークンの隠れ状態 h∈R^d
  - 極小の線形ヘッド θ(数千params) が h → 役割ロジット
  - star設計ではモデルは役割固定なので、ヘッドは「次の役割」を選ぶ(|A|=3)
  - 終了は Verifier の ACCEPT でループ側が判定（P0と同じ）
θ は sep-CMA-ES(trinity/train.py)で最適化する。
"""
from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Optional

from trinity.p0 import Coordinator, Action, Role, State

ROLE_ACTIONS = [Role.THINKER, Role.WORKER, Role.VERIFIER]
ROLE_TO_KEY = {Role.THINKER: "thinker", Role.WORKER: "worker", Role.VERIFIER: "verifier"}
N_ACTIONS = len(ROLE_ACTIONS)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


# ============================================================
# 特徴量化: state -> h ∈ R^d
# ============================================================
class Featurizer:
    dim: int
    def encode(self, state: State) -> np.ndarray:
        raise NotImplementedError


class MockFeaturizer(Featurizer):
    """SLM不要のダミー特徴量。transcriptのhashから決定的にd次元ベクトル。配線/学習テスト用。"""
    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, state: State) -> np.ndarray:
        digest = hashlib.sha256(state.transcript().encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return rng.standard_normal(self.dim)


class Qwen3HiddenStateFeaturizer(Featurizer):
    """本番: Qwen3-0.6B の penultimate トークン隠れ状態（設計どおり）。transformers/torch使用（重い）。"""
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "cuda", max_len: int = 4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", output_hidden_states=True
        ).to(device).eval()
        self.device = device
        self.max_len = max_len
        self.dim = self.model.config.hidden_size

    def encode(self, state: State) -> np.ndarray:
        torch = self._torch
        ids = self.tok(state.transcript(), return_tensors="pt",
                       truncation=True, max_length=self.max_len).to(self.device)
        with torch.no_grad():
            out = self.model(**ids)
        seq = out.hidden_states[-1][0]           # [seq_len, dim]
        pos = -2 if seq.shape[0] >= 2 else -1    # penultimate（短すぎる時は最終）
        return seq[pos, :].float().cpu().numpy()


# ============================================================
# 線形ヘッド θ: R^d -> 役割ロジット
# ============================================================
@dataclass
class LinearHead:
    dim: int
    n_actions: int = N_ACTIONS

    @property
    def n_params(self) -> int:
        return self.dim * self.n_actions + self.n_actions   # W + b

    def logits(self, theta: np.ndarray, h: np.ndarray) -> np.ndarray:
        k = self.dim * self.n_actions
        W = theta[:k].reshape(self.dim, self.n_actions)
        b = theta[k:]
        return h @ W + b


# ============================================================
# 学習コーディネータ（θを差し替えるだけで方策が変わる）
# ============================================================
class LearnedCoordinator(Coordinator):
    def __init__(self, featurizer: Featurizer, head: LinearHead, theta: np.ndarray,
                 greedy: bool = True, rng: Optional[np.random.Generator] = None,
                 mask_no_artifact: bool = True):
        self.f = featurizer
        self.head = head
        self.theta = np.asarray(theta, float)
        self.greedy = greedy
        self.rng = rng or np.random.default_rng(0)
        self.mask_no_artifact = mask_no_artifact

    def decide(self, state: State) -> Optional[Action]:
        logits = self.head.logits(self.theta, self.f.encode(state)).astype(float)
        if self.mask_no_artifact and state.artifact is None:
            logits[ROLE_ACTIONS.index(Role.VERIFIER)] = -1e30   # 成果物前のVerifyを禁止(軽い構造事前)
        a = int(np.argmax(logits)) if self.greedy else int(self.rng.choice(N_ACTIONS, p=softmax(logits)))
        role = ROLE_ACTIONS[a]
        return Action(role, ROLE_TO_KEY[role])
