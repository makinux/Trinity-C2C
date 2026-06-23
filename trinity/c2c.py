"""
P1: C2C導入 — Thinker→Worker のテキスト受け渡しを KV潜在融合(cache fuser)に置換
==============================================================================
Cache-to-Cache(arXiv:2510.03215)準拠の cache fuser を、本スタックの star設計に最小導入する。
P0では Worker は Thinker の計画を「テキスト」で受けた。P1では Worker(=Receiver) の KVキャッシュに
Thinker(=Sharer) の KV を「潜在で融合」して生成を駆動する（1辺だけC2C化）。

重要な前提:
  - C2Cは各モデルの内部KVへの read/inject が必要 → Thinker/Worker は HTTP(vLLM) ではなく
    in-process(transformers) で動かす必要がある（本ファイルの InProcessLM）。Verifier はテキストのままでよい。
  - cache fuser は「学習される」射影＋ゲート。未学習(gate≈0)なら Worker単独(計画テキストなし)に縮退。
    計画テキストも併用(plan_text)すれば gate=0 で厳密に P0 相当 → 段階導入が安全。学習が進むほど計画潜在を注入。

⚠️ 異種モデル(GLM Thinker→Qwen Worker)の本質的ブロッカー（C2C論文が主に解く所。本骨子は未解決を明示）:
  (1) GQA: num_key_value_heads が異なると C2CFuser.init が hard-fail → ヘッド射影/対応付けが必要。
  (2) トークナイザ整合: 位置iのKVは送受で意味が対応しない → TokenAligner(decode→相手トークナイザでre-encode)必須。
      本骨子の fuse() は「同一トークナイザ＝位置一致」を仮定した最小実装。
  (3) KV注入生成: transformers の Cache API/attention_mask/position(RoPE) 整合が版依存で脆い(generate_from_kv参照)。

検証:
  python -m trinity.c2c --selftest    # numpyでfuserのコア(射影/層整合/ゲート融合)を検証(torch不要)

依存(本番): pip install torch transformers   ※融合コアはnumpyのみ
"""
from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# KV表現（バッチ=1を潰した numpy 表現）
# ============================================================
@dataclass
class KVShape:
    n_layers: int
    n_heads: int
    head_dim: int


@dataclass
class KVCache:
    """layers[ℓ] = (K, V), 各 [n_heads, seq, head_dim]"""
    layers: list[tuple[np.ndarray, np.ndarray]]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def copy(self) -> "KVCache":
        return KVCache([(K.copy(), V.copy()) for K, V in self.layers])


def from_hf_past(past) -> KVCache:
    """transformers の past_key_values (tuple[(K,V)], K/V=[batch,heads,seq,hd]) → KVCache(batch=1)。"""
    return KVCache([(np.asarray(K)[0], np.asarray(V)[0]) for (K, V) in past])


def to_hf_past(kv: KVCache):
    """KVCache → transformers past_key_values 形式（batch次元を復元）。torch側でtensor化して使う。"""
    return tuple((K[None, ...], V[None, ...]) for (K, V) in kv.layers)


# ============================================================
# 層整合（terminal alignment）: 受信側の各層を、上(終端)から後ろ向きに送信側へ対応付け
# ============================================================
def terminal_alignment(n_sharer: int, n_receiver: int) -> dict[int, int]:
    """receiver層 i -> sharer層 j。終端(最終層)同士を先に対応させ、後ろ向きにペアリング。"""
    m: dict[int, int] = {}
    for k in range(min(n_sharer, n_receiver)):
        m[n_receiver - 1 - k] = n_sharer - 1 - k
    return m


class TokenAligner:
    """送信/受信のトークナイザが異なる場合、位置iのKVは意味的に対応しない（Codex指摘の最大リスク）。
    C2Cは『受信トークンをdecode→送信トークナイザでre-encode』して位置対応を作る。
    異種モデル(GLM↔Qwen)では本クラスを実装し、送信KVを受信の位置系列へ並べ替えてから fuse() する。"""
    def align(self, share_kv: "KVCache", share_tokens: list[int], recv_tokens: list[int]) -> "KVCache":
        raise NotImplementedError("異種トークナイザの位置対応は要実装（同一トークナイザなら不要）")


class IdentityTokenAligner(TokenAligner):
    """同一トークナイザ（＝共有プレフィックスが位置一致）前提。何もしない。"""
    def align(self, share_kv: "KVCache", share_tokens: list[int], recv_tokens: list[int]) -> "KVCache":
        return share_kv


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ============================================================
# cache fuser（C2C本体・numpyコア）
#   学習対象: 層ごとの射影 W_k/W_v（head_dim整合）と ゲート g（Gumbel-sigmoid@train / sigmoid@infer）
#   融合:    fused = (1-g)·受信KV + g·proj(送信KV)   ※共有プレフィックス長Lの範囲で
# ============================================================
class C2CFuser:
    def __init__(self, align: dict[int, int],
                 Wk: list[Optional[np.ndarray]], Wv: list[Optional[np.ndarray]],
                 gate_logit: np.ndarray):
        self.align = align
        self.Wk = Wk
        self.Wv = Wv
        self.gate_logit = gate_logit          # [n_receiver_layers]

    @classmethod
    def init(cls, sharer: KVShape, receiver: KVShape, default_gate: float = 0.05, seed: int = 0) -> "C2CFuser":
        """恒等初期化（head_dim一致なら恒等射影、不一致ならランダム）。gateは小さく＝既定はreceiver寄り。"""
        if sharer.n_heads != receiver.n_heads:
            # GQA等でKVヘッド数が異なる場合の対応(ヘッド射影/複製)はTODO。骨子では一致前提。
            raise ValueError(f"n_heads mismatch {sharer.n_heads}!={receiver.n_heads} (head projection is TODO)")
        rng = np.random.default_rng(seed)
        align = terminal_alignment(sharer.n_layers, receiver.n_layers)
        Wk: list[Optional[np.ndarray]] = [None] * receiver.n_layers
        Wv: list[Optional[np.ndarray]] = [None] * receiver.n_layers
        for i in range(receiver.n_layers):
            if i not in align:
                continue
            if sharer.head_dim == receiver.head_dim:
                Wk[i] = np.eye(sharer.head_dim)
                Wv[i] = np.eye(sharer.head_dim)
            else:
                scale = 1.0 / math.sqrt(sharer.head_dim)
                Wk[i] = rng.standard_normal((sharer.head_dim, receiver.head_dim)) * scale
                Wv[i] = rng.standard_normal((sharer.head_dim, receiver.head_dim)) * scale
        gate_logit = np.full(receiver.n_layers, _logit(default_gate))
        return cls(align, Wk, Wv, gate_logit)

    def set_gate(self, value: float, layer: Optional[int] = None) -> None:
        lg = -np.inf if value <= 0 else (np.inf if value >= 1 else _logit(value))  # 0/1は厳密に
        if layer is None:
            self.gate_logit[:] = lg
        else:
            self.gate_logit[layer] = lg

    def fuse(self, recv_kv: KVCache, share_kv: KVCache) -> KVCache:
        """受信KVに送信KVを層ごとにゲート融合した新KVを返す。共有プレフィックス長 L のみ融合。"""
        g_all = _sigmoid(self.gate_logit)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (Kr, Vr) in enumerate(recv_kv.layers):
            s = self.align.get(i)
            if s is None or self.Wk[i] is None:
                out.append((Kr.copy(), Vr.copy()))
                continue
            Ks, Vs = share_kv.layers[s]
            Ksp = Ks @ self.Wk[i]              # [heads, seq_s, head_dim_r]
            Vsp = Vs @ self.Wv[i]
            L = min(Kr.shape[1], Ksp.shape[1])  # 共有プレフィックス長（token alignment後の重なり）
            g = g_all[i]
            Kf, Vf = Kr.copy(), Vr.copy()
            Kf[:, :L] = (1 - g) * Kr[:, :L] + g * Ksp[:, :L]
            Vf[:, :L] = (1 - g) * Vr[:, :L] + g * Vsp[:, :L]
            out.append((Kf, Vf))
        return KVCache(out)


# ============================================================
# in-process LM（本番: torch/transformers。KVの取得と注入生成）
#   ※ selftestでは未使用（torch不要）。GPU機で利用。
# ============================================================
class InProcessLM:
    """KVを read/inject できる in-process モデル。C2CではThinker(Sharer)/Worker(Receiver)に使う。"""
    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device).eval()
        self.device = device

    def kv_shape(self) -> KVShape:
        c = self.model.config
        n_kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
        return KVShape(c.num_hidden_layers, n_kv, c.hidden_size // c.num_attention_heads)

    def encode(self, text: str, max_len: int = 4096) -> tuple[KVCache, list[int]]:
        """text を前方計算し past_key_values(KVCache) と token列を返す（Sharer側の文脈エンコード）。"""
        torch = self._torch
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        with torch.no_grad():
            out = self.model(**ids, use_cache=True)
        return from_hf_past(out.past_key_values), ids["input_ids"][0].tolist()

    def generate_from_kv(self, fused: KVCache, prompt: str, max_new_tokens: int = 1024) -> str:
        """融合済みKVで Receiver の生成を駆動する（注入生成）。
        ⚠️ Codex指摘の脆い箇所: (a)新Cache APIはlegacy tupleを受けない版がある→DynamicCacheに変換,
        (b)attention_mask は past+prompt の全長を被覆する必要, (c)position_ids/RoPE は past長ぶんオフセット。
        transformers の版により挙動が変わるため、実機で要検証。"""
        torch = self._torch
        from transformers import DynamicCache
        legacy = tuple((torch.tensor(K, device=self.device).unsqueeze(0),
                        torch.tensor(V, device=self.device).unsqueeze(0)) for (K, V) in fused.layers)
        cache = DynamicCache.from_legacy_cache(legacy)            # (a) 新Cache APIへ変換
        past_len = fused.layers[0][0].shape[1]
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        n_prompt = ids["input_ids"].shape[1]
        attn = torch.ones((1, past_len + n_prompt), device=self.device)   # (b) past+promptを被覆
        pos = torch.arange(past_len, past_len + n_prompt, device=self.device).unsqueeze(0)  # (c) RoPEオフセット
        with torch.no_grad():
            out = self.model.generate(input_ids=ids["input_ids"], attention_mask=attn,
                                      position_ids=pos, past_key_values=cache,
                                      max_new_tokens=max_new_tokens, use_cache=True)
        return self.tok.decode(out[0][n_prompt:], skip_special_tokens=True)

    def forward_logprobs(self, fused: KVCache, query: str, targets: list[str]) -> dict[str, float]:
        """融合KVを past に query を1パス前向き → 最終位置の次トークン分布から各target先頭tokenのlogprobを返す。
        C2C転移の定量指標 Δlogp(target) = logp_gate(target) - logp_gate0(target) を測るための基本要素（Codex推奨）。"""
        torch = self._torch
        from transformers import DynamicCache
        legacy = tuple((torch.tensor(K, device=self.device).unsqueeze(0),
                        torch.tensor(V, device=self.device).unsqueeze(0)) for (K, V) in fused.layers)
        cache = DynamicCache.from_legacy_cache(legacy)
        past_len = fused.layers[0][0].shape[1]
        ids = self.tok(query, return_tensors="pt").to(self.device)
        n = ids["input_ids"].shape[1]
        attn = torch.ones((1, past_len + n), device=self.device)
        pos = torch.arange(past_len, past_len + n, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(input_ids=ids["input_ids"], attention_mask=attn,
                             position_ids=pos, past_key_values=cache, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        res = {}
        for t in targets:
            tid = self.tok(t, add_special_tokens=False)["input_ids"][0]
            res[t] = float(logp[tid])
        return res


# ============================================================
# C2C版 Thinker→Worker エッジ（P0のテキスト受け渡しを置換する統合点）
# ============================================================
def c2c_thinker_to_worker(thinker: InProcessLM, worker: InProcessLM, fuser: C2CFuser,
                          query: str, plan_instruction: str, worker_instruction: str,
                          aligner: TokenAligner = IdentityTokenAligner(),
                          plan_text: Optional[str] = None) -> str:
    """P0のテキスト受け渡しを置換する Thinker→Worker の C2C エッジ。
    1) Thinker(Sharer): query+計画指示 を encode → 計画文脈のKV
    2) Worker(Receiver): query を encode → 受信KV
    3) aligner で送信KVを受信の位置系列へ整合（異種トークナイザ対策）→ fuser.fuse → 融合KV
    4) Worker: 融合KVを種に生成 → 成果物
    plan_text を渡すと「テキスト計画＋潜在融合」のハイブリッド（gate=0で厳密にP0相当＝安全な段階導入）。
    """
    share_kv, share_tok = thinker.encode(f"{query}\n\n{plan_instruction}")
    recv_kv, recv_tok = worker.encode(query)
    share_kv = aligner.align(share_kv, share_tok, recv_tok)
    fused = fuser.fuse(recv_kv, share_kv)
    prompt = worker_instruction if plan_text is None else f"[計画]\n{plan_text}\n\n{worker_instruction}"
    return worker.generate_from_kv(fused, prompt)


# ============================================================
# selftest（numpy: 融合コアの検証。torch/モデル不要）
# ============================================================
def _mock_kv(n_layers, n_heads, seq, head_dim, seed) -> KVCache:
    rng = np.random.default_rng(seed)
    return KVCache([(rng.standard_normal((n_heads, seq, head_dim)),
                     rng.standard_normal((n_heads, seq, head_dim))) for _ in range(n_layers)])


def selftest() -> None:
    # 層整合: 終端から後ろ向き
    assert terminal_alignment(4, 4) == {3: 3, 2: 2, 1: 1, 0: 0}
    assert terminal_alignment(6, 4) == {3: 5, 2: 4, 1: 3, 0: 2}      # 受信4層 ← 送信の上位4層
    print("[selftest] terminal_alignment OK")

    sh = KVShape(n_layers=4, n_heads=2, head_dim=5)
    rh = KVShape(n_layers=4, n_heads=2, head_dim=5)
    recv = _mock_kv(4, 2, 6, 5, seed=1)
    share = _mock_kv(4, 2, 6, 5, seed=2)
    fuser = C2CFuser.init(sh, rh, seed=0)

    # gate=0 → 受信KVと完全一致（＝Worker単独, P0縮退）
    fuser.set_gate(0.0)
    f0 = fuser.fuse(recv, share)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(f0.layers, recv.layers))
    print("[selftest] gate=0 → receiver-only (P0へグレースフル縮退) OK")

    # gate=1 → 射影済み送信KV（恒等射影なので send と一致, 共有長Lの範囲）
    fuser.set_gate(1.0)
    f1 = fuser.fuse(recv, share)
    assert np.allclose(f1.layers[3][0], share.layers[3][0])
    print("[selftest] gate=1 → projected sharer (恒等射影) OK")

    # 形状保存
    assert all(Kf.shape == Kr.shape for (Kf, _), (Kr, _) in zip(f1.layers, recv.layers))
    print("[selftest] shape preserved OK")

    # head_dim不一致 → ランダム射影で形状整合（送信hd=8 → 受信hd=5）
    sh2 = KVShape(4, 2, 8)
    share2 = _mock_kv(4, 2, 6, 8, seed=3)
    fuser2 = C2CFuser.init(sh2, rh, seed=0)
    fuser2.set_gate(0.5)
    f2 = fuser2.fuse(recv, share2)
    assert f2.layers[0][0].shape == (2, 6, 5)
    print("[selftest] head_dim projection (8→5) OK")

    # 共有プレフィックス: 送信seqが短い時は重なりのみ融合、残りは受信のまま
    share_short = _mock_kv(4, 2, 3, 5, seed=4)
    fuser.set_gate(1.0)
    f3 = fuser.fuse(recv, share_short)
    assert np.array_equal(f3.layers[3][0][:, 3:], recv.layers[3][0][:, 3:])   # 重なり外は受信のまま
    print("[selftest] partial-prefix fusion OK")
    print("[selftest] ALL PASSED")


# ============================================================
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("C2C P1 skeleton. 本番は torch/transformers と in-process Thinker/Worker が必要。")
        print("融合コアの検証: python -m trinity.c2c --selftest")
