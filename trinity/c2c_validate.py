"""
P1(a): 同一ファミリ C2C 最小検証
================================
狙い: 「Sharer の KV を Receiver に融合すると、送信側の内容が受信側の生成に転移するか」を
制御実験で確認する。同一ファミリ(同一トークナイザ・互換shape)なら token alignment と GQA の
ブロッカーが消えるため、C2Cの潜在転移そのものを最小条件で検証できる。

- mock検証(torch不要・この場で実行可): TinyMockLM で「gate=0→受信内容 / gate=1→送信内容」を定量確認。
  steering(送信-受信)が gate に対し単調増加することを assert。
- 実機検証(GPU): InProcessLM(Qwen3) で同じ手順。self-C2C(同一モデルを送受) が最もクリーン。
  次に Qwen3-0.6B(Thinker) → Qwen3-Coder(Worker) へ拡張（層整合＋head_dim射影を行使。n_kv_heads一致が条件）。

実行:
  python -m trinity.c2c_validate --selftest   # mock検証(torch不要)
  python -m trinity.c2c_validate --real       # 実機 self-C2C (Qwen3-0.6B)
"""
from __future__ import annotations

import sys
import numpy as np

from trinity.c2c import (
    KVShape, KVCache, C2CFuser, IdentityTokenAligner,
)

VOCAB = "ABCD"          # token id = index。head_dim>=len(VOCAB)。


# ============================================================
# TinyMockLM: InProcessLM 互換の極小ダミー。KVに「どのトークンが居たか」を埋め込む。
# ============================================================
class TinyMockLM:
    def __init__(self, n_layers: int = 4, n_heads: int = 2, head_dim: int = 4):
        self.n_layers, self.n_heads, self.head_dim = n_layers, n_heads, head_dim

    def kv_shape(self) -> KVShape:
        return KVShape(self.n_layers, self.n_heads, self.head_dim)

    def _emb(self, tok_id: int) -> np.ndarray:
        v = np.zeros(self.head_dim)
        v[tok_id % self.head_dim] = 1.0
        return v

    def encode(self, text: str):
        toks = [VOCAB.index(c) for c in text if c in VOCAB]
        seq = max(len(toks), 1)
        layers = []
        for _ in range(self.n_layers):
            K = np.zeros((self.n_heads, seq, self.head_dim))
            V = np.zeros_like(K)
            for p, t in enumerate(toks):
                e = self._emb(t)
                K[:, p, :] = e
                V[:, p, :] = e
            layers.append((K, V))
        return KVCache(layers), toks

    def _aggregate(self, fused: KVCache) -> np.ndarray:
        agg = np.zeros(self.head_dim)
        for (_, V) in fused.layers:
            agg += V.sum(axis=(0, 1))      # 全層・全ヘッド・全位置のVを集約
        return agg

    def generate_from_kv(self, fused: KVCache, prompt: str, max_new_tokens: int = 1) -> str:
        return VOCAB[int(np.argmax(self._aggregate(fused)))]

    def next_logprobs(self, fused: KVCache, targets) -> dict:
        """実機 forward_logprobs の mock版（aggregateを次トークンlogitsとみなす）。"""
        z = self._aggregate(fused)
        z = z - z.max()
        logp = z - np.log(np.exp(z).sum())
        return {t: float(logp[VOCAB.index(t)]) for t in targets}


# ============================================================
# steering 実験: gate を振って「受信内容→送信内容」への転移を測る
# ============================================================
def steering_sweep(lm, fuser: C2CFuser, recv_text: str, share_text: str,
                   gates=(0.0, 0.25, 0.5, 0.75, 1.0), verbose: bool = True):
    recv_kv, recv_tok = lm.encode(recv_text)
    share_kv, share_tok = lm.encode(share_text)
    share_kv = IdentityTokenAligner().align(share_kv, share_tok, recv_tok)   # 同一トークナイザ＝恒等
    rows = []
    for g in gates:
        fuser.set_gate(g)
        fused = fuser.fuse(recv_kv, share_kv)
        out = lm.generate_from_kv(fused, recv_text)
        agg = lm._aggregate(fused) if hasattr(lm, "_aggregate") else None
        steer = float(agg[VOCAB.index("B")] - agg[VOCAB.index("A")]) if agg is not None else None
        rows.append((g, out, steer))
        if verbose:
            s = f"{steer:+.3f}" if steer is not None else "  n/a"
            print(f"  gate={g:<4} output={out}   steering(B-A)={s}")
    return rows


# ============================================================
# mock検証（torch不要）
# ============================================================
def selftest() -> None:
    # ※ これは「配線(encode→fuse→inject→readout)＋ゲート＋指標計算」の検証であり、
    #   実LMの意味的転移そのものの証明ではない（それはGPUの real_validation で）。Codex指摘を明記。
    lm = TinyMockLM(n_layers=4, n_heads=2, head_dim=4)
    fuser = C2CFuser.init(lm.kv_shape(), lm.kv_shape(), seed=0)   # 同一shape→恒等射影
    print("[plumbing] Receiver='AAAA'(自分の内容) / Sharer='BBBB'(注入内容) ※等長・接頭で識別=正しい実験形")
    recv_kv, rt = lm.encode("AAAA")
    share_kv, st = lm.encode("BBBB")
    share_kv = IdentityTokenAligner().align(share_kv, st, rt)

    rows, prev = [], -1e9
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        fuser.set_gate(g)
        fused = fuser.fuse(recv_kv, share_kv)
        out = lm.generate_from_kv(fused, "AAAA")
        steer = float(lm._aggregate(fused)[1] - lm._aggregate(fused)[0])
        rows.append((g, out, steer))
        print(f"  gate={g:<4} output={out}  steering(B-A)={steer:+.2f}")
        assert steer >= prev - 1e-9
        prev = steer

    outs = {g: o for (g, o, _) in rows}
    assert outs[0.0] == "A" and outs[1.0] == "B"
    print("[plumbing] gate=0→受信(A) / gate=1→送信(B) / steering単調増加  ✓")

    # Δ次トークンlogprob 指標（実機 forward_logprobs と同形）を mock で検証
    fuser.set_gate(0.0); lp0 = lm.next_logprobs(fuser.fuse(recv_kv, share_kv), ["A", "B"])
    fuser.set_gate(1.0); lp1 = lm.next_logprobs(fuser.fuse(recv_kv, share_kv), ["A", "B"])
    dB = lp1["B"] - lp0["B"]
    assert dB > 0
    print(f"[plumbing] Δlogp('B') gate0→1 = {dB:+.2f} (>0: 注入で送信tokenの確率が上がる=指標が機能)")
    print("[plumbing] ALL PASSED  ※意味的転移の確認は GPU の --real で")


# ============================================================
# 実機検証（GPU・Qwen3）
# ============================================================
def real_validation_self(model: str = "Qwen/Qwen3-0.6B") -> None:
    """self-C2C 転移の定量検証（Codex指摘の交絡を回避した設計）。
    因果マスク制約: 識別情報は『接頭』に置く（後置だとprefix位置のKVに乗らない）。
    位置整合: send/recv を『等長』にして同一トークナイザの位置対応を成立させる。
    指標: Δ次トークンlogprob。gate↑で送信側ターゲットの確率が上がれば転移が起きている。"""
    from trinity.c2c import InProcessLM
    lm = InProcessLM(model)
    sh = lm.kv_shape()
    fuser = C2CFuser.init(sh, sh, seed=0)

    share_ctx = "Country: France.\nThe capital city is"     # 送信: France文脈（接頭で識別）
    recv_ctx = "Country: Japan.\nThe capital city is"       # 受信: Japan文脈（等長）
    n_s = len(lm.tok(share_ctx, add_special_tokens=False)["input_ids"])
    n_r = len(lm.tok(recv_ctx, add_special_tokens=False)["input_ids"])
    if n_s != n_r:
        print(f"  ⚠️ send/recvが等長でない({n_s}≠{n_r}) → 位置整合が崩れる。語を調整するか TokenAligner を実装。")

    share_kv, stok = lm.encode(share_ctx)
    recv_kv, rtok = lm.encode(recv_ctx)
    share_kv = IdentityTokenAligner().align(share_kv, stok, rtok)
    targets = [" Paris", " Tokyo"]                          # 送信側=Paris / 受信側=Tokyo
    cont = " "                                              # 短い継続で「文脈の次トークン分布」を読む
    print(f"[real] self-C2C {model}: recv=Japan文脈 に France文脈KVを注入 → Δlogp を測る")
    base = None
    for g in (0.0, 0.3, 0.6, 1.0):
        fuser.set_gate(g)
        fused = fuser.fuse(recv_kv, share_kv)
        lp = lm.forward_logprobs(fused, cont, targets)
        if g == 0.0:
            base = lp
        d = {t: round(lp[t] - base[t], 2) for t in targets}
        print(f"  gate={g}: logp={ {t: round(lp[t], 2) for t in targets} }  Δ={d}")
    print("  期待(転移が効けば): gate↑で Δlogp(' Paris')>0, Δlogp(' Tokyo')<0")
    print("  ※ 因果制約・等長前提を満たす最小設計。HF版のKV注入挙動に依存するため実機で要確認。")


def real_validation_cross(thinker_model: str = "Qwen/Qwen3-0.6B",
                          worker_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct") -> None:
    """Qwen3-0.6B(Thinker) → Qwen3-Coder(Worker)。層整合＋head_dim射影を行使。n_kv_heads一致が条件。"""
    from trinity.c2c import InProcessLM
    th, wk = InProcessLM(thinker_model), InProcessLM(worker_model)
    sh, rh = th.kv_shape(), wk.kv_shape()
    print(f"[real-cross] thinker {sh} / worker {rh}")
    if sh.n_heads != rh.n_heads:
        print(f"  ⚠️ n_kv_heads不一致({sh.n_heads}≠{rh.n_heads}) → ヘッド射影が未実装のため fuser.init は失敗する。"
              f"  まずは self-C2C か n_kv_heads一致ペアで検証を。")
        return
    fuser = C2CFuser.init(sh, rh, seed=0)   # 層整合(terminal)＋head_dim射影が効く
    query = "Write a function `add(a,b)` that returns a+b."
    share_kv, stok = th.encode(query + "\n\n(plan: just return a+b)")
    recv_kv, rtok = wk.encode(query)
    share_kv = IdentityTokenAligner().align(share_kv, stok, rtok)
    fuser.set_gate(0.3)
    fused = fuser.fuse(recv_kv, share_kv)
    print("  out:", wk.generate_from_kv(fused, query, max_new_tokens=80)[:200])


# ============================================================
if __name__ == "__main__":
    if "--real" in sys.argv:
        real_validation_self()
    else:
        selftest()
