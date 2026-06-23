"""
P1: 異種ペア対応（GLM Thinker → Qwen Worker 等）
===============================================
C2C の2大ブロッカーを解く最小実装:
  ① トークナイザ整合: 位置iのKVは送受で意味が対応しない。同一テキストの「文字スパン」で
     送信トークン→受信トークンの対応を作り、送信KVを受信の位置系列へ並べ替える。
  ② GQA/ヘッド不一致: num_key_value_heads・head_dim が違う → 層ごとに
     flatten射影 W:[H_s·hd_s → H_r·hd_r] で適応（学習対象。同形なら恒等初期化）。
  ③ 層数不一致: terminal alignment（既存）。
  ④ ⚠️ RoPE位相（Codex指摘の最大リスク）: 整列後の送信KVは「送信側の回転位相(送信トークン位置)」を
     保持したまま受信位置へ注入されるため、位置依存のRoPE位相がズレる。線形射影Wでは一般に補正不可。
     → 同一トークナイザ(=位置一致)なら発生しない（前回の self-C2C が効いたのはこのため）。
       真の異種では「未回転Kを整列→受信RoPEを再適用」等のRoPE-aware整列が必要（要実装の本丸）。
  注: ②の flatten射影は全KVヘッドを混合する簡易版。GQAのグループ局所性を尊重するなら
     受信KVヘッド/グループ単位の射影が望ましい（精緻化の余地）。

検証:
  python -m trinity.c2c_hetero --selftest    # 合成: 文字スパン整合 + 異形fuse（DL不要）
  python -m trinity.c2c_hetero --real-tok     # 実トークナイザ2種(Qwen↔gpt2)で整合を実演
"""
import sys
import numpy as np

from trinity.c2c import KVShape, KVCache, TokenAligner, terminal_alignment, _sigmoid, _logit


# ============================================================
# ① トークナイザ整合（文字スパン）
# ============================================================
def char_span_align(recv_offsets, share_offsets):
    """同一テキストを別トークナイザで分割した offset 列から、
    各受信トークン → 最も文字スパンが重なる送信トークン index を返す。"""
    smap = []
    for (ra, rb) in recv_offsets:
        best, best_ov = -1, 0                            # 正の重なりのみ採用（best_ov=0始点）
        for k, (sa, sb) in enumerate(share_offsets):
            ov = min(rb, sb) - max(ra, sa)               # 文字重なり長
            if ov > best_ov:
                best_ov, best = ov, k
        if best < 0:                                     # 重なり無し(gap/空白/特殊トークン) → 中点最近傍へ
            rm = (ra + rb) / 2
            best = int(np.argmin([abs((sa + sb) / 2 - rm) for (sa, sb) in share_offsets])) if share_offsets else 0
        smap.append(best)
    return smap


class CharSpanTokenAligner(TokenAligner):
    """異種トークナイザ対応の TokenAligner。送信KVを受信のトークン位置系列へ並べ替える。
    要 offsets（HF fast tokenizer の return_offsets_mapping=True）。"""
    def __init__(self, recv_offsets, share_offsets):
        self.map = char_span_align(recv_offsets, share_offsets)

    def align(self, share_kv: KVCache, share_tokens=None, recv_tokens=None) -> KVCache:
        idx = np.asarray(self.map)                       # 受信位置 → 送信位置
        return KVCache([(K[:, idx, :], V[:, idx, :]) for (K, V) in share_kv.layers])


# ============================================================
# ② 異形 fuser（ヘッド/次元の不一致を flatten射影で吸収）
# ============================================================
class HeteroC2CFuser:
    def __init__(self, align, W, gate_logit):
        self.align, self.W, self.gate_logit = align, W, gate_logit

    @classmethod
    def init(cls, sharer: KVShape, receiver: KVShape, default_gate: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        align = terminal_alignment(sharer.n_layers, receiver.n_layers)
        din, dout = sharer.n_heads * sharer.head_dim, receiver.n_heads * receiver.head_dim
        W = [None] * receiver.n_layers
        for i in align:
            W[i] = np.eye(din, dout) if din == dout else rng.standard_normal((din, dout)) / np.sqrt(din)
        return cls(align, W, np.full(receiver.n_layers, _logit(default_gate)))

    def set_gate(self, value, layer=None):
        lg = -np.inf if value <= 0 else (np.inf if value >= 1 else _logit(value))
        if layer is None:
            self.gate_logit[:] = lg
        else:
            self.gate_logit[layer] = lg

    def fuse(self, recv_kv: KVCache, share_kv_aligned: KVCache) -> KVCache:
        """share_kv_aligned は CharSpanTokenAligner で受信位置に整合済み（seq=受信長）。"""
        g = _sigmoid(self.gate_logit)
        out = []
        for i, (Kr, Vr) in enumerate(recv_kv.layers):
            s = self.align.get(i)
            if s is None or self.W[i] is None:
                out.append((Kr.copy(), Vr.copy())); continue
            Ks, Vs = share_kv_aligned.layers[s]
            Hr, hr = Kr.shape[0], Kr.shape[2]

            def proj(X):                                  # [H_s,L,hd_s] -> [H_r,L,hd_r]
                Lp = X.shape[1]
                return (X.transpose(1, 0, 2).reshape(Lp, -1) @ self.W[i]).reshape(Lp, Hr, hr).transpose(1, 0, 2)

            Ksp, Vsp = proj(Ks), proj(Vs)
            L = min(Kr.shape[1], Ksp.shape[1])
            gi = g[i]
            Kf, Vf = Kr.copy(), Vr.copy()
            Kf[:, :L] = (1 - gi) * Kr[:, :L] + gi * Ksp[:, :L]
            Vf[:, :L] = (1 - gi) * Vr[:, :L] + gi * Vsp[:, :L]
            out.append((Kf, Vf))
        return KVCache(out)


# ============================================================
# selftest（合成・DL不要）
# ============================================================
def selftest():
    # ① 文字スパン整合: "hello world" を 受信[hel|lo| wor|ld] / 送信[hello| world] に分割
    recv_off = [(0, 3), (3, 5), (5, 9), (9, 11)]
    share_off = [(0, 5), (5, 11)]
    m = char_span_align(recv_off, share_off)
    assert m == [0, 0, 1, 1], m
    print(f"[selftest] char-span alignment OK: {m}  (hel,lo→hello / wor,ld→ world)")

    # ② 異形 fuse: 送信(H=4,hd=8,4層,seq7) → 受信(H=2,hd=64,3層,seq5)
    sh, rh = KVShape(4, 4, 8), KVShape(3, 2, 64)
    rng = np.random.default_rng(0)
    recv = KVCache([(rng.standard_normal((2, 5, 64)), rng.standard_normal((2, 5, 64))) for _ in range(3)])
    share = KVCache([(rng.standard_normal((4, 7, 8)), rng.standard_normal((4, 7, 8))) for _ in range(4)])

    aligner = CharSpanTokenAligner.__new__(CharSpanTokenAligner)
    aligner.map = [0, 1, 2, 3, 4]                        # 送信seq7 → 受信seq5 の位置対応（先頭5）
    share_al = aligner.align(share)
    assert share_al.layers[0][0].shape == (4, 5, 8), share_al.layers[0][0].shape
    print("[selftest] token realignment OK: 送信KVが受信の位置系列(seq5)へ整列")

    fuser = HeteroC2CFuser.init(sh, rh, seed=0)
    fuser.set_gate(0.5)
    fused = fuser.fuse(recv, share_al)
    assert all(Kf.shape == (2, 5, 64) for (Kf, _) in fused.layers), "受信形状を維持できていない"
    print("[selftest] hetero fuse OK: H4·hd8 → H2·hd64 / 4層→3層 を吸収し受信形状を維持")

    fuser.set_gate(0.0)                                  # gate=0 → receiver-only
    f0 = fuser.fuse(recv, share_al)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(f0.layers, recv.layers))
    print("[selftest] gate=0 → receiver-only (グレースフル縮退) OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real-tok（実トークナイザ2種で整合を実演）
# ============================================================
def real_tok_demo():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from transformers import AutoTokenizer
    rt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")   # 受信側
    st = AutoTokenizer.from_pretrained("gpt2")                          # 送信側（別系統トークナイザ）
    text = "Country: France. The capital city is"
    ro = rt(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    so = st(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    m = char_span_align(ro, so)
    print(f"[real-tok] recv(Qwen)={len(ro)}tok / share(gpt2)={len(so)}tok → 整合map len={len(m)}")
    ok = all((min(ro[i][1], so[m[i]][1]) - max(ro[i][0], so[m[i]][0])) > 0 for i in range(len(ro)))
    print(f"[real-tok] 全受信トークンが重なる送信トークンへ対応: {ok}")
    for i in range(min(6, len(ro))):
        print(f"   recv[{i}] {text[ro[i][0]:ro[i][1]]!r} -> share[{m[i]}] {text[so[m[i]][0]:so[m[i]][1]]!r}")
    print("→ 異なるトークナイザでも文字スパンで送信KVを受信位置へ整合できる（ブロッカー①を解消）。")


if __name__ == "__main__":
    if "--real-tok" in sys.argv:
        real_tok_demo()
    else:
        selftest()
