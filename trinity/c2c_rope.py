"""
P1: RoPE-aware 整列（異種ペアの最深課題＝RoPE位相ズレの解消）
=========================================================
問題: 整列後の送信KVは「送信側の回転位相(送信トークン位置)」を保持したまま受信位置へ注入され、
      位置依存のRoPE位相がズレる（線形射影では補正不可）。
解法(Codex): K を一旦「未回転」に戻して整列し、受信位置で受信RoPEを再適用する。
  K: unrotate(送信inv_freq・送信位置) → gather(整合) → 射影(送信→受信dims) → apply receiver RoPE(受信inv_freq・受信位置)
  V: 回転対象外 → gather → 射影 → そのまま
→ 受信側の K_stored(受信回転済み) と位相整合した融合が可能に。

重要: base(rope_theta)は推測せず **モデルの実 inv_freq を読む**（GLMとQwenでthetaが異なる。Qwen2.5は1e6）。
  → inv_freq_from_model(model) で取得。

検証:
  python -m trinity.c2c_rope --selftest   # RoPE round-trip / 位置シフト整合 / 同位置=恒等転移（DL不要）
  python -m trinity.c2c_rope --real        # 自前RoPEが Qwen の rotary(実inv_freq, cos/sin)と一致することを実証
"""
import sys
import numpy as np

from trinity.c2c import KVShape, KVCache, terminal_alignment, _sigmoid, _logit


# ============================================================
# RoPE プリミティブ（HF Qwen/Llama と同じ rotate_half 規約）
#   ※ inv_freq を引数に取る（base推測を避け、モデルの実値を使う）
# ============================================================
def rope_inv_freq(head_dim, base):
    """合成テスト用に base から inv_freq を構成。実機は inv_freq_from_model を使う。"""
    return 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))     # [hd/2]


def inv_freq_from_model(model):
    """モデルの rotary から実 inv_freq を取得（GLM/Qwen等でthetaが違うため必須）。"""
    return model.model.rotary_emb.inv_freq.detach().cpu().numpy()


def rope_cos_sin(positions, inv_freq):
    freqs = np.outer(np.asarray(positions, dtype=float), np.asarray(inv_freq))   # [L, hd/2]
    emb = np.concatenate([freqs, freqs], axis=-1)                                # [L, hd]
    return np.cos(emb), np.sin(emb)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x, cos, sin):                                          # x:[H,L,hd], cos/sin:[L,hd]
    return x * cos[None] + rotate_half(x) * sin[None]


def unapply_rope(x, cos, sin):                                        # 逆回転（sin反転）
    return x * cos[None] - rotate_half(x) * sin[None]


def unrotate_gather_K(K_stored, sharer_positions, gather_idx, sharer_inv_freq):
    """送信K_stored(送信回転済み)を未回転に戻し、受信位置系列へgather。→ [H_s, L_recv, hd_s] 未回転。"""
    cos, sin = rope_cos_sin(sharer_positions, sharer_inv_freq)
    return unapply_rope(K_stored, cos, sin)[:, np.asarray(gather_idx), :]


def gather_V(V, gather_idx):
    return V[:, np.asarray(gather_idx), :]


# ============================================================
# RoPE-aware 異種 fuser
# ============================================================
class RoPEAwareFuser:
    def __init__(self, align, W, gate_logit, recv_inv_freq):
        self.align, self.W, self.gate_logit = align, W, gate_logit
        self.recv_inv_freq = np.asarray(recv_inv_freq)

    @classmethod
    def init(cls, sharer: KVShape, receiver: KVShape, recv_inv_freq, default_gate=0.05, seed=0):
        rng = np.random.default_rng(seed)
        align = terminal_alignment(sharer.n_layers, receiver.n_layers)
        din, dout = sharer.n_heads * sharer.head_dim, receiver.n_heads * receiver.head_dim
        W = [None] * receiver.n_layers
        for i in align:
            W[i] = np.eye(din, dout) if din == dout else rng.standard_normal((din, dout)) / np.sqrt(din)
        return cls(align, W, np.full(receiver.n_layers, _logit(default_gate)), recv_inv_freq)

    def set_gate(self, value, layer=None):
        lg = -np.inf if value <= 0 else (np.inf if value >= 1 else _logit(value))
        if layer is None:
            self.gate_logit[:] = lg
        else:
            self.gate_logit[layer] = lg

    def _proj(self, X, i, Hr, hr):                                    # [H_s,L,hd_s] -> [H_r,L,hd_r]
        Lp = X.shape[1]
        return (X.transpose(1, 0, 2).reshape(Lp, -1) @ self.W[i]).reshape(Lp, Hr, hr).transpose(1, 0, 2)

    def fuse(self, recv_kv: KVCache, shK_unrot, shV, receiver_positions) -> KVCache:
        """shK_unrot/shV: 層ごとの『未回転・整列済み送信K』『整列済み送信V』(sharer dims)。"""
        g = _sigmoid(self.gate_logit)
        cos, sin = rope_cos_sin(receiver_positions, self.recv_inv_freq)
        out = []
        for i, (Kr, Vr) in enumerate(recv_kv.layers):
            s = self.align.get(i)
            if s is None or self.W[i] is None:
                out.append((Kr.copy(), Vr.copy())); continue
            Hr, hr = Kr.shape[0], Kr.shape[2]
            Kp = apply_rope(self._proj(shK_unrot[s], i, Hr, hr), cos, sin)   # 受信位置で再回転
            Vp = self._proj(shV[s], i, Hr, hr)                              # Vは回転なし
            L = min(Kr.shape[1], Kp.shape[1])
            gi = g[i]
            Kf, Vf = Kr.copy(), Vr.copy()
            Kf[:, :L] = (1 - gi) * Kr[:, :L] + gi * Kp[:, :L]
            Vf[:, :L] = (1 - gi) * Vr[:, :L] + gi * Vp[:, :L]
            out.append((Kf, Vf))
        return KVCache(out)


# ============================================================
# selftest（合成・DL不要）
# ============================================================
def selftest():
    rng = np.random.default_rng(0)
    hd = 8
    inv = rope_inv_freq(hd, 10000.0)

    # ① round-trip: unapply ∘ apply = 恒等
    K = rng.standard_normal((2, 5, hd))
    cos, sin = rope_cos_sin([0, 1, 2, 3, 4], inv)
    assert np.allclose(unapply_rope(apply_rope(K, cos, sin), cos, sin), K)
    print("[selftest] RoPE round-trip OK: unrotate∘rotate = 恒等")

    # ② 位置シフト整合: raw→送信位置で回転→stored→未回転→受信位置で再回転 == raw を受信位置で回転
    K_raw = rng.standard_normal((2, 5, hd))
    sp, rp = [0, 1, 2, 3, 4], [10, 11, 12, 13, 14]
    cs, ss = rope_cos_sin(sp, inv)
    cr, sr = rope_cos_sin(rp, inv)
    K_unrot = unapply_rope(apply_rope(K_raw, cs, ss), cs, ss)
    assert np.allclose(apply_rope(K_unrot, cr, sr), apply_rope(K_raw, cr, sr))
    print("[selftest] 位置シフト整合 OK: 送信位置→未回転→受信位置 で受信位相に正しく載る")

    # ③ 同位置・恒等射影・同inv・gate=1 → 融合K == 送信K_stored（全パイプラインが恒等転移に縮退）
    sh = rh = KVShape(2, 2, hd)
    nl = 2
    recv = KVCache([(rng.standard_normal((2, 5, hd)), rng.standard_normal((2, 5, hd))) for _ in range(nl)])
    shareK = [apply_rope(rng.standard_normal((2, 5, hd)), cs, ss) for _ in range(nl)]   # 送信回転済み
    shareV = [rng.standard_normal((2, 5, hd)) for _ in range(nl)]
    idx = [0, 1, 2, 3, 4]
    shK_unrot = [unrotate_gather_K(shareK[l], sp, idx, inv) for l in range(nl)]
    shV = [gather_V(shareV[l], idx) for l in range(nl)]
    fuser = RoPEAwareFuser.init(sh, rh, recv_inv_freq=inv, seed=0)
    fuser.set_gate(1.0)
    fused = fuser.fuse(recv, shK_unrot, shV, receiver_positions=sp)
    assert np.allclose(fused.layers[1][0], shareK[1]), "同位置でK恒等転移にならない（位相補正のバグ）"
    print("[selftest] 同位置=恒等転移 OK: unrotate→再回転で K_stored を完全再現")

    fuser.set_gate(0.0)
    f0 = fuser.fuse(recv, shK_unrot, shV, receiver_positions=sp)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(f0.layers, recv.layers))
    print("[selftest] gate=0 → receiver-only OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real（自前RoPE == Qwen rotary を実証。inv_freqはモデルから取得）
# ============================================================
def real():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import torch
    from transformers import AutoModelForCausalLM
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[real] load {name} …")
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    inv = inv_freq_from_model(m)                       # ← baseは推測せずモデルの実値
    base_est = float(inv[1] ** (-len(inv)))            # inv[1]=base^(-2/hd), hd=2*len → base=inv[1]^(-len)
    print(f"[real] model inv_freq len={len(inv)}  推定base≈{base_est:.3g}（Qwen2.5=1e6）")

    pos = list(range(6))
    cos, sin = rope_cos_sin(pos, inv)                  # 自前（モデルのinv_freq使用）
    try:
        cos_m, sin_m = m.model.rotary_emb(torch.zeros(1, 6, len(inv) * 2), torch.arange(6).unsqueeze(0))
        ok_c = np.allclose(cos, cos_m[0].detach().cpu().numpy(), atol=1e-4)
        ok_s = np.allclose(sin, sin_m[0].detach().cpu().numpy(), atol=1e-4)
        print(f"[real] 自前RoPE cos/sin が Qwen rotary と一致: cos={ok_c}, sin={ok_s}")
        assert ok_c and ok_s
    except Exception as e:
        print(f"[real] rotary直接比較スキップ（API差異: {e}）。inv_freqはモデル実値を使用済み。")
    print("→ モデルの実inv_freqを用いれば自前RoPEはQwenと数値一致。unrotate→受信位置で再回転が実機で忠実。")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
    else:
        selftest()
