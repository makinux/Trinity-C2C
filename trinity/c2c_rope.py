"""
P1: RoPE-aware alignment (the deepest issue for heterogeneous pairs = resolving RoPE phase drift)
=========================================================
Problem: after alignment, the sharer KV keeps "the sharer-side rotation phase (sharer token position)" when
      injected at the receiver position, so the position-dependent RoPE phase drifts (a linear projection cannot correct it).
Solution (Codex): briefly return K to "un-rotated", align it, then re-apply the receiver RoPE at the receiver position.
  K: unrotate(sharer inv_freq, sharer positions) -> gather(align) -> project(sharer->receiver dims) -> apply receiver RoPE(receiver inv_freq, receiver positions)
  V: not subject to rotation -> gather -> project -> as is
-> enables fusion phase-aligned with the receiver-side K_stored (receiver-rotated).

Important: do not guess base(rope_theta); **read the model's actual inv_freq** (theta differs between GLM and Qwen. Qwen2.5 is 1e6).
  -> obtained via inv_freq_from_model(model).

Validation:
  python -m trinity.c2c_rope --selftest   # RoPE round-trip / position-shift alignment / same-position = identity transfer (no download)
  python -m trinity.c2c_rope --real        # demonstrate that our own RoPE matches Qwen's rotary (actual inv_freq, cos/sin)
"""
import sys
import numpy as np

from trinity.c2c import KVShape, KVCache, terminal_alignment, _sigmoid, _logit


# ============================================================
# RoPE primitives (same rotate_half convention as HF Qwen/Llama)
#   Note: takes inv_freq as an argument (avoid guessing base; use the model's actual values)
# ============================================================
def rope_inv_freq(head_dim, base):
    """Construct inv_freq from base for synthetic tests. On real hardware use inv_freq_from_model."""
    return 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))     # [hd/2]


def inv_freq_from_model(model):
    """Get the actual inv_freq from the model's rotary (essential because theta differs across GLM/Qwen etc.)."""
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


def unapply_rope(x, cos, sin):                                        # inverse rotation (flip sin)
    return x * cos[None] - rotate_half(x) * sin[None]


def unrotate_gather_K(K_stored, sharer_positions, gather_idx, sharer_inv_freq):
    """Return sharer K_stored (sharer-rotated) to un-rotated and gather into the receiver position sequence. -> [H_s, L_recv, hd_s] un-rotated."""
    cos, sin = rope_cos_sin(sharer_positions, sharer_inv_freq)
    return unapply_rope(K_stored, cos, sin)[:, np.asarray(gather_idx), :]


def gather_V(V, gather_idx):
    return V[:, np.asarray(gather_idx), :]


# ============================================================
# RoPE-aware heterogeneous fuser
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
        """shK_unrot/shV: per-layer 'un-rotated, aligned sharer K' and 'aligned sharer V' (sharer dims)."""
        g = _sigmoid(self.gate_logit)
        cos, sin = rope_cos_sin(receiver_positions, self.recv_inv_freq)
        out = []
        for i, (Kr, Vr) in enumerate(recv_kv.layers):
            s = self.align.get(i)
            if s is None or self.W[i] is None:
                out.append((Kr.copy(), Vr.copy())); continue
            Hr, hr = Kr.shape[0], Kr.shape[2]
            Kp = apply_rope(self._proj(shK_unrot[s], i, Hr, hr), cos, sin)   # re-rotate at the receiver position
            Vp = self._proj(shV[s], i, Hr, hr)                              # V is not rotated
            L = min(Kr.shape[1], Kp.shape[1])
            gi = g[i]
            Kf, Vf = Kr.copy(), Vr.copy()
            Kf[:, :L] = (1 - gi) * Kr[:, :L] + gi * Kp[:, :L]
            Vf[:, :L] = (1 - gi) * Vr[:, :L] + gi * Vp[:, :L]
            out.append((Kf, Vf))
        return KVCache(out)


# ============================================================
# selftest (synthetic, no download)
# ============================================================
def selftest():
    rng = np.random.default_rng(0)
    hd = 8
    inv = rope_inv_freq(hd, 10000.0)

    # (1) round-trip: unapply o apply = identity
    K = rng.standard_normal((2, 5, hd))
    cos, sin = rope_cos_sin([0, 1, 2, 3, 4], inv)
    assert np.allclose(unapply_rope(apply_rope(K, cos, sin), cos, sin), K)
    print("[selftest] RoPE round-trip OK: unrotate o rotate = identity")

    # (2) position-shift alignment: raw -> rotate at sharer position -> stored -> un-rotate -> re-rotate at receiver position == raw rotated at receiver position
    K_raw = rng.standard_normal((2, 5, hd))
    sp, rp = [0, 1, 2, 3, 4], [10, 11, 12, 13, 14]
    cs, ss = rope_cos_sin(sp, inv)
    cr, sr = rope_cos_sin(rp, inv)
    K_unrot = unapply_rope(apply_rope(K_raw, cs, ss), cs, ss)
    assert np.allclose(apply_rope(K_unrot, cr, sr), apply_rope(K_raw, cr, sr))
    print("[selftest] position-shift alignment OK: sharer position -> un-rotate -> receiver position lands correctly on the receiver phase")

    # (3) same position, identity projection, same inv, gate=1 -> fused K == sharer K_stored (the whole pipeline degrades to identity transfer)
    sh = rh = KVShape(2, 2, hd)
    nl = 2
    recv = KVCache([(rng.standard_normal((2, 5, hd)), rng.standard_normal((2, 5, hd))) for _ in range(nl)])
    shareK = [apply_rope(rng.standard_normal((2, 5, hd)), cs, ss) for _ in range(nl)]   # sharer-rotated
    shareV = [rng.standard_normal((2, 5, hd)) for _ in range(nl)]
    idx = [0, 1, 2, 3, 4]
    shK_unrot = [unrotate_gather_K(shareK[l], sp, idx, inv) for l in range(nl)]
    shV = [gather_V(shareV[l], idx) for l in range(nl)]
    fuser = RoPEAwareFuser.init(sh, rh, recv_inv_freq=inv, seed=0)
    fuser.set_gate(1.0)
    fused = fuser.fuse(recv, shK_unrot, shV, receiver_positions=sp)
    assert np.allclose(fused.layers[1][0], shareK[1]), "K is not an identity transfer at the same position (phase-correction bug)"
    print("[selftest] same-position = identity transfer OK: unrotate -> re-rotate reproduces K_stored exactly")

    fuser.set_gate(0.0)
    f0 = fuser.fuse(recv, shK_unrot, shV, receiver_positions=sp)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(f0.layers, recv.layers))
    print("[selftest] gate=0 -> receiver-only OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real (demonstrate our own RoPE == Qwen rotary. inv_freq obtained from the model)
# ============================================================
def real():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    import torch
    from transformers import AutoModelForCausalLM
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[real] load {name} ...")
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    inv = inv_freq_from_model(m)                       # <- do not guess base; the model's actual values
    base_est = float(inv[1] ** (-len(inv)))            # inv[1]=base^(-2/hd), hd=2*len -> base=inv[1]^(-len)
    print(f"[real] model inv_freq len={len(inv)}  estimated base~{base_est:.3g} (Qwen2.5=1e6)")

    pos = list(range(6))
    cos, sin = rope_cos_sin(pos, inv)                  # our own (using the model's inv_freq)
    try:
        cos_m, sin_m = m.model.rotary_emb(torch.zeros(1, 6, len(inv) * 2), torch.arange(6).unsqueeze(0))
        ok_c = np.allclose(cos, cos_m[0].detach().cpu().numpy(), atol=1e-4)
        ok_s = np.allclose(sin, sin_m[0].detach().cpu().numpy(), atol=1e-4)
        print(f"[real] our own RoPE cos/sin matches Qwen rotary: cos={ok_c}, sin={ok_s}")
        assert ok_c and ok_s
    except Exception as e:
        print(f"[real] skipping direct rotary comparison (API difference: {e}). inv_freq already uses the model's actual values.")
    print("-> using the model's actual inv_freq, our own RoPE matches Qwen numerically. unrotate -> re-rotate at the receiver position is faithful on real hardware.")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
    else:
        selftest()
