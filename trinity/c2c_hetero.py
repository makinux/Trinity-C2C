"""
P1: heterogeneous-pair support (e.g. GLM Thinker -> Qwen Worker)
===============================================
A minimal implementation that solves C2C's two big blockers:
  (1) tokenizer alignment: KV at position i does not correspond in meaning across sharer/receiver. Using the
     "character spans" of the same text, build a sharer-token -> receiver-token mapping and reorder the sharer KV into the receiver's position sequence.
  (2) GQA/head mismatch: num_key_value_heads / head_dim differ -> adapt per layer via a
     flatten projection W:[H_s*hd_s -> H_r*hd_r] (learned; identity-initialized when shapes match).
  (3) layer-count mismatch: terminal alignment (existing).
  (4) [!] RoPE phase (Codex's top risk): the aligned sharer KV keeps "the sharer-side rotation phase (sharer token position)"
     when injected at the receiver position, so the position-dependent RoPE phase drifts. A linear projection W generally cannot correct it.
     -> Does not occur for the same tokenizer (= matching positions) (that is why the earlier self-C2C worked).
       For true heterogeneity, RoPE-aware alignment such as "align the un-rotated K -> re-apply the receiver RoPE" is needed (the real crux, to be implemented).
  Note: the flatten projection in (2) is a simplified version that mixes all KV heads. To respect GQA's group locality,
     a per-receiver-KV-head/group projection is preferable (room for refinement).

Validation:
  python -m trinity.c2c_hetero --selftest    # synthetic: character-span alignment + heterogeneous fuse (no download)
  python -m trinity.c2c_hetero --real-tok     # demonstrate alignment with two real tokenizers (Qwen <-> gpt2)
"""
import sys
import numpy as np

from trinity.c2c import KVShape, KVCache, TokenAligner, terminal_alignment, _sigmoid, _logit


# ============================================================
# (1) tokenizer alignment (character spans)
# ============================================================
def char_span_align(recv_offsets, share_offsets):
    """From the offset lists of the same text split by different tokenizers,
    return for each receiver token the index of the sharer token whose character span overlaps most."""
    smap = []
    for (ra, rb) in recv_offsets:
        best, best_ov = -1, 0                            # accept only positive overlap (best_ov starts at 0)
        for k, (sa, sb) in enumerate(share_offsets):
            ov = min(rb, sb) - max(ra, sa)               # character-overlap length
            if ov > best_ov:
                best_ov, best = ov, k
        if best < 0:                                     # no overlap (gap/whitespace/special token) -> fall back to nearest midpoint
            rm = (ra + rb) / 2
            best = int(np.argmin([abs((sa + sb) / 2 - rm) for (sa, sb) in share_offsets])) if share_offsets else 0
        smap.append(best)
    return smap


class CharSpanTokenAligner(TokenAligner):
    """A TokenAligner for heterogeneous tokenizers. Reorders the sharer KV into the receiver's token-position sequence.
    Needs offsets (HF fast tokenizer's return_offsets_mapping=True)."""
    def __init__(self, recv_offsets, share_offsets):
        self.map = char_span_align(recv_offsets, share_offsets)

    def align(self, share_kv: KVCache, share_tokens=None, recv_tokens=None) -> KVCache:
        idx = np.asarray(self.map)                       # receiver position -> sharer position
        return KVCache([(K[:, idx, :], V[:, idx, :]) for (K, V) in share_kv.layers])


# ============================================================
# (2) heterogeneous fuser (absorb head/dim mismatch via a flatten projection)
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
        """share_kv_aligned is already aligned to the receiver positions by CharSpanTokenAligner (seq = receiver length)."""
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
# selftest (synthetic, no download)
# ============================================================
def selftest():
    # (1) character-span alignment: split "hello world" into receiver[hel|lo| wor|ld] / sharer[hello| world]
    recv_off = [(0, 3), (3, 5), (5, 9), (9, 11)]
    share_off = [(0, 5), (5, 11)]
    m = char_span_align(recv_off, share_off)
    assert m == [0, 0, 1, 1], m
    print(f"[selftest] char-span alignment OK: {m}  (hel,lo->hello / wor,ld-> world)")

    # (2) heterogeneous fuse: sharer(H=4,hd=8,4 layers,seq7) -> receiver(H=2,hd=64,3 layers,seq5)
    sh, rh = KVShape(4, 4, 8), KVShape(3, 2, 64)
    rng = np.random.default_rng(0)
    recv = KVCache([(rng.standard_normal((2, 5, 64)), rng.standard_normal((2, 5, 64))) for _ in range(3)])
    share = KVCache([(rng.standard_normal((4, 7, 8)), rng.standard_normal((4, 7, 8))) for _ in range(4)])

    aligner = CharSpanTokenAligner.__new__(CharSpanTokenAligner)
    aligner.map = [0, 1, 2, 3, 4]                        # sharer seq7 -> receiver seq5 position correspondence (first 5)
    share_al = aligner.align(share)
    assert share_al.layers[0][0].shape == (4, 5, 8), share_al.layers[0][0].shape
    print("[selftest] token realignment OK: sharer KV aligned to the receiver's position sequence (seq5)")

    fuser = HeteroC2CFuser.init(sh, rh, seed=0)
    fuser.set_gate(0.5)
    fused = fuser.fuse(recv, share_al)
    assert all(Kf.shape == (2, 5, 64) for (Kf, _) in fused.layers), "receiver shape not preserved"
    print("[selftest] hetero fuse OK: absorbs H4*hd8 -> H2*hd64 / 4 layers -> 3 layers and preserves the receiver shape")

    fuser.set_gate(0.0)                                  # gate=0 -> receiver-only
    f0 = fuser.fuse(recv, share_al)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(f0.layers, recv.layers))
    print("[selftest] gate=0 -> receiver-only (graceful degradation) OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real-tok (demonstrate alignment with two real tokenizers)
# ============================================================
def real_tok_demo():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from transformers import AutoTokenizer
    rt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")   # receiver side
    st = AutoTokenizer.from_pretrained("gpt2")                          # sharer side (different-lineage tokenizer)
    text = "Country: France. The capital city is"
    ro = rt(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    so = st(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    m = char_span_align(ro, so)
    print(f"[real-tok] recv(Qwen)={len(ro)}tok / share(gpt2)={len(so)}tok -> alignment map len={len(m)}")
    ok = all((min(ro[i][1], so[m[i]][1]) - max(ro[i][0], so[m[i]][0])) > 0 for i in range(len(ro)))
    print(f"[real-tok] every receiver token maps to an overlapping sharer token: {ok}")
    for i in range(min(6, len(ro))):
        print(f"   recv[{i}] {text[ro[i][0]:ro[i][1]]!r} -> share[{m[i]}] {text[so[m[i]][0]:so[m[i]][1]]!r}")
    print("-> even with different tokenizers, character spans align the sharer KV to receiver positions (resolves blocker (1)).")


if __name__ == "__main__":
    if "--real-tok" in sys.argv:
        real_tok_demo()
    else:
        selftest()
