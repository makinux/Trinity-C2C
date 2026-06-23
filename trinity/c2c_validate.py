"""
P1(a): minimal same-family C2C validation
================================
Aim: confirm via a controlled experiment whether "fusing the Sharer's KV into the Receiver transfers
the sharer-side content into the receiver's generation". For the same family (same tokenizer, compatible
shapes) the token-alignment and GQA blockers disappear, so the C2C latent transfer itself can be
validated under minimal conditions.

- mock check (no torch, runnable here): with TinyMockLM, quantitatively confirm "gate=0 -> receiver content / gate=1 -> sharer content".
  Assert that steering (sharer - receiver) increases monotonically in the gate.
- real check (GPU): the same procedure with InProcessLM(Qwen3). self-C2C (same model as sharer and receiver) is cleanest.
  Then extend to Qwen3-0.6B(Thinker) -> Qwen3-Coder(Worker) (exercising layer alignment + head_dim projection; requires matching n_kv_heads).

Run:
  python -m trinity.c2c_validate --selftest   # mock check (no torch)
  python -m trinity.c2c_validate --real       # real self-C2C (Qwen3-0.6B)
"""
from __future__ import annotations

import sys
import numpy as np

from trinity.c2c import (
    KVShape, KVCache, C2CFuser, IdentityTokenAligner,
)

VOCAB = "ABCD"          # token id = index. head_dim >= len(VOCAB).


# ============================================================
# TinyMockLM: a tiny InProcessLM-compatible dummy. Embeds "which token was present" into the KV.
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
            agg += V.sum(axis=(0, 1))      # aggregate V over all layers, heads, and positions
        return agg

    def generate_from_kv(self, fused: KVCache, prompt: str, max_new_tokens: int = 1) -> str:
        return VOCAB[int(np.argmax(self._aggregate(fused)))]

    def next_logprobs(self, fused: KVCache, targets) -> dict:
        """Mock of the real forward_logprobs (treats the aggregate as next-token logits)."""
        z = self._aggregate(fused)
        z = z - z.max()
        logp = z - np.log(np.exp(z).sum())
        return {t: float(logp[VOCAB.index(t)]) for t in targets}


# ============================================================
# steering experiment: sweep the gate to measure transfer from "receiver content -> sharer content"
# ============================================================
def steering_sweep(lm, fuser: C2CFuser, recv_text: str, share_text: str,
                   gates=(0.0, 0.25, 0.5, 0.75, 1.0), verbose: bool = True):
    recv_kv, recv_tok = lm.encode(recv_text)
    share_kv, share_tok = lm.encode(share_text)
    share_kv = IdentityTokenAligner().align(share_kv, share_tok, recv_tok)   # same tokenizer = identity
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
# mock check (no torch)
# ============================================================
def selftest() -> None:
    # Note: this validates the wiring (encode -> fuse -> inject -> readout) + gate + metric computation,
    #   not a proof of semantic transfer in a real LM (that's the GPU real_validation). Stated per Codex.
    lm = TinyMockLM(n_layers=4, n_heads=2, head_dim=4)
    fuser = C2CFuser.init(lm.kv_shape(), lm.kv_shape(), seed=0)   # same shape -> identity projection
    print("[plumbing] Receiver='AAAA' (own content) / Sharer='BBBB' (injected content). Note: equal length, distinguished by prefix = correct experimental form")
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
    print("[plumbing] gate=0 -> receiver(A) / gate=1 -> sharer(B) / steering monotonic increase  OK")

    # validate the delta next-token-logprob metric (same form as the real forward_logprobs) with the mock
    fuser.set_gate(0.0); lp0 = lm.next_logprobs(fuser.fuse(recv_kv, share_kv), ["A", "B"])
    fuser.set_gate(1.0); lp1 = lm.next_logprobs(fuser.fuse(recv_kv, share_kv), ["A", "B"])
    dB = lp1["B"] - lp0["B"]
    assert dB > 0
    print(f"[plumbing] delta-logp('B') gate0->1 = {dB:+.2f} (>0: injection raises the sharer token's probability = the metric works)")
    print("[plumbing] ALL PASSED  Note: confirm semantic transfer with GPU --real")


# ============================================================
# real check (GPU, Qwen3)
# ============================================================
def real_validation_self(model: str = "Qwen/Qwen3-0.6B") -> None:
    """Quantitative validation of self-C2C transfer (designed to avoid the confounds Codex flagged).
    Causal-mask constraint: put the distinguishing info in the 'prefix' (a suffix would not land on the prefix-position KV).
    Position alignment: make send/recv 'equal length' so the same-tokenizer position correspondence holds.
    Metric: delta next-token-logprob. If the sharer-side target's probability rises as the gate goes up, transfer is happening."""
    from trinity.c2c import InProcessLM
    lm = InProcessLM(model)
    sh = lm.kv_shape()
    fuser = C2CFuser.init(sh, sh, seed=0)

    share_ctx = "Country: France.\nThe capital city is"     # sharer: France context (distinguished by prefix)
    recv_ctx = "Country: Japan.\nThe capital city is"       # receiver: Japan context (equal length)
    n_s = len(lm.tok(share_ctx, add_special_tokens=False)["input_ids"])
    n_r = len(lm.tok(recv_ctx, add_special_tokens=False)["input_ids"])
    if n_s != n_r:
        print(f"  [!] send/recv are not equal length ({n_s}!={n_r}) -> position alignment breaks. Adjust the wording or implement a TokenAligner.")

    share_kv, stok = lm.encode(share_ctx)
    recv_kv, rtok = lm.encode(recv_ctx)
    share_kv = IdentityTokenAligner().align(share_kv, stok, rtok)
    targets = [" Paris", " Tokyo"]                          # sharer-side = Paris / receiver-side = Tokyo
    cont = " "                                              # read the context's next-token distribution with a short continuation
    print(f"[real] self-C2C {model}: inject France-context KV into recv=Japan-context -> measure delta-logp")
    base = None
    for g in (0.0, 0.3, 0.6, 1.0):
        fuser.set_gate(g)
        fused = fuser.fuse(recv_kv, share_kv)
        lp = lm.forward_logprobs(fused, cont, targets)
        if g == 0.0:
            base = lp
        d = {t: round(lp[t] - base[t], 2) for t in targets}
        print(f"  gate={g}: logp={ {t: round(lp[t], 2) for t in targets} }  d={d}")
    print("  Expected (if transfer works): as the gate rises, delta-logp(' Paris')>0, delta-logp(' Tokyo')<0")
    print("  Note: a minimal design meeting the causal constraint and equal-length premise. Depends on the HF version's KV-injection behavior, so verify on real hardware.")


def real_validation_cross(thinker_model: str = "Qwen/Qwen3-0.6B",
                          worker_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct") -> None:
    """Qwen3-0.6B(Thinker) -> Qwen3-Coder(Worker). Exercises layer alignment + head_dim projection. Requires matching n_kv_heads."""
    from trinity.c2c import InProcessLM
    th, wk = InProcessLM(thinker_model), InProcessLM(worker_model)
    sh, rh = th.kv_shape(), wk.kv_shape()
    print(f"[real-cross] thinker {sh} / worker {rh}")
    if sh.n_heads != rh.n_heads:
        print(f"  [!] n_kv_heads mismatch ({sh.n_heads}!={rh.n_heads}) -> head projection is unimplemented, so fuser.init will fail."
              f"  First validate with self-C2C or an n_kv_heads-matching pair.")
        return
    fuser = C2CFuser.init(sh, rh, seed=0)   # terminal layer alignment + head_dim projection apply
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
