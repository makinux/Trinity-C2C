"""
P1: introduce C2C — replace the Thinker->Worker text hand-off with KV latent fusion (cache fuser)
==============================================================================
Minimally introduce a Cache-to-Cache (arXiv:2510.03215)-style cache fuser into this stack's star design.
In P0 the Worker received the Thinker's plan as "text". In P1 the Worker (=Receiver)'s KV cache is driven
by "latently fusing" the Thinker (=Sharer)'s KV into it (C2C on a single edge only).

Key assumptions:
  - C2C needs read/inject of each model's internal KV -> Thinker/Worker must run in-process (transformers),
    not over HTTP (vLLM) (the InProcessLM in this file). The Verifier can stay text-based.
  - The cache fuser is a "learned" projection + gate. Untrained (gate~0) it degrades to Worker-alone (no plan text).
    Also passing the plan text (plan_text) makes gate=0 strictly equivalent to P0 -> safe phased rollout. The more it learns, the more plan latents it injects.

[!] Essential blockers for heterogeneous models (GLM Thinker -> Qwen Worker) (mostly what the C2C paper solves; this skeleton flags them as unsolved):
  (1) GQA: if num_key_value_heads differ, C2CFuser.init hard-fails -> head projection/mapping is needed.
  (2) tokenizer alignment: KV at position i does not correspond in meaning across sharer/receiver -> a TokenAligner (decode -> re-encode with the other tokenizer) is required.
      This skeleton's fuse() is a minimal implementation that assumes "same tokenizer = position match".
  (3) KV-injection generation: alignment of transformers' Cache API / attention_mask / position (RoPE) is version-dependent and fragile (see generate_from_kv).

Validation:
  python -m trinity.c2c --selftest    # validate the fuser core (projection / layer alignment / gated fusion) in numpy (no torch)

Depends (production): pip install torch transformers   Note: the fusion core is numpy-only
"""
from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# KV representation (numpy form with batch=1 collapsed)
# ============================================================
@dataclass
class KVShape:
    n_layers: int
    n_heads: int
    head_dim: int


@dataclass
class KVCache:
    """layers[l] = (K, V), each [n_heads, seq, head_dim]"""
    layers: list[tuple[np.ndarray, np.ndarray]]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def copy(self) -> "KVCache":
        return KVCache([(K.copy(), V.copy()) for K, V in self.layers])


def from_hf_past(past) -> KVCache:
    """transformers' past_key_values (tuple[(K,V)], K/V=[batch,heads,seq,hd]) -> KVCache(batch=1)."""
    return KVCache([(np.asarray(K)[0], np.asarray(V)[0]) for (K, V) in past])


def to_hf_past(kv: KVCache):
    """KVCache -> transformers past_key_values form (restores the batch dim). Tensorize on the torch side to use."""
    return tuple((K[None, ...], V[None, ...]) for (K, V) in kv.layers)


# ============================================================
# Layer alignment (terminal alignment): map each receiver layer to the sharer, backward from the top (terminal)
# ============================================================
def terminal_alignment(n_sharer: int, n_receiver: int) -> dict[int, int]:
    """receiver layer i -> sharer layer j. Pair the terminals (last layers) first, then pair backward."""
    m: dict[int, int] = {}
    for k in range(min(n_sharer, n_receiver)):
        m[n_receiver - 1 - k] = n_sharer - 1 - k
    return m


class TokenAligner:
    """When the sharer/receiver tokenizers differ, KV at position i does not correspond semantically (Codex's top risk).
    C2C builds the position correspondence by 'decoding the receiver tokens -> re-encoding with the sharer tokenizer'.
    For heterogeneous models (GLM <-> Qwen), implement this class and reorder the sharer KV into the receiver's position sequence before fuse()."""
    def align(self, share_kv: "KVCache", share_tokens: list[int], recv_tokens: list[int]) -> "KVCache":
        raise NotImplementedError("position correspondence for heterogeneous tokenizers must be implemented (not needed for the same tokenizer)")


class IdentityTokenAligner(TokenAligner):
    """Assumes the same tokenizer (= the shared prefix matches by position). Does nothing."""
    def align(self, share_kv: "KVCache", share_tokens: list[int], recv_tokens: list[int]) -> "KVCache":
        return share_kv


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ============================================================
# cache fuser (the C2C core, numpy)
#   Trainable: per-layer projections W_k/W_v (head_dim matching) and a gate g (Gumbel-sigmoid@train / sigmoid@infer)
#   Fusion:   fused = (1-g)*receiver_KV + g*proj(sharer_KV)   Note: within the shared-prefix length L
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
        """Identity initialization (identity projection if head_dim matches, random otherwise). The gate is small = default leans toward the receiver."""
        if sharer.n_heads != receiver.n_heads:
            # Handling differing KV head counts (head projection/duplication) for GQA etc. is TODO. The skeleton assumes a match.
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
        lg = -np.inf if value <= 0 else (np.inf if value >= 1 else _logit(value))  # 0/1 exactly
        if layer is None:
            self.gate_logit[:] = lg
        else:
            self.gate_logit[layer] = lg

    def fuse(self, recv_kv: KVCache, share_kv: KVCache) -> KVCache:
        """Returns new KV with the sharer KV gate-fused into the receiver KV per layer. Fuses only the shared-prefix length L."""
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
            L = min(Kr.shape[1], Ksp.shape[1])  # shared-prefix length (overlap after token alignment)
            g = g_all[i]
            Kf, Vf = Kr.copy(), Vr.copy()
            Kf[:, :L] = (1 - g) * Kr[:, :L] + g * Ksp[:, :L]
            Vf[:, :L] = (1 - g) * Vr[:, :L] + g * Vsp[:, :L]
            out.append((Kf, Vf))
        return KVCache(out)


# ============================================================
# in-process LM (production: torch/transformers. KV read and injection generation)
#   Note: unused in selftest (no torch). Use it on a GPU machine.
# ============================================================
class InProcessLM:
    """An in-process model that can read/inject KV. In C2C, used for Thinker(Sharer)/Worker(Receiver)."""
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
        """Forward-compute text and return past_key_values (KVCache) and the token list (the Sharer-side context encoding)."""
        torch = self._torch
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        with torch.no_grad():
            out = self.model(**ids, use_cache=True)
        return from_hf_past(out.past_key_values), ids["input_ids"][0].tolist()

    def generate_from_kv(self, fused: KVCache, prompt: str, max_new_tokens: int = 1024) -> str:
        """Drive the Receiver's generation with the fused KV (injection generation).
        [!] Fragile spots (Codex): (a) some versions of the new Cache API do not accept a legacy tuple -> convert to DynamicCache,
        (b) attention_mask must cover the full past+prompt length, (c) position_ids/RoPE are offset by the past length.
        Behavior varies by transformers version, so verify on real hardware."""
        torch = self._torch
        from transformers import DynamicCache
        legacy = tuple((torch.tensor(K, device=self.device).unsqueeze(0),
                        torch.tensor(V, device=self.device).unsqueeze(0)) for (K, V) in fused.layers)
        cache = DynamicCache.from_legacy_cache(legacy)            # (a) convert to the new Cache API
        past_len = fused.layers[0][0].shape[1]
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        n_prompt = ids["input_ids"].shape[1]
        attn = torch.ones((1, past_len + n_prompt), device=self.device)   # (b) cover past+prompt
        pos = torch.arange(past_len, past_len + n_prompt, device=self.device).unsqueeze(0)  # (c) RoPE offset
        with torch.no_grad():
            out = self.model.generate(input_ids=ids["input_ids"], attention_mask=attn,
                                      position_ids=pos, past_key_values=cache,
                                      max_new_tokens=max_new_tokens, use_cache=True)
        return self.tok.decode(out[0][n_prompt:], skip_special_tokens=True)

    def forward_logprobs(self, fused: KVCache, query: str, targets: list[str]) -> dict[str, float]:
        """Run query one forward pass with the fused KV as past -> from the next-token distribution at the last position,
        return the logprob of each target's first token.
        The basic element for the quantitative C2C-transfer metric delta-logp(target) = logp_gate(target) - logp_gate0(target) (Codex-recommended)."""
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
# C2C version of the Thinker->Worker edge (the integration point replacing P0's text hand-off)
# ============================================================
def c2c_thinker_to_worker(thinker: InProcessLM, worker: InProcessLM, fuser: C2CFuser,
                          query: str, plan_instruction: str, worker_instruction: str,
                          aligner: TokenAligner = IdentityTokenAligner(),
                          plan_text: Optional[str] = None) -> str:
    """The Thinker->Worker C2C edge that replaces P0's text hand-off.
    1) Thinker(Sharer): encode query+plan instruction -> KV of the plan context
    2) Worker(Receiver): encode query -> receiver KV
    3) align the sharer KV to the receiver's position sequence with the aligner (for heterogeneous tokenizers) -> fuser.fuse -> fused KV
    4) Worker: generate seeded by the fused KV -> artifact
    Passing plan_text gives a "text plan + latent fusion" hybrid (gate=0 is strictly equivalent to P0 = safe phased rollout).
    """
    share_kv, share_tok = thinker.encode(f"{query}\n\n{plan_instruction}")
    recv_kv, recv_tok = worker.encode(query)
    share_kv = aligner.align(share_kv, share_tok, recv_tok)
    fused = fuser.fuse(recv_kv, share_kv)
    prompt = worker_instruction if plan_text is None else f"[Plan]\n{plan_text}\n\n{worker_instruction}"
    return worker.generate_from_kv(fused, prompt)


# ============================================================
# selftest (numpy: validate the fusion core. no torch/model)
# ============================================================
def _mock_kv(n_layers, n_heads, seq, head_dim, seed) -> KVCache:
    rng = np.random.default_rng(seed)
    return KVCache([(rng.standard_normal((n_heads, seq, head_dim)),
                     rng.standard_normal((n_heads, seq, head_dim))) for _ in range(n_layers)])


def selftest() -> None:
    # layer alignment: backward from the terminal
    assert terminal_alignment(4, 4) == {3: 3, 2: 2, 1: 1, 0: 0}
    assert terminal_alignment(6, 4) == {3: 5, 2: 4, 1: 3, 0: 2}      # 4 receiver layers <- the top 4 sharer layers
    print("[selftest] terminal_alignment OK")

    sh = KVShape(n_layers=4, n_heads=2, head_dim=5)
    rh = KVShape(n_layers=4, n_heads=2, head_dim=5)
    recv = _mock_kv(4, 2, 6, 5, seed=1)
    share = _mock_kv(4, 2, 6, 5, seed=2)
    fuser = C2CFuser.init(sh, rh, seed=0)

    # gate=0 -> exactly matches the receiver KV (= Worker alone, P0 degradation)
    fuser.set_gate(0.0)
    f0 = fuser.fuse(recv, share)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(f0.layers, recv.layers))
    print("[selftest] gate=0 -> receiver-only (graceful degradation to P0) OK")

    # gate=1 -> projected sharer KV (identity projection, so equals send within the shared length L)
    fuser.set_gate(1.0)
    f1 = fuser.fuse(recv, share)
    assert np.allclose(f1.layers[3][0], share.layers[3][0])
    print("[selftest] gate=1 -> projected sharer (identity projection) OK")

    # shape preservation
    assert all(Kf.shape == Kr.shape for (Kf, _), (Kr, _) in zip(f1.layers, recv.layers))
    print("[selftest] shape preserved OK")

    # head_dim mismatch -> random projection matches shapes (sharer hd=8 -> receiver hd=5)
    sh2 = KVShape(4, 2, 8)
    share2 = _mock_kv(4, 2, 6, 8, seed=3)
    fuser2 = C2CFuser.init(sh2, rh, seed=0)
    fuser2.set_gate(0.5)
    f2 = fuser2.fuse(recv, share2)
    assert f2.layers[0][0].shape == (2, 6, 5)
    print("[selftest] head_dim projection (8->5) OK")

    # shared prefix: when the sharer seq is short, fuse only the overlap, leaving the rest as the receiver's
    share_short = _mock_kv(4, 2, 3, 5, seed=4)
    fuser.set_gate(1.0)
    f3 = fuser.fuse(recv, share_short)
    assert np.array_equal(f3.layers[3][0][:, 3:], recv.layers[3][0][:, 3:])   # outside the overlap stays the receiver's
    print("[selftest] partial-prefix fusion OK")
    print("[selftest] ALL PASSED")


# ============================================================
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("C2C P1 skeleton. Production needs torch/transformers and in-process Thinker/Worker.")
        print("Validate the fusion core: python -m trinity.c2c --selftest")
