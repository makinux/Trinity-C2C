"""
trinity/c2c_edge.py — reusable heterogeneous C2C edge + generation from fused KV
================================================================================
The standalone scripts (``c2c_hetero_realrun``) proved the heterogeneous *fuse* and a
*single-token* continuation logprob. To drive the gateway's Worker turn we need the missing
piece: **multi-token generation seeded by a heterogeneous fused KV**, packaged as a reusable
engine so the orchestration (``trinity.p1``) and the gateway can call one function.

Design:
  - ``C2CEngine`` loads Sharer(Thinker) + Receiver(Worker) **frozen / CPU / fp32 / eager**
    (heterogeneous by default: SmolLM2-135M -> Qwen2.5-0.5B) and builds one
    ``TorchHeteroRoPEFuser`` (layer/head/dim absorption + RoPE-aware alignment).
  - ``c2c_edge(share_text, recv_text, gen_prompt)`` = encode x2 -> ``char_span_align`` ->
    ``fuser.fuse`` -> ``_generate_from_fused``. Contexts are all arguments (so the REVISE
    loop can re-fuse from a fresh canonical text state — see ``trinity.p1``).
  - ``_generate_from_fused`` builds a ``DynamicCache`` via ``.update()`` (the transformers-5
    convention verified in ``c2c_realrun.build_cache`` — NOT ``from_legacy_cache``) and then
    runs a **manual greedy loop** that reuses the exact forward call shape proven correct by
    ``c2c_hetero_realrun``'s sanity check, rather than the fragile ``model.generate`` +
    pre-filled-cache path. This makes the new generation step verifiable by the gate-0 invariant.

Verify (real, uses the small cached models; tens of seconds on CPU):
  python -m trinity.c2c_edge          # gate~0 generation == receiver-alone (invariant); gate>0 changes output
"""
from __future__ import annotations

import math
import os
from typing import Optional

try:
    import truststore; truststore.inject_into_ssl()   # corporate SSL-inspection workaround for HF downloads
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from trinity.c2c import KVShape
from trinity.c2c_rope import inv_freq_from_model
from trinity.c2c_hetero import char_span_align
from trinity.c2c_fuser_hetero import TorchHeteroRoPEFuser, load_fuser_into, build_fuser_for_checkpoint
from trinity.config import get
from trinity.p0 import strip_think


# ---------------------------------------------------------------------------
# load / encode helpers (mirroring trinity.c2c_hetero_realrun, parameterized by device/dtype)
# ---------------------------------------------------------------------------
def _load(name: str, device: str, dtype):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="eager").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    c = model.config
    n_kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    return model, tok, inv_freq_from_model(model), KVShape(c.num_hidden_layers, n_kv, hd)


def _encode(model, tok, text: str, device: str, max_len: int = 4096):
    """Forward-compute ``text`` -> (per-layer K, per-layer V, input_ids, char offsets).

    Both tokenizer calls use identical truncation so input_ids and offsets stay length-matched;
    the cap is a safety net against pathological input (the transcript is already length-managed)
    and bounds how long a single forward can hold the shared C2C lock."""
    enc = tok(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        pkv = model(input_ids=ids, use_cache=True).past_key_values
    K = [l.keys[0].detach() for l in pkv.layers]      # [H, L, hd] per layer (batch dim dropped)
    V = [l.values[0].detach() for l in pkv.layers]
    off = tok(text, return_offsets_mapping=True, add_special_tokens=False,
              truncation=True, max_length=max_len)["offset_mapping"]
    return K, V, ids, off


def _gate_to_logit(value: float) -> float:
    if value <= 0:
        return -50.0          # sigmoid(-50) ~ 2e-22 == 0 in fp32 (saturate, not exactly -inf)
    if value >= 1:
        return 50.0
    return math.log(value / (1.0 - value))


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
class C2CEngine:
    """Heterogeneous C2C engine: Sharer(Thinker) -> [KV fusion] -> Receiver(Worker).

    Heavy (loads two models). The gateway holds **one** cached instance behind a lock; never
    build one per request. ``device`` defaults to CPU so it runs with no GPU.
    """

    def __init__(self, sharer_model: Optional[str] = None, receiver_model: Optional[str] = None, *,
                 device: str = "cpu", dtype=torch.float32,
                 init_gate: Optional[float] = None, tau: Optional[float] = None,
                 max_new_tokens: int = 256, max_ctx: int = 4096,
                 fuser_path: Optional[str] = None):
        # precedence: explicit arg > env override > config.yml > built-in default
        self.sharer_name = (sharer_model or os.getenv("SHARER_MODEL_ID")
                            or get("c2c", "sharer_model", "HuggingFaceTB/SmolLM2-135M-Instruct"))
        self.receiver_name = (receiver_model or os.getenv("RECEIVER_MODEL_ID")
                              or get("c2c", "receiver_model", "Qwen/Qwen2.5-0.5B-Instruct"))
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.max_ctx = int(max_ctx)
        gate_explicit = init_gate is not None      # did the caller pin a uniform gate?
        gate = float(get("c2c", "init_gate", 0.05) if init_gate is None else init_gate)
        tau = float(get("c2c", "tau", 1.0) if tau is None else tau)

        self.sm, self.stok, s_inv, self.s_shape = _load(self.sharer_name, device, dtype)
        self.rm, self.rtok, r_inv, self.r_shape = _load(self.receiver_name, device, dtype)
        # Optional trained fuser checkpoint. Resolution: arg > TRINITY_C2C_FUSER > config c2c.fuser_path.
        self.fuser_path = fuser_path or os.getenv("TRINITY_C2C_FUSER") or get("c2c", "fuser_path", None) or None
        # Construct with a clamped init_gate — the fuser ctor does math.log(g/(1-g)), which blows up at
        # exactly 0 or 1 (a documented TRINITY_C2C_GATE=0 would crash) — then apply the requested gate
        # through the saturating set_gate() below. When a checkpoint is set we build the CLASS/width it
        # records (linear or MLP) so the strict load matches; reading it never loads weights or
        # validates compatibility (load_fuser_into does that, with a safe fallback).
        clamped = min(max(gate, 1e-3), 1 - 1e-3)
        if self.fuser_path:
            try:
                self.fuser = build_fuser_for_checkpoint(self.fuser_path, self.s_shape, self.r_shape,
                                                        s_inv, r_inv, init_gate=clamped, tau=tau).to(device)
            except Exception as e:
                print(f"[C2CEngine] WARNING: could not read fuser checkpoint {self.fuser_path!r}: {e}"
                      f" -- using a default linear fuser.")
                self.fuser = TorchHeteroRoPEFuser(self.s_shape, self.r_shape, s_inv, r_inv,
                                                  init_gate=clamped, tau=tau).to(device)
        else:
            self.fuser = TorchHeteroRoPEFuser(self.s_shape, self.r_shape, s_inv, r_inv,
                                              init_gate=clamped, tau=tau).to(device)
        self.fuser.eval()                 # eval => deterministic sigmoid gate (no Gumbel noise)
        self._recv_eos = {i for i in [self.rtok.eos_token_id] if i is not None}
        self.set_gate(gate)               # applies exact 0/1 via saturation; also sets self.gate

        # On any incompatibility we keep the fresh (untrained) fuser, which is gate-0 safe.
        self.fuser_loaded = False
        self.fuser_label = "untrained"
        if self.fuser_path:
            try:
                load_fuser_into(self.fuser, self.fuser_path,
                                expect_sharer_model=self.sharer_name,
                                expect_receiver_model=self.receiver_name)
                self.fuser.eval()
                self.fuser_loaded = True
                arch = type(self.fuser).arch_name.replace("_rope_v1", "")
                self.fuser_label = f"trained:{os.path.basename(self.fuser_path)} [{arch}]"
                if gate_explicit:
                    self.set_gate(gate)   # an explicit gate overrides the checkpoint's learned gates
                else:
                    # keep the checkpoint's learned per-layer gates; report their mean for the trace
                    self.gate = float(torch.sigmoid(self.fuser.gate_logit).mean().detach())
            except Exception as e:
                # The user asked for a trained fuser and it failed to load — fall back to gate=0
                # (receiver-alone, no fusion), which is genuinely non-degrading, rather than an
                # untrained fuser at a nonzero gate (which would degrade).
                self.set_gate(0.0)
                print(f"[C2CEngine] WARNING: could not load fuser checkpoint {self.fuser_path!r}: {e}"
                      f" -- falling back to gate=0 (receiver-alone, no fusion).")

    # -- gate control ------------------------------------------------------
    def set_gate(self, value: float) -> None:
        self.fuser.gate_logit.data.fill_(_gate_to_logit(float(value)))
        self.gate = float(value)

    # -- plain text chat (used by the Verifier and the Thinker's text plan) -
    def _chat(self, model, tok, system: str, user: str, max_new_tokens: int) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # transformers 5.x apply_chat_template returns a BatchEncoding (dict) — pass it through
        # with ** so generate also gets the attention_mask (avoids the unset-mask warning).
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        return strip_think(tok.decode(out[0][input_len:], skip_special_tokens=True)).strip()

    def receiver_chat(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        return self._chat(self.rm, self.rtok, system, user, max_new_tokens or self.max_new_tokens)

    def sharer_chat(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        return self._chat(self.sm, self.stok, system, user, max_new_tokens or self.max_new_tokens)

    # -- the C2C edge ------------------------------------------------------
    def _build_cache(self, fused_layers) -> DynamicCache:
        """fused list[(K,V)] with K/V = [H, L, hd] -> transformers-5 DynamicCache (batch dim restored)."""
        cache = DynamicCache()
        for i, (K, V) in enumerate(fused_layers):
            cache.update(K.unsqueeze(0).contiguous(), V.unsqueeze(0).contiguous(), i)
        return cache

    def _greedy_from_cache(self, cache: DynamicCache, past_len: int, prompt_ids: torch.Tensor,
                           max_new_tokens: int, should_stop=None) -> list[int]:
        """Greedy-decode from a pre-seeded KV cache.

        Reuses the exact forward shape verified by c2c_hetero_realrun (full-ones 2D attention
        mask + explicit position_ids offset by the cached length + in-place DynamicCache), just
        iterated — deliberately avoiding model.generate()'s pre-filled-cache path (version-fragile).
        ``should_stop`` (optional predicate) lets a disconnected request abort between steps so the
        shared C2C lock is released promptly.
        """
        if max_new_tokens <= 0:
            return []
        n = prompt_ids.shape[1]
        cur = past_len                                       # tokens already in the cache
        # prefill the whole gen prompt in one forward
        attn = torch.ones((1, cur + n), dtype=torch.long, device=self.device)
        pos = torch.arange(cur, cur + n, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.rm(input_ids=prompt_ids, attention_mask=attn, position_ids=pos,
                          past_key_values=cache, use_cache=True)
        cur += n
        nxt = int(out.logits[0, -1].argmax())
        gen = [nxt]
        for _ in range(max_new_tokens - 1):
            if nxt in self._recv_eos or (should_stop is not None and should_stop()):
                break
            attn = torch.ones((1, cur + 1), dtype=torch.long, device=self.device)
            pos = torch.tensor([[cur]], device=self.device)
            with torch.no_grad():
                out = self.rm(input_ids=torch.tensor([[nxt]], device=self.device),
                              attention_mask=attn, position_ids=pos,
                              past_key_values=cache, use_cache=True)
            cur += 1
            nxt = int(out.logits[0, -1].argmax())
            gen.append(nxt)
        return gen

    def c2c_edge(self, share_text: str, recv_text: str, gen_prompt: str, *,
                 gate: Optional[float] = None, max_new_tokens: Optional[int] = None,
                 should_stop=None) -> tuple[str, dict]:
        """Thinker(share_text) --[KV fusion]--> Worker(recv_text) then generate from ``gen_prompt``.

        Returns (artifact_text, meta) where ``meta`` is a plain dict (the ``fusion`` event
        payload — kept torch-free so a mock engine can produce the same shape). The fused
        receiver KV (length = recv tokens) is the generation prefix; ``gen_prompt`` is appended
        live and the Worker continues from there.
        """
        # A per-call gate is a TEMPORARY override: save & restore gate_logit / self.gate so it never
        # clobbers the engine's persistent (e.g. checkpoint-learned per-layer) gates — this engine is
        # a shared, lock-serialized singleton, so a leaked gate would corrupt later requests.
        saved = None
        if gate is not None:
            saved = (self.fuser.gate_logit.detach().clone(), self.gate)
            self.set_gate(gate)
        self.fuser.eval()
        mnt = int(max_new_tokens or self.max_new_tokens)
        try:
            sK, sV, s_ids, s_off = _encode(self.sm, self.stok, share_text, self.device, self.max_ctx)
            rK, rV, r_ids, r_off = _encode(self.rm, self.rtok, recv_text, self.device, self.max_ctx)
            gidx = char_span_align(r_off, s_off)             # receiver position -> sharer position
            recv_layers = [(rK[l], rV[l]) for l in range(self.r_shape.n_layers)]
            sp = list(range(s_ids.shape[1]))
            rp = list(range(r_ids.shape[1]))
            with torch.no_grad():
                fused = self.fuser.fuse(recv_layers, sK, sV, sp, gidx, rp)

            past_len = fused[0][0].shape[1]                  # = receiver length
            cache = self._build_cache(fused)
            prompt_ids = self.rtok(gen_prompt, return_tensors="pt",
                                   add_special_tokens=False)["input_ids"].to(self.device)
            gen_ids = self._greedy_from_cache(cache, past_len, prompt_ids, mnt, should_stop=should_stop)
            text = self.rtok.decode(gen_ids, skip_special_tokens=True).strip()
            meta = {
                "gate": round(float(self.gate), 4), "aligned_layers": len(self.fuser.align),
                "share_len": int(s_ids.shape[1]), "recv_len": int(r_ids.shape[1]),
                "sharer_model": self.sharer_name, "receiver_model": self.receiver_name,
                "new_tokens": len(gen_ids), "fuser": self.fuser_label,
            }
            return text, meta
        finally:
            if saved is not None:
                with torch.no_grad():
                    self.fuser.gate_logit.copy_(saved[0])
                self.gate = saved[1]

    # -- receiver-alone generation (the gate-0 reference / no-fusion baseline) --
    def receiver_only(self, recv_text: str, gen_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Worker generating from recv_text+gen_prompt with NO fusion — same token path as a
        gate~0 c2c_edge (identical tokenization), so the two must agree. The gate-0 invariant."""
        mnt = int(max_new_tokens or self.max_new_tokens)
        rK, rV, r_ids, _ = _encode(self.rm, self.rtok, recv_text, self.device, self.max_ctx)
        recv_layers = [(rK[l], rV[l]) for l in range(self.r_shape.n_layers)]
        cache = self._build_cache(recv_layers)
        prompt_ids = self.rtok(gen_prompt, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(self.device)
        gen_ids = self._greedy_from_cache(cache, r_ids.shape[1], prompt_ids, mnt)
        return self.rtok.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# real smoke test (the P1a gate): gate~0 generation == receiver-alone; gate>0 changes output
# ---------------------------------------------------------------------------
def smoke() -> None:
    print("[smoke] loading heterogeneous engine (SmolLM2-135M -> Qwen2.5-0.5B, CPU) ...")
    eng = C2CEngine(max_new_tokens=24)
    print(f"[smoke] shapes: sharer {eng.s_shape} -> receiver {eng.r_shape}")

    share = "Country: France.\nThe capital city is Paris, a major European capital."
    recv = "Country: Japan.\nThe capital city is"
    gen_prompt = " "                                          # minimal continuation prompt

    # gate ~ 0 : fused == receiver KV -> generation must equal the no-fusion receiver baseline
    eng.set_gate(0.0)
    g0, meta0 = eng.c2c_edge(share, recv, gen_prompt)
    base = eng.receiver_only(recv, gen_prompt)
    match = (g0 == base)
    print(f"[smoke] gate=0  edge={g0!r}")
    print(f"[smoke] gate=0  base={base!r}")
    print(f"[gate-0 invariant] fused-generation == receiver-alone : {'OK' if match else 'X MISMATCH'}")
    assert match, "gate-0 fused generation diverged from the receiver-alone baseline"

    # gate > 0 : the injected (untrained) sharer KV must change the continuation (wiring live)
    eng.set_gate(0.8)
    g8, meta8 = eng.c2c_edge(share, recv, gen_prompt)
    print(f"[smoke] gate=0.8 edge={g8!r}  (aligned_layers={meta8['aligned_layers']}, "
          f"share_len={meta8['share_len']}, recv_len={meta8['recv_len']})")
    print(f"[gate>0 wiring] output differs from gate=0 : {'OK' if g8 != g0 else 'X no change'}")
    assert g8 != g0, "gate>0 injection did not change the output (fusion not wired into generation)"
    print("[smoke] ALL PASSED -- heterogeneous fused-KV generation works through the engine.")


if __name__ == "__main__":
    smoke()
