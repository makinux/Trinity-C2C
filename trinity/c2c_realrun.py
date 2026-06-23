"""
P1(a) on-device: confirm the C2C transfer signal with an SLM
=============================================
self-C2C: fuse share='France context' KV into recv='Japan context' and measure the next token's
delta-logp(' Paris') / delta-logp(' Tokyo') per gate. If, as the gate rises, dParis>0 & dTokyo<0, there is transfer.

Robustness:
  - The continuation is computed as "the last 1 token of the context KV as live input, the remaining (L-1) as the fused past" (textbook KV-cache behavior).
    -> gate=0 should match a normal forward pass (confirmed by a sanity check).
  - Corporate SSL: use the OS certificate store via truststore to get HF downloads through.

Run: python -m trinity.c2c_realrun        (the C2C_MODEL env var can change the model)
"""
import os
import numpy as np

try:
    import truststore; truststore.inject_into_ssl()   # workaround for corporate SSL-inspection environments
except Exception as e:
    print("[warn] truststore unavailable:", e)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from trinity.c2c import KVShape, KVCache, C2CFuser, IdentityTokenAligner

from trinity.config import get
MODEL = os.environ.get("C2C_MODEL") or get("c2c", "receiver_model", "Qwen/Qwen2.5-0.5B-Instruct")
torch.manual_seed(0)


def kv_slice(kv: KVCache, upto: int) -> KVCache:
    return KVCache([(K[:, :upto, :], V[:, :upto, :]) for (K, V) in kv.layers])


def extract_kv(pkv) -> KVCache:
    """transformers 5.x DynamicCache -> KVCache (drop the batch dim)."""
    return KVCache([(lyr.keys.detach().cpu().numpy()[0], lyr.values.detach().cpu().numpy()[0])
                    for lyr in pkv.layers])


def build_cache(kv: KVCache) -> DynamicCache:
    """KVCache -> transformers 5.x DynamicCache (feed each layer via update(); from_legacy_cache is removed)."""
    c = DynamicCache()
    for i, (K, V) in enumerate(kv.layers):
        c.update(torch.tensor(K, dtype=torch.float32).unsqueeze(0),
                 torch.tensor(V, dtype=torch.float32).unsqueeze(0), i)
    return c


def main():
    print(f"[load] {MODEL} (CPU, eager) ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    shape = KVShape(cfg.num_hidden_layers, n_kv, head_dim)
    print(f"[load] KV shape = {shape}")

    def tid(s):  # target's first token ID
        return tok(s, add_special_tokens=False)["input_ids"][0]

    def encode(text):
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        return extract_kv(out.past_key_values), ids[0].tolist()

    def cont_logprobs(fused_full: KVCache, last_token_id: int, targets):
        """Drop the tail of fused (length L) so past=(L-1), feed last_token live -> next-token logp."""
        L = fused_full.layers[0][0].shape[1]
        cache = build_cache(kv_slice(fused_full, L - 1))
        inp = torch.tensor([[last_token_id]])
        attn = torch.ones((1, L), dtype=torch.long)               # past(L-1)+1
        pos = torch.tensor([[L - 1]])
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=attn, position_ids=pos,
                        past_key_values=cache, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return {t: float(logp[tid(t)]) for t in targets}

    recv_ctx = "Country: Japan.\nThe capital city is"
    share_ctx = "Country: France.\nThe capital city is"
    targets = [" Paris", " Tokyo"]

    recv_kv, recv_ids = encode(recv_ctx)
    share_kv, share_ids = encode(share_ctx)
    print(f"[ctx] len(recv)={len(recv_ids)} len(share)={len(share_ids)} "
          f"({'equal-length OK' if len(recv_ids)==len(share_ids) else '[!] not equal length (position alignment broken)'})")
    share_kv = IdentityTokenAligner().align(share_kv, share_ids, recv_ids)
    fuser = C2CFuser.init(shape, shape, seed=0)

    # --- sanity: gate=0 injection == normal forward's next-token distribution ---
    with torch.no_grad():
        ln = torch.log_softmax(model(input_ids=torch.tensor([recv_ids])).logits[0, -1].float(), dim=-1)
    normal = {t: float(ln[tid(t)]) for t in targets}
    fuser.set_gate(0.0)
    inj0 = cont_logprobs(fuser.fuse(recv_kv, share_kv), recv_ids[-1], targets)
    ok = all(abs(normal[t] - inj0[t]) < 1e-2 for t in targets)
    print(f"[sanity] normal={{P:{normal[' Paris']:.2f},T:{normal[' Tokyo']:.2f}}} "
          f"gate0-inj={{P:{inj0[' Paris']:.2f},T:{inj0[' Tokyo']:.2f}}}  "
          f"{'OK match (KV injection is correct)' if ok else 'X mismatch -> injection-path inconsistency'}")

    # --- transfer sweep ---
    print("[transfer] inject France-context KV into recv=Japan-context:")
    base = None
    for g in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        fuser.set_gate(g)
        lp = cont_logprobs(fuser.fuse(recv_kv, share_kv), recv_ids[-1], targets)
        if g == 0.0:
            base = lp
        dP, dT = lp[" Paris"] - base[" Paris"], lp[" Tokyo"] - base[" Tokyo"]
        print(f"  gate={g:<4} logp(Paris)={lp[' Paris']:6.2f} logp(Tokyo)={lp[' Tokyo']:6.2f} "
              f"| dParis={dP:+.2f} dTokyo={dT:+.2f}")
    print("\nVerdict: if dParis>0 and dTokyo<0 as the gate rises, there is a 'C2C transfer signal'.")


if __name__ == "__main__":
    main()
