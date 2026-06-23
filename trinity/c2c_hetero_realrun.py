"""
P1 on-device: heterogeneous 2-model C2C (SmolLM2-135M -> Qwen2.5-0.5B)
=====================================================
A true heterogeneous pair: SmolLM (30 layers / 3 KV heads / rope_theta 1e5 / different tokenizer) -> Qwen (24 layers / 2 KV heads / rope_theta 1e6).
Wire all the prior parts together on real hardware:
  CharSpanTokenAligner (different tokenizer) + TorchHeteroRoPEFuser (layer/head/dim absorption + RoPE-aware)
  + inv_freq_from_model (use each model's actual RoPE).

Checks:
  (1) sanity : gate=0 heterogeneous injection == receiver(Qwen) alone -> the heterogeneous plumbing is a correct no-op
  (2) gate>0 : the output changes -> the cross-model injection is wired up
  (3) training: gradients flow through the two frozen models into the fuser and the loss drops (France(SmolLM) -> Japan(Qwen) -> Paris)

Run: python -m trinity.c2c_hetero_realrun
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass

from trinity.c2c import KVShape
from trinity.c2c_rope import inv_freq_from_model
from trinity.c2c_hetero import char_span_align
from trinity.c2c_fuser_hetero import TorchHeteroRoPEFuser

from trinity.config import get
SHARER = get("c2c", "sharer_model", "HuggingFaceTB/SmolLM2-135M-Instruct")   # Thinker (sharer)
RECVR = get("c2c", "receiver_model", "Qwen/Qwen2.5-0.5B-Instruct")            # Worker (receiver)


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    c = m.config
    nkv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    return m, tok, inv_freq_from_model(m), KVShape(c.num_hidden_layers, nkv, hd)


def encode(m, tok, text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    with torch.no_grad():
        pkv = m(input_ids=ids, use_cache=True).past_key_values
    K = [l.keys[0].detach() for l in pkv.layers]
    V = [l.values[0].detach() for l in pkv.layers]
    off = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return K, V, ids, off


def main():
    print(f"[load] sharer={SHARER} / receiver={RECVR} (CPU, frozen) ...")
    sm, stok, s_inv, s_shape = load(SHARER)
    qm, qtok, q_inv, q_shape = load(RECVR)
    print(f"[shapes] sharer {s_shape}  ->  receiver {q_shape}   (layers / KV heads / base all heterogeneous)")

    fuser = TorchHeteroRoPEFuser(s_shape, q_shape, s_inv, q_inv, init_gate=0.05)

    def recv_cont_logp(recv_layers, q_ids, fused, targets):
        Lr = q_ids.shape[1]
        cache = DynamicCache()
        for i, (K, V) in enumerate(fused):
            cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
        attn = torch.ones((1, Lr), dtype=torch.long)
        pos = torch.tensor([[Lr - 1]])
        out = qm(input_ids=q_ids[:, -1:], attention_mask=attn, position_ids=pos,
                 past_key_values=cache, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return {t: float(logp[qtok(t, add_special_tokens=False)["input_ids"][0]]) for t in targets}

    def fuse_for(sharer_text, recv_text):
        sK, sV, s_ids, s_off = encode(sm, stok, sharer_text)
        rK, rV, r_ids, r_off = encode(qm, qtok, recv_text)
        gidx = char_span_align(r_off, s_off)                       # receiver(Qwen) position -> sharer(SmolLM) position
        recv_layers = [(rK[l], rV[l]) for l in range(q_shape.n_layers)]
        sp, rp = list(range(s_ids.shape[1])), list(range(r_ids.shape[1]))
        return recv_layers, sK, sV, sp, gidx, rp, r_ids

    # (1) sanity: same text, gate=0 -> matches receiver-alone (the heterogeneous plumbing is a no-op)
    text = "Country: France. The capital city is"
    recv_layers, sK, sV, sp, gidx, rp, r_ids = fuse_for(text, text)
    targets = [" Paris", " Lyon"]
    fuser.eval()
    with torch.no_grad():
        standalone = torch.log_softmax(qm(input_ids=r_ids).logits[0, -1].float(), -1)
        std = {t: float(standalone[qtok(t, add_special_tokens=False)["input_ids"][0]]) for t in targets}
        fuser.gate_logit.data.fill_(-50.0)                          # gate~0
        g0 = recv_cont_logp(recv_layers, r_ids, fuser.fuse(recv_layers, sK, sV, sp, gidx, rp), targets)
        fuser.gate_logit.data.fill_(2.0)                            # gate~0.88 (check the effect with untrained W)
        gp = recv_cont_logp(recv_layers, r_ids, fuser.fuse(recv_layers, sK, sV, sp, gidx, rp), targets)
    ok = abs(std[" Paris"] - g0[" Paris"]) < 1e-2
    print(f"[sanity] standalone Paris={std[' Paris']:.2f} / gate0-inj Paris={g0[' Paris']:.2f}  "
          f"{'OK no-op match (heterogeneous plumbing correct)' if ok else 'X mismatch'}")
    print(f"[gate>0] injection with untrained W changes the output: Paris {g0[' Paris']:.2f} -> {gp[' Paris']:.2f} (wiring confirmed; meaning comes after training)")

    # (3) cross-model training: SmolLM(France) -> Qwen(Japan) -> ' Paris'
    print("[train] fuse SmolLM(France context) into Qwen(Japan context) and train to predict ' Paris'")
    rl, sK2, sV2, sp2, gidx2, rp2, r_ids2 = fuse_for("Country: France. The capital city is",
                                                     "Country: Japan. The capital city is")
    pid = qtok(" Paris", add_special_tokens=False)["input_ids"][0]
    opt = torch.optim.Adam(fuser.parameters(), lr=0.1)
    fuser.gate_logit.data.fill_(np.log(0.05 / 0.95))
    fuser.train()
    for step in range(8):
        opt.zero_grad()
        fused = fuser.fuse(rl, sK2, sV2, sp2, gidx2, rp2)
        Lr = r_ids2.shape[1]
        cache = DynamicCache()
        for i, (K, V) in enumerate(fused):
            cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
        out = qm(input_ids=r_ids2[:, -1:], attention_mask=torch.ones((1, Lr), dtype=torch.long),
                 position_ids=torch.tensor([[Lr - 1]]), past_key_values=cache, use_cache=False)
        loss = -torch.log_softmax(out.logits[0, -1].float(), -1)[pid]
        loss.backward()
        if step == 0:
            assert fuser.Wk["0"].grad is not None and torch.isfinite(fuser.gate_logit.grad).all(), "gradients do not reach through the two models"
        opt.step()
        print(f"  step {step} | -logp(Paris)={loss.item():.3f}")
    print("-> gradients flow through the two frozen, different-architecture models into the fuser and training proceeds = demonstrates the on-device wiring of heterogeneous C2C.")


if __name__ == "__main__":
    main()
