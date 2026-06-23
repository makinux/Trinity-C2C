"""
P1: cache fuser training template
==========================
From identity projection + manual gate (numpy) to a trainable torch nn.Module:
  - per-layer head_dim projections W_k/W_v (learned)
  - per-layer gate (Gumbel-sigmoid@train / sigmoid@infer)
Training (C2C-style): the sharer/receiver models are frozen. Inject a "fused KV (differentiable)" into the
  receiver model and minimize the LM cross-entropy of the target continuation -> update only the fuser.
  (The frozen LM's weights need no grad, but the forward over the injected KV is differentiable, so gradients reach the fuser.)

Validation:
  python -m trinity.c2c_train_fuser --selftest   # torch only. validate autograd / optimization / gate learning (no model)
  python -m trinity.c2c_train_fuser --real        # on Qwen2.5-0.5B (CPU), demonstrate "gradients reach through the frozen LM"
"""
import sys
import math
import torch
import torch.nn as nn

from trinity.c2c import KVShape, terminal_alignment


# ============================================================
# Trainable cache fuser (torch)
# ============================================================
class TorchC2CFuser(nn.Module):
    def __init__(self, sharer: KVShape, receiver: KVShape, init_gate: float = 0.05, tau: float = 1.0):
        super().__init__()
        if sharer.n_heads != receiver.n_heads:
            raise ValueError("GQA head-count mismatch needs projection/mapping (same-family premise)")
        self.align = terminal_alignment(sharer.n_layers, receiver.n_layers)
        self.tau = tau
        hd_s, hd_r = sharer.head_dim, receiver.head_dim
        self.Wk, self.Wv = nn.ParameterDict(), nn.ParameterDict()
        for i in self.align:
            eye = torch.eye(hd_s, hd_r) if hd_s == hd_r else torch.randn(hd_s, hd_r) * (hd_s ** -0.5)
            self.Wk[str(i)] = nn.Parameter(eye.clone())
            self.Wv[str(i)] = nn.Parameter(eye.clone())
        self.gate_logit = nn.Parameter(torch.full((receiver.n_layers,), math.log(init_gate / (1 - init_gate))))

    def gates(self) -> torch.Tensor:
        if self.training:                                  # Gumbel-sigmoid (reparam, differentiable)
            u = torch.rand_like(self.gate_logit).clamp_(1e-6, 1 - 1e-6)
            noise = torch.log(u) - torch.log(1 - u)
            return torch.sigmoid((self.gate_logit + noise) / self.tau)
        return torch.sigmoid(self.gate_logit)

    def forward(self, recv_layers, share_layers):
        """recv/share: list[(K,V)], K/V=[heads,seq,hd] (constants from the frozen base) -> fused list (differentiable)."""
        g = self.gates()
        out = []
        for i, (Kr, Vr) in enumerate(recv_layers):
            s = self.align.get(i)
            if s is None or str(i) not in self.Wk:
                out.append((Kr, Vr)); continue
            Ks, Vs = share_layers[s]
            Ksp, Vsp = Ks @ self.Wk[str(i)], Vs @ self.Wv[str(i)]
            L = min(Kr.shape[1], Ksp.shape[1])
            gi = g[i]
            Kf = torch.cat([(1 - gi) * Kr[:, :L] + gi * Ksp[:, :L], Kr[:, L:]], dim=1)   # avoid in-place
            Vf = torch.cat([(1 - gi) * Vr[:, :L] + gi * Vsp[:, :L], Vr[:, L:]], dim=1)
            out.append((Kf, Vf))
        return out


# ============================================================
# On-device helpers (transformers)
# ============================================================
def _encode_layers(model, tok, text):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():                                  # the base is frozen = constant encoding
        pkv = model(input_ids=ids, use_cache=True).past_key_values
    return [(l.keys[0].detach(), l.values[0].detach()) for l in pkv.layers], ids


def _cache_from(fused, upto):
    from transformers import DynamicCache
    c = DynamicCache()
    for i, (K, V) in enumerate(fused):
        c.update(K[:, :upto, :].unsqueeze(0), V[:, :upto, :].unsqueeze(0), i)   # feed grad-bearing tensors
    return c


def neg_logp_target(model, tok, fused, recv_ids, target: str):
    """Drop the tail of fused (receiver-context KV), feed the receiver's last token live -> -logp of the target's first token (differentiable)."""
    L = fused[0][0].shape[1]
    cache = _cache_from(fused, L - 1)
    last = recv_ids[0, -1].view(1, 1)
    attn = torch.ones((1, L), dtype=torch.long)
    pos = torch.tensor([[L - 1]])
    out = model(input_ids=last, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=False)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    tgt = tok(target, add_special_tokens=False)["input_ids"][0]
    return -logp[tgt]


# ============================================================
# Training loop (on-device)
# ============================================================
def train_fuser(model, tok, fuser, triples, steps=30, lr=0.05, log=True):
    """triples: [(share_ctx, recv_ctx, target)]. Update only the fuser."""
    opt = torch.optim.Adam(fuser.parameters(), lr=lr)
    enc = {}
    for s, r, t in triples:
        if s not in enc: enc[s] = _encode_layers(model, tok, s)
        if r not in enc: enc[r] = _encode_layers(model, tok, r)
    fuser.train()
    for step in range(steps):
        opt.zero_grad()
        loss = 0.0
        for s, r, t in triples:
            (sh_l, _), (rc_l, rc_ids) = enc[s], enc[r]
            fused = fuser(rc_l, sh_l)
            loss = loss + neg_logp_target(model, tok, fused, rc_ids, t)
        loss = loss / len(triples)
        loss.backward()
        if step == 0:                                       # core: are gradients reaching through the frozen LM?
            g = fuser.gate_logit.grad
            assert g is not None and torch.isfinite(g).all(), "gradients are not reaching the fuser (injection path is broken)"
            assert fuser.Wk["0"].grad is not None, "gradients are not reaching the projection"
        gnorm = torch.nn.utils.clip_grad_norm_(fuser.parameters(), 10.0)
        opt.step()
        if log:
            print(f"  step {step:2d} | loss {loss.item():.3f} | grad {float(gnorm):.3f} | gate {torch.sigmoid(fuser.gate_logit).mean():.3f}")
    return fuser


# ============================================================
# selftest (torch only, no model): autograd / optimization / gate learning
# ============================================================
def selftest():
    torch.manual_seed(0)
    sh = KVShape(3, 2, 5); rh = KVShape(3, 2, 5)
    fuser = TorchC2CFuser(sh, rh, init_gate=0.05)
    recv = [(torch.randn(2, 6, 5), torch.randn(2, 6, 5)) for _ in range(3)]
    share = [(torch.randn(2, 6, 5), torch.randn(2, 6, 5)) for _ in range(3)]
    opt = torch.optim.Adam(fuser.parameters(), lr=0.1)
    g0 = torch.sigmoid(fuser.gate_logit).mean().item()
    fuser.train()
    losses = []
    for _ in range(200):
        fused = fuser(recv, share)                          # goal: fused matches share (needs gate->1, W->I)
        loss = sum(((Kf - Ks) ** 2).mean() + ((Vf - Vs) ** 2).mean()
                   for (Kf, Vf), (Ks, Vs) in zip(fused, share))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    g1 = torch.sigmoid(fuser.gate_logit).mean().item()
    assert losses[-1] < losses[0] * 0.5, f"loss not decreasing: {losses[0]:.3f}->{losses[-1]:.3f}"
    assert g1 > g0 + 0.1, f"gate didn't rise: {g0:.2f}->{g1:.2f}"
    assert fuser.Wk["2"].grad is not None, "grad missing on projection"
    print(f"[selftest] fuser trains : loss {losses[0]:.3f}->{losses[-1]:.3f} / gate {g0:.2f}->{g1:.2f} / grad->W OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real smoke (Qwen2.5-0.5B, CPU): do gradients reach the fuser through the frozen LM?
# ============================================================
def real_smoke(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[real] load {model_name} (CPU, eager, frozen) ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, attn_implementation="eager").eval()
    for p in model.parameters():
        p.requires_grad_(False)                             # freeze the base
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    shape = KVShape(cfg.num_hidden_layers, n_kv, head_dim)

    fuser = TorchC2CFuser(shape, shape, init_gate=0.05)
    triples = [("Country: France.\nThe capital city is",
                "Country: Japan.\nThe capital city is", " Paris")]
    print("[real] goal: train the fuser so that fusing France KV into the Japan context predicts ' Paris'")
    train_fuser(model, tok, fuser, triples, steps=12, lr=0.1)

    # ablation (Codex-recommended): separate the injection's effect from "whether it actually uses the share content vs memorizes"
    fuser.eval()
    sh_F, _ = _encode_layers(model, tok, "Country: France.\nThe capital city is")
    sh_G, _ = _encode_layers(model, tok, "Country: Germany.\nThe capital city is")
    rc_l, rc_ids = _encode_layers(model, tok, "Country: Japan.\nThe capital city is")
    with torch.no_grad():
        l_learn = neg_logp_target(model, tok, fuser(rc_l, sh_F), rc_ids, " Paris").item()
        l_gate0 = neg_logp_target(model, tok, rc_l, rc_ids, " Paris").item()              # no injection
        l_shuf = neg_logp_target(model, tok, fuser(rc_l, sh_G), rc_ids, " Paris").item()  # wrong share
    print(f"[ablation] -logp(Paris): learned(France)={l_learn:.2f}  gate0/no-inj={l_gate0:.2f}  shuffled(Germany)={l_shuf:.2f}")
    print("  learned << gate0 -> injection works. learned << shuffled -> actually uses the share content / if close, it memorized (= needs data augmentation).")
    print("-> passing the gradient assertion demonstrates 'gradients reach through the frozen LM and the fuser is trainable'.")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real_smoke()
    else:
        selftest()
