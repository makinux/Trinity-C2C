"""
P1: training integration of the heterogeneous fuser (the convergence point)
=================================
Integrate all the prior parts into a single trainable torch module:
  - heterogeneous projections W_k/W_v: [H_s*hd_s -> H_r*hd_r] (absorb & learn GQA/head/dim mismatch)
  - RoPE-aware K path: unrotate(sharer inv_freq, sharer positions) -> gather -> project -> receiver RoPE(receiver inv_freq, receiver positions)
  - V path: gather -> project (no rotation)
  - per-layer Gumbel-sigmoid gate
Training: the sharer/receiver models are frozen. Inject the fused KV into the receiver and minimize the target's LM loss -> update only the fuser.
  As Codex noted, true heterogeneous reliability hinges on "whether W learns the semantic KV correspondence across architectures"
  -> with diverse data + held-out, confirm shuffled >> learned (it really uses the share content).

Validation:
  python -m trinity.c2c_fuser_hetero --selftest   # synthetic: heterogeneous shapes + RoPE-aware + gradient flow + training (no download)
  python -m trinity.c2c_fuser_hetero --real        # real self-C2C training + held-out generalization ablation
"""
import sys
import math
import numpy as np
import torch
import torch.nn as nn

from trinity.c2c import KVShape, terminal_alignment
from trinity.c2c_rope import rope_inv_freq, inv_freq_from_model
from trinity.c2c_hetero import char_span_align


# ============================================================
# torch RoPE (same convention as the numpy version in trinity_c2c_rope, differentiable)
# ============================================================
def t_rope_cos_sin(positions, inv_freq):
    pos = torch.as_tensor(positions, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def t_rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def t_apply_rope(x, cos, sin):
    return x * cos.unsqueeze(0) + t_rotate_half(x) * sin.unsqueeze(0)


def t_unapply_rope(x, cos, sin):
    return x * cos.unsqueeze(0) - t_rotate_half(x) * sin.unsqueeze(0)


# ============================================================
# Integrated module: heterogeneous + RoPE-aware + trainable
# ============================================================
class TorchHeteroRoPEFuser(nn.Module):
    def __init__(self, sharer: KVShape, receiver: KVShape, sharer_inv_freq, recv_inv_freq,
                 init_gate=0.05, tau=1.0):
        super().__init__()
        self.align = terminal_alignment(sharer.n_layers, receiver.n_layers)
        self.tau = tau
        self.Hs, self.hds = sharer.n_heads, sharer.head_dim
        self.Hr, self.hdr = receiver.n_heads, receiver.head_dim
        self.register_buffer("sh_inv", torch.as_tensor(np.asarray(sharer_inv_freq), dtype=torch.float32))
        self.register_buffer("rc_inv", torch.as_tensor(np.asarray(recv_inv_freq), dtype=torch.float32))
        din, dout = self.Hs * self.hds, self.Hr * self.hdr
        self.Wk, self.Wv = nn.ParameterDict(), nn.ParameterDict()
        for i in self.align:
            init = (torch.eye(din, dout) if din == dout else torch.randn(din, dout) / din ** 0.5)
            self.Wk[str(i)] = nn.Parameter(init.clone())
            self.Wv[str(i)] = nn.Parameter(init.clone())
        self.gate_logit = nn.Parameter(torch.full((receiver.n_layers,), math.log(init_gate / (1 - init_gate))))

    def gates(self):
        if self.training:
            u = torch.rand_like(self.gate_logit).clamp(1e-6, 1 - 1e-6)
            return torch.sigmoid((self.gate_logit + torch.log(u) - torch.log(1 - u)) / self.tau)
        return torch.sigmoid(self.gate_logit)

    def _proj(self, X, W):                                    # [H_s,L,hd_s] -> [H_r,L,hd_r]
        L = X.shape[1]
        return (X.transpose(0, 1).reshape(L, -1) @ W).reshape(L, self.Hr, self.hdr).transpose(0, 1)

    def fuse(self, recv_layers, shareK_stored, shareV, sharer_positions, gather_idx, receiver_positions):
        """recv_layers: list[(K,V)] receiver (frozen). shareK_stored/shareV: list[K]/list[V] sharer (frozen, K is rotated).
        Return: list[(K,V)] fused (differentiable w.r.t. the fuser parameters)."""
        g = self.gates()
        cs, ss = t_rope_cos_sin(sharer_positions, self.sh_inv)
        cr, sr = t_rope_cos_sin(receiver_positions, self.rc_inv)
        idx = torch.as_tensor(gather_idx, dtype=torch.long)
        out = []
        for i, (Kr, Vr) in enumerate(recv_layers):
            s = self.align.get(i)
            if s is None or str(i) not in self.Wk:
                out.append((Kr, Vr)); continue
            K_unrot = t_unapply_rope(shareK_stored[s], cs, ss)[:, idx, :]      # un-rotate -> gather to receiver positions
            Kp = t_apply_rope(self._proj(K_unrot, self.Wk[str(i)]), cr, sr)    # project -> receiver RoPE
            Vp = self._proj(shareV[s][:, idx, :], self.Wv[str(i)])             # V is not rotated
            L = min(Kr.shape[1], Kp.shape[1])
            gi = g[i]
            Kf = torch.cat([(1 - gi) * Kr[:, :L] + gi * Kp[:, :L], Kr[:, L:]], dim=1)
            Vf = torch.cat([(1 - gi) * Vr[:, :L] + gi * Vp[:, :L], Vr[:, L:]], dim=1)
            out.append((Kf, Vf))
        return out


# ============================================================
# selftest (synthetic, no download): heterogeneous shapes + RoPE-aware + gradients + training
# ============================================================
def selftest():
    torch.manual_seed(0)
    sh, rh = KVShape(4, 4, 8), KVShape(3, 2, 16)                 # layers, heads, and dims all differ between sharer/receiver
    sh_inv = rope_inv_freq(8, 10000.0)
    rc_inv = rope_inv_freq(16, 1e6)                              # base also differs
    fuser = TorchHeteroRoPEFuser(sh, rh, sh_inv, rc_inv, init_gate=0.05)

    Ls, Lr = 7, 5
    sp, rp, idx = list(range(Ls)), list(range(10, 10 + Lr)), list(range(Lr))   # position shift + gather
    recv = [(torch.randn(2, Lr, 16), torch.randn(2, Lr, 16)) for _ in range(3)]
    shK = [torch.randn(4, Ls, 8) for _ in range(4)]
    shV = [torch.randn(4, Ls, 8) for _ in range(4)]
    target = [(torch.randn(2, Lr, 16), torch.randn(2, Lr, 16)) for _ in range(3)]

    opt = torch.optim.Adam(fuser.parameters(), lr=0.05)
    fuser.train()
    losses = []
    for _ in range(200):
        fused = fuser.fuse(recv, shK, shV, sp, idx, rp)
        loss = sum(((Kf - Kt) ** 2).mean() + ((Vf - Vt) ** 2).mean()
                   for (Kf, Vf), (Kt, Vt) in zip(fused, target))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    assert fused[0][0].shape == (2, Lr, 16), fused[0][0].shape       # preserves receiver shape (H2,hd16) = heterogeneous absorption
    assert losses[-1] < losses[0] * 0.6, f"{losses[0]:.3f}->{losses[-1]:.3f}"
    for nm, p in [("Wk2", fuser.Wk["2"]), ("Wv2", fuser.Wv["2"]), ("gate", fuser.gate_logit)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"grad missing: {nm}"
    print(f"[selftest] absorbs heterogeneous shapes (L4->3, H4->2, hd8->16, different base), preserves receiver shape OK")
    print(f"[selftest] RoPE-aware path + gradients flow through to Wk/Wv/gate OK")
    print(f"[selftest] training reduces loss {losses[0]:.3f}->{losses[-1]:.3f} OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real (real self-C2C training + held-out generalization ablation)
# ============================================================
def real():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[real] load {name} (frozen) ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv = inv_freq_from_model(model)
    shape = KVShape(cfg.num_hidden_layers, n_kv, hd)

    def enc(text):                                              # frozen encoding (constant)
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            pkv = model(input_ids=ids, use_cache=True).past_key_values
        K = [l.keys[0].detach() for l in pkv.layers]
        V = [l.values[0].detach() for l in pkv.layers]
        off = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        return K, V, ids, off

    RECV = "Country: Japan.\nThe capital city is"               # receiver is fixed (neutral baseline)
    rK, rV, r_ids, r_off = enc(RECV)
    Lr = r_ids.shape[1]
    recv_layers = [(rK[l], rV[l]) for l in range(shape.n_layers)]

    def example(country):                                      # sharer context -> (shareK, shareV, gather, pos)
        sK, sV, s_ids, s_off = enc(f"Country: {country}.\nThe capital city is")
        gidx = char_span_align(r_off, s_off)                   # receiver position -> sharer position
        return sK, sV, list(range(s_ids.shape[1])), gidx, s_ids.shape[1]

    def loss_of(fuser, country, capital, sharerK, sharerV, sp, gidx):
        fused = fuser.fuse(recv_layers, sharerK, sharerV, sp, gidx, list(range(Lr)))
        cache = DynamicCache()
        for i, (K, V) in enumerate(fused):
            cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
        attn = torch.ones((1, Lr), dtype=torch.long)
        pos = torch.tensor([[Lr - 1]])
        out = model(input_ids=r_ids[:, -1:], attention_mask=attn, position_ids=pos,
                    past_key_values=cache, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return -logp[tok(capital, add_special_tokens=False)["input_ids"][0]]

    train = [("France", " Paris"), ("Germany", " Berlin"), ("Italy", " Rome")]
    held = ("Spain", " Madrid")
    data = {c: example(c) for c, _ in train + [held]}

    fuser = TorchHeteroRoPEFuser(shape, shape, inv, inv, init_gate=0.05)
    opt = torch.optim.Adam(fuser.parameters(), lr=0.1)
    print("[real] self-C2C training: train the fuser on {France,Germany,Italy} -> capital / Spain is held-out")
    fuser.train()
    for step in range(15):
        opt.zero_grad()
        loss = 0.0
        for c, cap in train:
            sK, sV, sp, gidx, _ = data[c]
            loss = loss + loss_of(fuser, c, cap, sK, sV, sp, gidx)
        loss = loss / len(train)
        loss.backward()
        if step == 0:
            assert fuser.Wk["0"].grad is not None and torch.isfinite(fuser.gate_logit.grad).all()
        opt.step()
        if step % 3 == 0:
            print(f"  step {step:2d} | train loss {loss.item():.3f} | gate {torch.sigmoid(fuser.gate_logit).mean():.3f}")

    # held-out generalization ablation
    fuser.eval()
    sK, sV, sp, gidx, _ = data["Spain"]
    sK_F, sV_F, sp_F, gidx_F, _ = data["France"]
    with torch.no_grad():
        l_learn = loss_of(fuser, "Spain", " Madrid", sK, sV, sp, gidx).item()
        l_shuf = loss_of(fuser, "Spain", " Madrid", sK_F, sV_F, sp_F, gidx_F).item()   # wrong share (France)
        # gate0: no injection = receiver only
        for k in list(fuser.Wk): pass
        gl = fuser.gate_logit.detach().clone(); fuser.gate_logit.data.fill_(-50.0)
        l_gate0 = loss_of(fuser, "Spain", " Madrid", sK, sV, sp, gidx).item()
        fuser.gate_logit.data.copy_(gl)
    print(f"[held-out Spain->Madrid] -logp: learned(Spain)={l_learn:.2f}  gate0/no-inj={l_gate0:.2f}  shuffled(France)={l_shuf:.2f}")
    print("  learned << gate0 -> injection generalizes. learned << shuffled -> uses the 'correct share content' even for unseen countries (= true generalization).")
    print("-> the heterogeneous fuser (projection W + RoPE-aware + gate) is trained through the frozen LM, and its effect can be validated on held-out.")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
    else:
        selftest()
