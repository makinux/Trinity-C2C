"""
P1: 異種fuserの学習統合（収束点）
=================================
これまでの部品を1つの学習可能 torch モジュールに統合:
  - 異種射影 W_k/W_v: [H_s·hd_s → H_r·hd_r]（GQA/ヘッド/次元の不一致を吸収・学習）
  - RoPE-aware K経路: unrotate(送信inv_freq・送信位置)→gather→射影→受信RoPE(受信inv_freq・受信位置)
  - V経路: gather→射影（回転なし）
  - 層ごと Gumbel-sigmoid ゲート
学習: 送信/受信モデルは凍結。受信に fused KV を注入し target の LM 損失を最小化 → fuser のみ更新。
  Codex指摘の通り、真の異種信頼性は「W がアーキ間の意味的KV対応を学習するか」に懸かる
  → 多様データ＋held-out で shuffled≫learned（share内容を本当に使う）を確認する。

検証:
  python -m trinity.c2c_fuser_hetero --selftest   # 合成: 異種形状＋RoPE-aware＋勾配貫通＋学習（DL不要）
  python -m trinity.c2c_fuser_hetero --real        # 実機 self-C2C 学習＋held-out汎化アブレーション
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
# torch RoPE（trinity_c2c_rope の numpy 版と同規約・微分可能）
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
# 統合モジュール: 異種 ＋ RoPE-aware ＋ 学習可能
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
        """recv_layers: list[(K,V)] 受信(凍結)。shareK_stored/shareV: list[K]/list[V] 送信(凍結, Kは回転済み)。
        返り値: list[(K,V)] 融合済み（fuserパラメータに対し微分可能）。"""
        g = self.gates()
        cs, ss = t_rope_cos_sin(sharer_positions, self.sh_inv)
        cr, sr = t_rope_cos_sin(receiver_positions, self.rc_inv)
        idx = torch.as_tensor(gather_idx, dtype=torch.long)
        out = []
        for i, (Kr, Vr) in enumerate(recv_layers):
            s = self.align.get(i)
            if s is None or str(i) not in self.Wk:
                out.append((Kr, Vr)); continue
            K_unrot = t_unapply_rope(shareK_stored[s], cs, ss)[:, idx, :]      # 未回転→受信位置へgather
            Kp = t_apply_rope(self._proj(K_unrot, self.Wk[str(i)]), cr, sr)    # 射影→受信RoPE
            Vp = self._proj(shareV[s][:, idx, :], self.Wv[str(i)])             # Vは回転なし
            L = min(Kr.shape[1], Kp.shape[1])
            gi = g[i]
            Kf = torch.cat([(1 - gi) * Kr[:, :L] + gi * Kp[:, :L], Kr[:, L:]], dim=1)
            Vf = torch.cat([(1 - gi) * Vr[:, :L] + gi * Vp[:, :L], Vr[:, L:]], dim=1)
            out.append((Kf, Vf))
        return out


# ============================================================
# selftest（合成・DL不要）: 異種形状＋RoPE-aware＋勾配＋学習
# ============================================================
def selftest():
    torch.manual_seed(0)
    sh, rh = KVShape(4, 4, 8), KVShape(3, 2, 16)                 # 送信/受信で 層数・ヘッド・次元すべて異なる
    sh_inv = rope_inv_freq(8, 10000.0)
    rc_inv = rope_inv_freq(16, 1e6)                              # baseも異なる
    fuser = TorchHeteroRoPEFuser(sh, rh, sh_inv, rc_inv, init_gate=0.05)

    Ls, Lr = 7, 5
    sp, rp, idx = list(range(Ls)), list(range(10, 10 + Lr)), list(range(Lr))   # 位置シフト＋gather
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

    assert fused[0][0].shape == (2, Lr, 16), fused[0][0].shape       # 受信形状(H2,hd16)を維持＝異種吸収
    assert losses[-1] < losses[0] * 0.6, f"{losses[0]:.3f}->{losses[-1]:.3f}"
    for nm, p in [("Wk2", fuser.Wk["2"]), ("Wv2", fuser.Wv["2"]), ("gate", fuser.gate_logit)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"grad missing: {nm}"
    print(f"[selftest] 異種形状(L4→3,H4→2,hd8→16,base違い)を吸収・受信形状維持 OK")
    print(f"[selftest] RoPE-aware経路＋勾配が Wk/Wv/gate に貫通 OK")
    print(f"[selftest] 学習で loss {losses[0]:.3f}->{losses[-1]:.3f} OK")
    print("[selftest] ALL PASSED")


# ============================================================
# real（実機 self-C2C 学習＋held-out汎化アブレーション）
# ============================================================
def real():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[real] load {name} (frozen) …")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv = inv_freq_from_model(model)
    shape = KVShape(cfg.num_hidden_layers, n_kv, hd)

    def enc(text):                                              # 凍結エンコード（定数）
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            pkv = model(input_ids=ids, use_cache=True).past_key_values
        K = [l.keys[0].detach() for l in pkv.layers]
        V = [l.values[0].detach() for l in pkv.layers]
        off = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        return K, V, ids, off

    RECV = "Country: Japan.\nThe capital city is"               # 受信は固定（neutral baseline）
    rK, rV, r_ids, r_off = enc(RECV)
    Lr = r_ids.shape[1]
    recv_layers = [(rK[l], rV[l]) for l in range(shape.n_layers)]

    def example(country):                                      # 送信文脈→(shareK,shareV,gather,pos)
        sK, sV, s_ids, s_off = enc(f"Country: {country}.\nThe capital city is")
        gidx = char_span_align(r_off, s_off)                   # 受信位置→送信位置
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
    print("[real] self-C2C 学習: {France,Germany,Italy}→首都 で fuser を学習 / Spain は held-out")
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

    # held-out 汎化アブレーション
    fuser.eval()
    sK, sV, sp, gidx, _ = data["Spain"]
    sK_F, sV_F, sp_F, gidx_F, _ = data["France"]
    with torch.no_grad():
        l_learn = loss_of(fuser, "Spain", " Madrid", sK, sV, sp, gidx).item()
        l_shuf = loss_of(fuser, "Spain", " Madrid", sK_F, sV_F, sp_F, gidx_F).item()   # 誤share(France)
        # gate0: 注入なし＝受信のみ
        for k in list(fuser.Wk): pass
        gl = fuser.gate_logit.detach().clone(); fuser.gate_logit.data.fill_(-50.0)
        l_gate0 = loss_of(fuser, "Spain", " Madrid", sK, sV, sp, gidx).item()
        fuser.gate_logit.data.copy_(gl)
    print(f"[held-out Spain→Madrid] -logp: learned(Spain)={l_learn:.2f}  gate0/no-inj={l_gate0:.2f}  shuffled(France)={l_shuf:.2f}")
    print("  learned≪gate0 → 注入が汎化。 learned≪shuffled → 未知国でも『正しいshare内容』を使用(=真の汎化)。")
    print("→ 異種fuser(射影W＋RoPE-aware＋ゲート)が凍結LM越しに学習され、held-outで効果を検証できる。")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
    else:
        selftest()
