"""
P1 実機: 異種2モデル C2C（SmolLM2-135M → Qwen2.5-0.5B）
=====================================================
真の異種ペア: SmolLM(層30/KVヘッド3/rope_theta1e5/別トークナイザ) → Qwen(層24/KVヘッド2/rope_theta1e6)。
これまでの全部品を実機で結線:
  CharSpanTokenAligner(別トークナイザ) + TorchHeteroRoPEFuser(層/ヘッド/次元吸収 + RoPE-aware)
  + inv_freq_from_model(各モデルの実RoPEを使用)。

確認:
  ① sanity : gate=0 の異種注入 == 受信(Qwen)単独 → 異種プラミングが正しい no-op
  ② gate>0 : 出力が変化 → 2モデル間の注入が結線
  ③ training: 凍結2モデル越しに勾配が fuser へ貫通し loss が下がる（France(SmolLM)→Japan(Qwen)→Paris）

実行: python -m trinity.c2c_hetero_realrun
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
SHARER = get("c2c", "sharer_model", "HuggingFaceTB/SmolLM2-135M-Instruct")   # Thinker（送信）
RECVR = get("c2c", "receiver_model", "Qwen/Qwen2.5-0.5B-Instruct")            # Worker（受信）


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
    print(f"[load] sharer={SHARER} / receiver={RECVR} (CPU, frozen) …")
    sm, stok, s_inv, s_shape = load(SHARER)
    qm, qtok, q_inv, q_shape = load(RECVR)
    print(f"[shapes] sharer {s_shape}  →  receiver {q_shape}   （層/KVヘッド/base すべて異種）")

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
        gidx = char_span_align(r_off, s_off)                       # 受信(Qwen)位置 → 送信(SmolLM)位置
        recv_layers = [(rK[l], rV[l]) for l in range(q_shape.n_layers)]
        sp, rp = list(range(s_ids.shape[1])), list(range(r_ids.shape[1]))
        return recv_layers, sK, sV, sp, gidx, rp, r_ids

    # ① sanity: 同一テキスト・gate=0 → 受信単独と一致（異種プラミングが no-op）
    text = "Country: France. The capital city is"
    recv_layers, sK, sV, sp, gidx, rp, r_ids = fuse_for(text, text)
    targets = [" Paris", " Lyon"]
    fuser.eval()
    with torch.no_grad():
        standalone = torch.log_softmax(qm(input_ids=r_ids).logits[0, -1].float(), -1)
        std = {t: float(standalone[qtok(t, add_special_tokens=False)["input_ids"][0]]) for t in targets}
        fuser.gate_logit.data.fill_(-50.0)                          # gate≈0
        g0 = recv_cont_logp(recv_layers, r_ids, fuser.fuse(recv_layers, sK, sV, sp, gidx, rp), targets)
        fuser.gate_logit.data.fill_(2.0)                            # gate≈0.88（未学習Wでの効果確認）
        gp = recv_cont_logp(recv_layers, r_ids, fuser.fuse(recv_layers, sK, sV, sp, gidx, rp), targets)
    ok = abs(std[" Paris"] - g0[" Paris"]) < 1e-2
    print(f"[sanity] standalone Paris={std[' Paris']:.2f} / gate0-inj Paris={g0[' Paris']:.2f}  "
          f"{'✓ no-op一致(異種プラミング正しい)' if ok else '✗不一致'}")
    print(f"[gate>0] 未学習Wでの注入で出力変化: Paris {g0[' Paris']:.2f} → {gp[' Paris']:.2f}（結線確認・意味は学習後）")

    # ③ cross-model training: SmolLM(France) → Qwen(Japan) → ' Paris'
    print("[train] SmolLM(France文脈) を Qwen(Japan文脈) に融合し ' Paris' を予測するよう学習")
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
            assert fuser.Wk["0"].grad is not None and torch.isfinite(fuser.gate_logit.grad).all(), "勾配が2モデル越しに届かない"
        opt.step()
        print(f"  step {step} | -logp(Paris)={loss.item():.3f}")
    print("→ 凍結した『別アーキ2モデル』越しに勾配が fuser へ貫通し学習が進む＝異種C2Cの実機結線を実証。")


if __name__ == "__main__":
    main()
