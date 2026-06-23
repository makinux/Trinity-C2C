"""
P1: データ規模学習で「汎化」を狙う（最後のフロンティア）
=====================================================
これまで単一例では fuser が記憶するだけで held-out 汎化しなかった。Codexのレシピで再挑戦:
  - 関係データ(国→首都) ~40件、train/held-out 分割
  - 受信は固定プレースホルダ "Country: Unknown..."、送信が実国 → fuser が「国の同一性」を注入
  - contrastive: 正しい送信は誤った送信より target を高く（share内容を使うよう強制）
  - 正則化: 射影Wを恒等近傍にL2（記憶容量を抑制）
判定: held-out で mean(learned) < mean(gate0) なら『注入が未知国でも効く＝真の汎化』。
      mean(learned) < mean(shuffled) なら『正しいshare内容を使用』。

効率: 凍結エンコード(送信/受信KV)は1回だけ計算してキャッシュ。学習は継続1トークンの forward/backward のみ。

実行: python -m trinity.c2c_train_general
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

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
torch.manual_seed(0)

PAIRS = [
    ("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Spain", " Madrid"),
    ("Germany", " Berlin"), ("Russia", " Moscow"), ("China", " Beijing"), ("Egypt", " Cairo"),
    ("Greece", " Athens"), ("Portugal", " Lisbon"), ("Austria", " Vienna"), ("Poland", " Warsaw"),
    ("Cuba", " Havana"), ("Norway", " Oslo"), ("Sweden", " Stockholm"), ("Finland", " Helsinki"),
    ("Ireland", " Dublin"), ("Turkey", " Ankara"), ("Iran", " Tehran"), ("Thailand", " Bangkok"),
    ("Peru", " Lima"), ("Chile", " Santiago"), ("Kenya", " Nairobi"), ("Hungary", " Budapest"),
    ("Belgium", " Brussels"), ("Netherlands", " Amsterdam"), ("Iraq", " Baghdad"), ("Vietnam", " Hanoi"),
    ("Denmark", " Copenhagen"), ("Morocco", " Rabat"), ("Lebanon", " Beirut"), ("Jordan", " Amman"),
    ("Ukraine", " Kyiv"), ("Romania", " Bucharest"), ("Bulgaria", " Sofia"), ("Croatia", " Zagreb"),
    ("Serbia", " Belgrade"), ("Iceland", " Reykjavik"), ("Qatar", " Doha"), ("Afghanistan", " Kabul"),
]
RECV = "Country: Unknown.\nThe capital city is"


def main():
    print(f"[load] {MODEL} (frozen) …")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    inv = inv_freq_from_model(model)
    shape = KVShape(cfg.num_hidden_layers, nkv, hd)

    def encode(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            pkv = model(input_ids=ids, use_cache=True).past_key_values
        K = [l.keys[0].detach() for l in pkv.layers]
        V = [l.values[0].detach() for l in pkv.layers]
        off = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        return K, V, ids, off

    rK, rV, r_ids, r_off = encode(RECV)
    Lr = r_ids.shape[1]
    recv_layers = [(rK[l], rV[l]) for l in range(shape.n_layers)]
    rp = list(range(Lr))

    # 各国の送信エンコードをキャッシュ
    cache_c = {}
    for c, cap in PAIRS:
        sK, sV, s_ids, s_off = encode(f"Country: {c}.\nThe capital city is")
        cache_c[c] = dict(sK=sK, sV=sV, sp=list(range(s_ids.shape[1])),
                          gidx=char_span_align(r_off, s_off),
                          tgt=tok(cap, add_special_tokens=False)["input_ids"][0])

    fuser = TorchHeteroRoPEFuser(shape, shape, inv, inv, init_gate=0.1)
    W_init = {k: v.detach().clone() for k, v in {**fuser.Wk, **fuser.Wv}.items()}

    def logp_target(country, sharer_country):
        """sharer_country の KV を recv に融合し、country の首都(=target)の logp。"""
        d, sd = cache_c[country], cache_c[sharer_country]
        fused = fuser.fuse(recv_layers, sd["sK"], sd["sV"], sd["sp"], sd["gidx"], rp)
        cache = DynamicCache()
        for i, (K, V) in enumerate(fused):
            cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
        out = model(input_ids=r_ids[:, -1:], attention_mask=torch.ones((1, Lr), dtype=torch.long),
                    position_ids=torch.tensor([[Lr - 1]]), past_key_values=cache, use_cache=False)
        return torch.log_softmax(out.logits[0, -1].float(), -1)[d["tgt"]]

    def logp_gate0(country):
        with torch.no_grad():
            standalone = torch.log_softmax(model(input_ids=r_ids).logits[0, -1].float(), -1)
        return float(standalone[cache_c[country]["tgt"]])

    rng = np.random.default_rng(0)
    countries = [c for c, _ in PAIRS]
    rng.shuffle(countries)
    train, held = countries[:24], countries[24:]
    print(f"[data] train={len(train)} held-out={len(held)} 国")

    opt = torch.optim.Adam(fuser.parameters(), lr=0.08)
    MARGIN, LAM_C, LAM_R = 2.0, 1.0, 1e-3
    for step in range(15):
        fuser.train()
        opt.zero_grad()
        pos = neg_hinge = 0.0
        for c in train:
            wrong = c
            while wrong == c:
                wrong = train[int(rng.integers(len(train)))]
            lp_c = logp_target(c, c)
            lp_w = logp_target(c, wrong)
            pos = pos - lp_c
            neg_hinge = neg_hinge + torch.relu(MARGIN - (lp_c - lp_w))
        reg = sum(((p - W_init[k]) ** 2).sum() for k, p in {**fuser.Wk, **fuser.Wv}.items())
        loss = (pos + LAM_C * neg_hinge) / len(train) + LAM_R * reg
        loss.backward()
        opt.step()
        if step % 4 == 0 or step == 19:
            fuser.eval()
            with torch.no_grad():
                lrn = np.mean([float(logp_target(c, c)) for c in held])
                shf = np.mean([float(logp_target(c, held[(held.index(c) + 1) % len(held)])) for c in held])
            g0 = np.mean([logp_gate0(c) for c in held])
            print(f"  step {step:2d} | loss {loss.item():.3f} | gate {torch.sigmoid(fuser.gate_logit).mean():.2f} "
                  f"| held-out logp: learned={lrn:.2f} gate0={g0:.2f} shuffled={shf:.2f}")

    fuser.eval()                                          # 最終 fuser で held-out 再評価
    with torch.no_grad():
        lrn = np.mean([float(logp_target(c, c)) for c in held])
        shf = np.mean([float(logp_target(c, held[(held.index(c) + 1) % len(held)])) for c in held])
    g0 = np.mean([logp_gate0(c) for c in held])

    print("\n[held-out 判定]")
    print(f"  learned > gate0 か? {'YES→注入が未知国でも有効=汎化✓' if lrn > g0 else 'NO→まだ汎化不十分'}"
          f"  (learned={lrn:.2f} vs gate0={g0:.2f})")
    print(f"  learned > shuffled か? {'YES→正しいshare内容を使用✓' if lrn > shf else 'NO'}"
          f"  (learned={lrn:.2f} vs shuffled={shf:.2f})")
    print("  ※ logpなので大きいほど良い。learned>gate0 かつ learned>shuffled が汎化の証拠。")


if __name__ == "__main__":
    main()
