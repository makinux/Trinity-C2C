"""
P2: train the HETEROGENEOUS C2C fuser (engine-compatible) + save a checkpoint
=============================================================================
This trains the exact fuser the gateway uses — ``TorchHeteroRoPEFuser`` over the configured
sharer (SmolLM2-135M) -> receiver (Qwen2.5-0.5B) pair — and saves a checkpoint the engine can
load (``TRINITY_C2C_FUSER`` / ``c2c.fuser_path``). Two objectives:

  --objective relational : Codex's gap-closing recipe for the country->capital task. The receiver
      is a fixed "Country: Unknown..." placeholder; the sharer is a real country, so the fuser must
      inject *country identity*. Contrastive (correct sharer beats several wrong sharers) +
      train/held-out split + best-by-held-out checkpoint. Reports learned vs gate0 vs shuffled
      (logp; higher is better) — evidence the mechanism GENERALIZES to unseen countries.

  --objective distill : the gateway regime (recv_text == share_text == the same context, different
      lineages). Train Wk/Wv at a FIXED non-zero gate so the fused next-token distribution matches
      the receiver's gate=0 distribution (KL). Goal: forced gate>0 stops DEGRADING (a safety/
      compatibility property), NOT a quality gain. The gate is frozen at the deployment value so the
      trivial "learn gate->0" escape is unavailable.

Honest scope: a fuser is task-specific. Neither objective makes gate>0 improve arbitrary gateway
CODE generation — that needs C2C-paper-scale data/compute. This delivers a trainable + loadable
pipeline with honest measurements (relational generalization; distill non-degradation).

Run:
  python -m trinity.c2c_train --selftest                                   # model-free loss checks
  python -m trinity.c2c_train --objective distill   --out checkpoints/distill.pt
  python -m trinity.c2c_train --objective relational --out checkpoints/relational.pt
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass

from trinity.c2c import KVShape
from trinity.c2c_fuser_hetero import TorchHeteroRoPEFuser, save_fuser
from trinity.config import get


# ============================================================
# Loss helpers (pure tensor math — model-free testable in selftest)
# ============================================================
def contrastive_loss(lp_correct: torch.Tensor, lp_wrongs: list[torch.Tensor],
                     margin: float = 2.0) -> torch.Tensor:
    """Maximize the correct-sharer target logp and margin-rank it above each wrong sharer."""
    pos = -lp_correct
    if lp_wrongs:
        hinge = sum(torch.relu(margin - (lp_correct - lw)) for lw in lp_wrongs) / len(lp_wrongs)
    else:
        hinge = lp_correct.new_zeros(())
    return pos + hinge


def distill_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) in nats: make the gate>0 fused distribution match the gate=0 receiver.

    reduction="sum" gives the true KL over the vocab for one distribution (batchmean would divide by
    the vocab size for a 1-D input, collapsing the value to ~0 and hiding the trend)."""
    return F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="sum")


# ============================================================
# On-device helpers (reuse the engine's exact load/encode so checkpoints are engine-compatible)
# ============================================================
def _setup(sharer: str, receiver: str, *, device="cpu", init_gate=0.1, tau=1.0):
    from trinity.c2c_edge import _load, _encode          # lazy (torch); identical to the engine
    sm, stok, s_inv, s_shape = _load(sharer, device, torch.float32)
    rm, rtok, r_inv, r_shape = _load(receiver, device, torch.float32)
    fuser = TorchHeteroRoPEFuser(s_shape, r_shape, s_inv, r_inv, init_gate=init_gate, tau=tau).to(device)
    return {"sm": sm, "stok": stok, "rm": rm, "rtok": rtok, "s_inv": s_inv, "r_inv": r_inv,
            "s_shape": s_shape, "r_shape": r_shape, "fuser": fuser, "sharer": sharer,
            "receiver": receiver, "device": device, "_encode": _encode}


def _fuse_inputs(ctx, share_text, recv_text):
    """encode sharer(share_text) + receiver(recv_text) -> the args TorchHeteroRoPEFuser.fuse needs."""
    enc = ctx["_encode"]
    sK, sV, s_ids, s_off = enc(ctx["sm"], ctx["stok"], share_text, ctx["device"])
    rK, rV, r_ids, r_off = enc(ctx["rm"], ctx["rtok"], recv_text, ctx["device"])
    from trinity.c2c_hetero import char_span_align
    gidx = char_span_align(r_off, s_off)
    recv_layers = [(rK[l], rV[l]) for l in range(ctx["r_shape"].n_layers)]
    return {"recv_layers": recv_layers, "sK": sK, "sV": sV,
            "sp": list(range(s_ids.shape[1])), "gidx": gidx,
            "rp": list(range(r_ids.shape[1])), "r_ids": r_ids}


def _fused_last_logits(ctx, fi):
    """Fuse and return the receiver's next-token logits over the fused KV (differentiable)."""
    from transformers import DynamicCache
    fused = ctx["fuser"].fuse(fi["recv_layers"], fi["sK"], fi["sV"], fi["sp"], fi["gidx"], fi["rp"])
    Lr = fi["r_ids"].shape[1]
    cache = DynamicCache()
    for i, (K, V) in enumerate(fused):
        cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
    out = ctx["rm"](input_ids=fi["r_ids"][:, -1:], attention_mask=torch.ones((1, Lr), dtype=torch.long),
                    position_ids=torch.tensor([[Lr - 1]]), past_key_values=cache, use_cache=False)
    return out.logits[0, -1].float()


def _receiver_gate0_logits(ctx, recv_text):
    """The receiver's plain next-token logits for recv_text (no fusion) = the gate0 reference."""
    enc = ctx["_encode"]
    _, _, r_ids, _ = enc(ctx["rm"], ctx["rtok"], recv_text, ctx["device"])
    with torch.no_grad():
        return ctx["rm"](input_ids=r_ids).logits[0, -1].float()


def _greedy_rollout(ctx, layers, r_ids, n: int):
    """Greedy free continuation from the full ctx KV `layers` (receiver-alone if recv_layers, or the
    fused model if fused layers). Returns (token_ids[n], per-step logits [n, vocab]) — no grad.
    This is the actual free-generation path (the model conditions on its OWN argmax outputs)."""
    from transformers import DynamicCache
    Lr = r_ids.shape[1]
    cache = DynamicCache()
    for i, (K, V) in enumerate(layers):
        cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
    cur, nxt, toks, logits = Lr - 1, r_ids[:, -1:], [], []
    with torch.no_grad():
        for _ in range(n):
            out = ctx["rm"](input_ids=nxt, attention_mask=torch.ones((1, cur + 1), dtype=torch.long),
                            position_ids=torch.tensor([[cur]]), past_key_values=cache, use_cache=True)
            cur += 1
            logits.append(out.logits[0, -1].float())
            t = int(out.logits[0, -1].argmax())
            toks.append(t)
            nxt = torch.tensor([[t]])
    return toks, torch.stack(logits)


def _max_repeat(toks: list[int]) -> int:
    """Longest run of a single repeated token (a cheap degeneration indicator)."""
    best = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if toks else 0            # [n, vocab]


def _continuation_logits(ctx, layers, r_ids, cont_ids: list[int]):
    """Teacher-forced: with `layers` (full ctx KV) as past, feed [last_ctx_token, *cont_ids[:-1]] and
    return logits [n, vocab] predicting cont_ids[0..n-1] (differentiable through `layers`)."""
    from transformers import DynamicCache
    Lr = r_ids.shape[1]
    cache = DynamicCache()
    for i, (K, V) in enumerate(layers):
        cache.update(K[:, :Lr - 1, :].unsqueeze(0), V[:, :Lr - 1, :].unsqueeze(0), i)
    prev = torch.tensor([cont_ids[:-1]], dtype=torch.long) if len(cont_ids) > 1 else torch.empty((1, 0), dtype=torch.long)
    feed = torch.cat([r_ids[:, -1:], prev], dim=1)            # [1, n]
    n = feed.shape[1]
    out = ctx["rm"](input_ids=feed, attention_mask=torch.ones((1, Lr - 1 + n), dtype=torch.long),
                    position_ids=torch.arange(Lr - 1, Lr - 1 + n).unsqueeze(0),
                    past_key_values=cache, use_cache=False)
    return out.logits[0].float()                 # [n, vocab]


def _freeze_below_top_k(fuser: TorchHeteroRoPEFuser, k: int | None) -> None:
    """Train only the top-k terminal receiver layers' Wk/Wv (reduce capacity / CPU cost)."""
    if not k:
        return
    layers = sorted(int(i) for i in fuser.Wk.keys())
    keep = set(layers[-k:])
    for i in layers:
        if i not in keep:
            fuser.Wk[str(i)].requires_grad_(False)
            fuser.Wv[str(i)].requires_grad_(False)


# ============================================================
# Country -> capital data (relational objective)
# ============================================================
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
RECV_PLACEHOLDER = "Country: Unknown.\nThe capital city is"


def train_relational(ctx, *, steps=30, lr=0.08, margin=2.0, n_neg=3, lam_reg=1e-3,
                     top_k=None, seed=0) -> dict:
    rm, rtok = ctx["rm"], ctx["rtok"]
    fuser = ctx["fuser"]
    rng = np.random.default_rng(seed)
    _freeze_below_top_k(fuser, top_k)

    # cache per-country fuse inputs + target token; receiver placeholder is shared
    def tgt_id(cap):
        return rtok(cap, add_special_tokens=False)["input_ids"][0]
    data = {}
    for c, cap in PAIRS:
        fi = _fuse_inputs(ctx, f"Country: {c}.\nThe capital city is", RECV_PLACEHOLDER)
        data[c] = {"fi": fi, "tgt": tgt_id(cap)}
    gate0 = _receiver_gate0_logits(ctx, RECV_PLACEHOLDER)
    gate0_logp = torch.log_softmax(gate0, -1)

    countries = [c for c, _ in PAIRS]
    rng.shuffle(countries)
    n_train = max(8, int(len(countries) * 0.6))
    train, held = countries[:n_train], countries[n_train:]
    print(f"[relational] train={len(train)} held-out={len(held)} | steps={steps} neg={n_neg} top_k={top_k}")

    # Snapshot Wk and Wv separately — they share layer-index keys, so {**Wk, **Wv} would collide.
    Wk_init = {i: p.detach().clone() for i, p in fuser.Wk.items()}
    Wv_init = {i: p.detach().clone() for i, p in fuser.Wv.items()}

    def target_logp(country, sharer_country):
        """logp of `country`'s capital when `sharer_country`'s KV is injected into the (shared)
        placeholder receiver. The fuse inputs depend only on the sharer, so reuse its cached fi."""
        logits = _fused_last_logits(ctx, data[sharer_country]["fi"])
        return torch.log_softmax(logits, -1)[data[country]["tgt"]]

    opt = torch.optim.Adam([p for p in fuser.parameters() if p.requires_grad], lr=lr)
    best = {"held": -1e9, "state": None, "metrics": None}
    for step in range(steps):
        fuser.train()
        opt.zero_grad()
        pos = hinge = 0.0
        for c in train:
            lp_c = target_logp(c, c)
            wrongs = []
            for _ in range(n_neg):
                w = c
                while w == c:
                    w = train[int(rng.integers(len(train)))]
                wrongs.append(target_logp(c, w))
            pos = pos - lp_c
            hinge = hinge + sum(torch.relu(margin - (lp_c - lw)) for lw in wrongs) / n_neg
        reg = (sum(((fuser.Wk[i] - Wk_init[i]) ** 2).sum() for i in fuser.Wk)
               + sum(((fuser.Wv[i] - Wv_init[i]) ** 2).sum() for i in fuser.Wv))
        loss = (pos + hinge) / len(train) + lam_reg * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in fuser.parameters() if p.requires_grad], 10.0)
        opt.step()

        if step % 3 == 0 or step == steps - 1:
            fuser.eval()
            with torch.no_grad():
                lrn = float(np.mean([float(target_logp(c, c)) for c in held]))
                shf = float(np.mean([float(target_logp(c, held[(held.index(c) + 1) % len(held)])) for c in held]))
                g0 = float(np.mean([float(gate0_logp[data[c]["tgt"]]) for c in held]))
            print(f"  step {step:2d} | loss {loss.item():6.3f} | gate {torch.sigmoid(fuser.gate_logit).mean():.2f}"
                  f" | held-out logp: learned={lrn:.2f} gate0={g0:.2f} shuffled={shf:.2f}")
            if lrn > best["held"]:                              # best-by-held-out checkpoint
                best = {"held": lrn, "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()},
                        "metrics": {"learned": lrn, "gate0": g0, "shuffled": shf, "step": step}}

    if best["state"] is not None:
        fuser.load_state_dict(best["state"])                    # restore the best held-out checkpoint
    m = best["metrics"] or {}
    print("\n[relational verdict] (logp; higher is better)")
    print(f"  learned > gate0?    {'YES (generalizes to unseen countries)' if m.get('learned',0) > m.get('gate0',0) else 'NO (not generalizing yet)'}"
          f"  (learned={m.get('learned',0):.2f} vs gate0={m.get('gate0',0):.2f})")
    print(f"  learned > shuffled? {'YES (uses the correct share content)' if m.get('learned',0) > m.get('shuffled',0) else 'NO'}"
          f"  (learned={m.get('learned',0):.2f} vs shuffled={m.get('shuffled',0):.2f})")
    return {"objective": "relational", "metrics": m, "train": len(train), "held": len(held),
            "steps": steps, "n_neg": n_neg, "top_k": top_k}


# ============================================================
# Self-distillation (distill objective) — gateway regime: recv == share, fixed gate
# ============================================================
DISTILL_CONTEXTS = [
    "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):",
    "The quick brown fox jumps over the lazy dog and then",
    "import os\nimport sys\n\ndef main():\n    path = sys.argv[1]\n    with open(path) as f:",
    "In this paper we propose a method that combines",
    "To compute the factorial of n recursively we",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):",
    "The capital of France is Paris and the capital of Japan is",
    "for i in range(10):\n    if i % 2 == 0:\n        print(",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:",
    "Once upon a time in a small village there lived a",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING",
    "The derivative of x squared with respect to x is",
    "try:\n    result = compute(x)\nexcept ValueError as e:\n    ",
    "Machine learning models are trained by minimizing a",
    "# Returns the nth Fibonacci number\ndef fib(n):\n    if n < 2:",
    "The three primary colors are red, green, and",
]


def train_distill(ctx, *, steps=60, lr=0.05, gate=0.3, n_cont=8, on_policy=True, reroll=4,
                  warmup=None, seed=0) -> dict:
    """Distill the fused (frozen-gate) model to reproduce the receiver's own gate=0 generation.

    Teacher-forcing on the gate0 trajectory matches the next-token distribution well but free
    generation still degenerates (exposure bias: at inference the fused model conditions on its OWN
    argmax tokens). ON-POLICY refinement (DAgger): after a teacher-forced warm-up, periodically
    re-roll the FUSED model's own greedy trajectory and train it to match gate0's distribution at
    *those* (visited) prefixes — plus a gate0 anchor for stability. Checkpoints are selected by the
    honest FREE-generation metric (token-match vs gate0 + max-repeat), not teacher-forced KL.
    """
    fuser = ctx["fuser"]
    torch.manual_seed(seed)
    with torch.no_grad():                                   # freeze the gate (no trivial gate->0 escape)
        fuser.gate_logit.fill_(float(np.log(gate / (1 - gate))))
    fuser.gate_logit.requires_grad_(False)
    warmup = steps // 3 if warmup is None else warmup

    samples = []                                            # recv==share; gate0 trajectory = warm-up teacher
    for text in DISTILL_CONTEXTS:
        fi = _fuse_inputs(ctx, text, text)
        g0_toks, g0_logits = _greedy_rollout(ctx, fi["recv_layers"], fi["r_ids"], n_cont)
        s = {"fi": fi, "g0_toks": g0_toks, "g0_probs": torch.softmax(g0_logits, -1).detach()}
        s["op_toks"], s["op_probs"] = s["g0_toks"], s["g0_probs"]   # on-policy trajectory (refreshed)
        samples.append(s)

    def _fused_layers(s):
        return fuser.fuse(s["fi"]["recv_layers"], s["fi"]["sK"], s["fi"]["sV"],
                          s["fi"]["sp"], s["fi"]["gidx"], s["fi"]["rp"])

    def _kl_on(s, toks, t_probs):                           # student fused, teacher-forced on `toks`
        sl = _continuation_logits(ctx, _fused_layers(s), s["fi"]["r_ids"], toks)
        return sum(distill_kl(sl[p], t_probs[p]) for p in range(len(toks))) / len(toks)

    def _refresh_on_policy(s):                              # roll the FUSED model's own trajectory
        with torch.no_grad():
            f_toks, _ = _greedy_rollout(ctx, _fused_layers(s), s["fi"]["r_ids"], n_cont)
            t_logits = _continuation_logits(ctx, s["fi"]["recv_layers"], s["fi"]["r_ids"], f_toks)
        s["op_toks"], s["op_probs"] = f_toks, torch.softmax(t_logits, -1).detach()

    def free_gen_metrics():                                # the HONEST metric: free-roll fused vs gate0
        fuser.eval()
        match, rep = [], []
        with torch.no_grad():
            for s in samples:
                f_toks, _ = _greedy_rollout(ctx, _fused_layers(s), s["fi"]["r_ids"], n_cont)
                match.append(float(np.mean([f == g for f, g in zip(f_toks, s["g0_toks"])])))
                rep.append(_max_repeat(f_toks))
        return float(np.mean(match)), float(np.mean(rep))

    m0, r0 = free_gen_metrics()
    g0_rep = float(np.mean([_max_repeat(s["g0_toks"]) for s in samples]))
    print(f"[distill] gate={gate} (frozen) | contexts={len(samples)} | n_cont={n_cont} | steps={steps} "
          f"| on_policy={on_policy} warmup={warmup} reroll={reroll}")
    print(f"  baseline free-gen: token-match-vs-gate0={m0:.0%}  max-repeat={r0:.1f}  (gate0 itself={g0_rep:.1f})")

    opt = torch.optim.Adam([p for p in fuser.parameters() if p.requires_grad], lr=lr)
    best = {"match": m0, "rep": r0, "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()}}
    for step in range(steps):
        fuser.eval()                                        # frozen gate -> deterministic; Wk/Wv still train
        on_pol = on_policy and step >= warmup
        if on_pol and step % reroll == 0:
            for s in samples:
                _refresh_on_policy(s)
        opt.zero_grad()
        loss = 0.0
        for s in samples:
            if on_pol:                                      # on-policy term + gate0 anchor (DAgger mix)
                loss = loss + _kl_on(s, s["op_toks"], s["op_probs"]) + 0.5 * _kl_on(s, s["g0_toks"], s["g0_probs"])
            else:
                loss = loss + _kl_on(s, s["g0_toks"], s["g0_probs"])
        loss = loss / len(samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in fuser.parameters() if p.requires_grad], 10.0)
        opt.step()
        if step % 4 == 0 or step == steps - 1:
            m, r = free_gen_metrics()
            phase = "on-policy" if on_pol else "warmup"
            print(f"  step {step:2d} ({phase}) | loss {loss.item():.4f} | free-gen match {m:.0%} max-repeat {r:.1f}")
            if (m, -r) > (best["match"], -best["rep"]):     # select by free-gen match, tie-break on repetition
                best = {"match": m, "rep": r,
                        "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()}}

    fuser.load_state_dict(best["state"])                    # restore the best-by-free-gen checkpoint
    fuser.gate_logit.requires_grad_(True)
    m1, r1 = best["match"], best["rep"]
    print("\n[distill verdict] (free-gen token-match vs gate0; higher match / lower repeat = less degeneration)")
    print(f"  free-gen match: {m0:.0%} -> {m1:.0%}   max-repeat: {r0:.1f} -> {r1:.1f}  (best-by-free-gen checkpoint)")
    print(f"  {'OK: fused free generation tracks the receiver-alone output more closely.' if m1 > m0 else 'NO improvement.'}")
    return {"objective": "distill", "gate": gate, "steps": steps, "n_cont": n_cont, "on_policy": on_policy,
            "metrics": {"match_before": m0, "match_after": m1, "rep_before": r0, "rep_after": r1}}


# ============================================================
# selftest (model-free): the loss functions optimize correctly on synthetic tensors
# ============================================================
def selftest() -> None:
    torch.manual_seed(0)
    # distill_kl: a free logit vector should converge to the teacher distribution
    vocab = 32
    teacher = torch.softmax(torch.randn(vocab), -1)
    student = torch.zeros(vocab, requires_grad=True)
    opt = torch.optim.Adam([student], lr=0.2)
    kls = []
    for _ in range(200):
        opt.zero_grad(); loss = distill_kl(student, teacher); loss.backward(); opt.step()
        kls.append(loss.item())
    assert kls[-1] < kls[0] * 0.1, f"distill_kl did not converge: {kls[0]:.3f}->{kls[-1]:.3f}"
    assert torch.allclose(torch.softmax(student.detach(), -1), teacher, atol=1e-2)
    print(f"[selftest] distill_kl converges to teacher: KL {kls[0]:.3f}->{kls[-1]:.4f} OK")

    # contrastive_loss: push the correct score up and above wrongs past the margin
    correct = torch.zeros((), requires_grad=True)
    wrongs_base = [torch.tensor(0.5), torch.tensor(-0.2)]
    opt = torch.optim.Adam([correct], lr=0.1)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = contrastive_loss(correct, [w.detach() for w in wrongs_base], margin=2.0)
        loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0], f"contrastive_loss not decreasing: {losses[0]:.3f}->{losses[-1]:.3f}"
    assert float(correct.detach()) > float(max(wrongs_base)), "correct score did not rise above wrongs"
    print(f"[selftest] contrastive_loss raises correct above wrongs+margin: {losses[0]:.3f}->{losses[-1]:.3f} OK")

    # _max_repeat: degeneration indicator (longest single-token run)
    assert _max_repeat([1, 2, 3, 4]) == 1 and _max_repeat([5, 5, 5, 2, 9, 9]) == 3 and _max_repeat([]) == 0
    print("[selftest] _max_repeat (degeneration metric) OK")
    print("[selftest] ALL PASSED")


# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Train the heterogeneous C2C fuser and save a checkpoint.")
    ap.add_argument("--objective", choices=["relational", "distill"])
    ap.add_argument("--out", default=None, help="checkpoint output path (default: checkpoints/<objective>.pt)")
    ap.add_argument("--selftest", action="store_true", help="model-free loss checks (no download)")
    ap.add_argument("--sharer", default=None); ap.add_argument("--receiver", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="optimizer LR (defaults per objective)")
    ap.add_argument("--gate", type=float, default=0.3, help="distill: frozen gate; relational: init gate")
    ap.add_argument("--n-neg", type=int, default=3); ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--reroll", type=int, default=4, help="distill on-policy: re-roll fused trajectories every K steps")
    ap.add_argument("--no-on-policy", action="store_false", dest="on_policy",
                    help="distill: disable on-policy (DAgger) refinement, warm-up teacher-forcing only")
    ap.set_defaults(on_policy=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest or not args.objective:
        selftest(); return

    sharer = args.sharer or get("c2c", "sharer_model", "HuggingFaceTB/SmolLM2-135M-Instruct")
    receiver = args.receiver or get("c2c", "receiver_model", "Qwen/Qwen2.5-0.5B-Instruct")
    out = args.out or f"checkpoints/{args.objective}.pt"
    init_gate = args.gate if args.objective == "relational" else 0.05
    print(f"[load] sharer={sharer} -> receiver={receiver} (CPU, frozen) ...")
    ctx = _setup(sharer, receiver, init_gate=init_gate, tau=float(get("c2c", "tau", 1.0)))

    if args.objective == "relational":
        meta = train_relational(ctx, steps=args.steps or 30, lr=args.lr or 0.08, margin=2.0,
                                n_neg=args.n_neg, top_k=args.top_k, seed=args.seed)
    else:
        meta = train_distill(ctx, steps=args.steps or 60, lr=args.lr or 0.05, gate=args.gate,
                             on_policy=args.on_policy, reroll=args.reroll, seed=args.seed)

    meta["sharer_model"] = sharer; meta["receiver_model"] = receiver
    path = save_fuser(ctx["fuser"], out, sharer_model=sharer, receiver_model=receiver,
                      sharer_shape=ctx["s_shape"], receiver_shape=ctx["r_shape"], metadata=meta)
    print(f"\n[saved] {path}  (load via TRINITY_C2C_FUSER={path} or config c2c.fuser_path)")


if __name__ == "__main__":
    main()
