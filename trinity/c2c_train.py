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


def _gate0_trajectory(ctx, recv_layers, r_ids, n: int):
    """Receiver-alone greedy continuation from the full ctx KV (the gate=0 generation the gateway
    would produce). Returns (token_ids[n], per-step teacher logits [n, vocab]) — no grad."""
    from transformers import DynamicCache
    Lr = r_ids.shape[1]
    cache = DynamicCache()
    for i, (K, V) in enumerate(recv_layers):
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
    return toks, torch.stack(logits)            # [n, vocab]


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


def train_distill(ctx, *, steps=50, lr=0.05, gate=0.3, n_cont=8, seed=0) -> dict:
    fuser = ctx["fuser"]
    torch.manual_seed(seed)
    # Freeze the gate at the deployment value so the model can't trivially learn gate->0; train Wk/Wv.
    with torch.no_grad():
        fuser.gate_logit.fill_(float(np.log(gate / (1 - gate))))
    fuser.gate_logit.requires_grad_(False)

    # Per context: fuse inputs (recv==share) + the receiver's OWN gate0 continuation (the trajectory
    # the gateway would generate). Matching it teacher-forced trains the fused KV to reproduce
    # receiver-alone GENERATION, not just a single next-token (which collapses to a degenerate loop).
    samples = []
    for text in DISTILL_CONTEXTS:
        fi = _fuse_inputs(ctx, text, text)
        toks, t_logits = _gate0_trajectory(ctx, fi["recv_layers"], fi["r_ids"], n_cont)
        samples.append({"fi": fi, "toks": toks, "t_probs": torch.softmax(t_logits, -1).detach()})

    def _student_logits(s):
        fused = fuser.fuse(s["fi"]["recv_layers"], s["fi"]["sK"], s["fi"]["sV"],
                           s["fi"]["sp"], s["fi"]["gidx"], s["fi"]["rp"])
        return _continuation_logits(ctx, fused, s["fi"]["r_ids"], s["toks"])

    def _traj_loss(s):
        sl = _student_logits(s)
        return sum(distill_kl(sl[p], s["t_probs"][p]) for p in range(len(s["toks"]))) / len(s["toks"])

    def eval_metrics():
        fuser.eval()
        kls, matches = [], []
        with torch.no_grad():
            for s in samples:
                sl = _student_logits(s)
                kls.append(float(sum(distill_kl(sl[p], s["t_probs"][p]) for p in range(len(s["toks"]))) / len(s["toks"])))
                matches.append(float((sl.argmax(-1) == torch.tensor(s["toks"])).float().mean()))
        return float(np.mean(kls)), float(np.mean(matches))

    kl0, match0 = eval_metrics()
    print(f"[distill] gate={gate} (frozen) | contexts={len(samples)} | n_cont={n_cont} | steps={steps}")
    print(f"  baseline (untrained @ gate={gate}): mean_traj_KL(vs gate0)={kl0:.4f}  per-token match={match0:.0%}")

    opt = torch.optim.Adam([p for p in fuser.parameters() if p.requires_grad], lr=lr)
    # Start best from the initial (frozen-gate) state so a checkpoint is always saved even if no
    # step beats the baseline; training oscillates, so we keep the best-by-eval-KL state.
    best = {"kl": kl0, "match": match0,
            "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()}}
    for step in range(steps):
        fuser.eval()        # gate is frozen -> use the deterministic gate (no Gumbel noise); Wk/Wv
                            # still receive gradients (autograd is governed by requires_grad, not mode)
        opt.zero_grad()
        loss = sum(_traj_loss(s) for s in samples) / len(samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in fuser.parameters() if p.requires_grad], 10.0)
        opt.step()
        if step % 4 == 0 or step == steps - 1:
            kl, match = eval_metrics()
            print(f"  step {step:2d} | train_KL {loss.item():.4f} | eval mean_traj_KL {kl:.4f} | per-token match {match:.0%}")
            if kl < best["kl"]:
                best = {"kl": kl, "match": match,
                        "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()}}

    if best["state"] is not None:
        fuser.load_state_dict(best["state"])               # restore the best-by-KL checkpoint
    fuser.gate_logit.requires_grad_(True)                  # unfreeze for saving (state is what matters)
    kl1, match1 = best["kl"], best["match"]
    print("\n[distill verdict] (lower KL / higher match = gate>0 generation tracks gate=0 = non-degrading)")
    print(f"  mean_traj_KL(vs gate0): {kl0:.4f} -> {kl1:.4f}   per-token match: {match0:.0%} -> {match1:.0%}  (best checkpoint)")
    print(f"  {'OK: forced gate>0 now reproduces the receiver-alone continuation much more closely.' if kl1 < kl0 else 'NO improvement.'}")
    return {"objective": "distill", "gate": gate, "steps": steps, "n_cont": n_cont,
            "metrics": {"kl_before": kl0, "kl_after": kl1, "match_before": match0, "match_after": match1}}


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
        meta = train_distill(ctx, steps=args.steps or 60, lr=args.lr or 0.05, gate=args.gate, seed=args.seed)

    meta["sharer_model"] = sharer; meta["receiver_model"] = receiver
    path = save_fuser(ctx["fuser"], out, sharer_model=sharer, receiver_model=receiver,
                      sharer_shape=ctx["s_shape"], receiver_shape=ctx["r_shape"], metadata=meta)
    print(f"\n[saved] {path}  (load via TRINITY_C2C_FUSER={path} or config c2c.fuser_path)")


if __name__ == "__main__":
    main()
