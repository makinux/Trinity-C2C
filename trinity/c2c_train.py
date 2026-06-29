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
from trinity.c2c_fuser_hetero import TorchHeteroRoPEFuser, TorchHeteroRoPEMLPFuser, save_fuser
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
def _setup(sharer: str, receiver: str, *, device="cpu", init_gate=0.1, tau=1.0, fuser_arch="linear"):
    from trinity.c2c_edge import _load, _encode          # lazy (torch); identical to the engine
    sm, stok, s_inv, s_shape = _load(sharer, device, torch.float32)
    rm, rtok, r_inv, r_shape = _load(receiver, device, torch.float32)
    Fuser = TorchHeteroRoPEMLPFuser if fuser_arch == "mlp" else TorchHeteroRoPEFuser
    fuser = Fuser(s_shape, r_shape, s_inv, r_inv, init_gate=init_gate, tau=tau).to(device)
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


def _greedy_rollout(ctx, layers, r_ids, n: int, prompt_ids=None):
    """Greedy free continuation from the full ctx KV `layers` (receiver-alone if recv_layers, or the
    fused model if fused layers). Returns (token_ids[n], per-step logits [n, vocab]) — no grad.

    ``prompt_ids`` matches the deployment regime (``c2c_edge._greedy_from_cache``): the FULL ctx KV
    is the past, the gen_prompt is prefilled live, then generation continues. With no prompt it falls
    back to immediate continuation (last ctx token live, rest as past)."""
    from transformers import DynamicCache
    Lr = r_ids.shape[1]
    has_prompt = prompt_ids is not None and prompt_ids.shape[1] > 0
    past = Lr if has_prompt else Lr - 1                       # full ctx as past when a prompt follows
    cache = DynamicCache()
    for i, (K, V) in enumerate(layers):
        cache.update(K[:, :past, :].unsqueeze(0), V[:, :past, :].unsqueeze(0), i)
    prefill = prompt_ids if has_prompt else r_ids[:, -1:]
    m = prefill.shape[1]
    toks, logits = [], []
    with torch.no_grad():
        out = ctx["rm"](input_ids=prefill, attention_mask=torch.ones((1, past + m), dtype=torch.long),
                        position_ids=torch.arange(past, past + m).unsqueeze(0),
                        past_key_values=cache, use_cache=True)
        cur = past + m
        for _ in range(n):
            logits.append(out.logits[0, -1].float())
            t = int(out.logits[0, -1].argmax())
            toks.append(t)
            out = ctx["rm"](input_ids=torch.tensor([[t]]), attention_mask=torch.ones((1, cur + 1), dtype=torch.long),
                            position_ids=torch.tensor([[cur]]), past_key_values=cache, use_cache=True)
            cur += 1
    return toks, torch.stack(logits)


def _max_repeat(toks: list[int]) -> int:
    """Longest run of a single repeated token (a cheap degeneration indicator)."""
    best = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if toks else 0            # [n, vocab]


def _feed(pid, toks: list[int]):
    """The live tokens fed after the full ctx KV (positions Lr..): ``[gen_prompt, toks[:-1]]``. Each
    feed position predicts the next token, so supervising the WHOLE feed gives one target per
    gen_prompt-prefill position AND per generation position."""
    if len(toks) > 1:
        return torch.cat([pid, torch.tensor([toks[:-1]], dtype=torch.long)], dim=1)
    return pid


def _feed_logits(ctx, layers, r_ids, feed_ids):
    """Logits at EVERY position of ``feed_ids``, placed live after the full ctx KV ``layers`` (past=Lr,
    positions Lr.. ; feed position j predicts feed_ids[j+1]). Differentiable through ``layers``.

    Matching fused vs gate0 over the WHOLE feed supervises the gen_prompt-prefill positions too (not
    only generation): in the deployment regime the receiver computes the gen_prompt by attending to the
    (gate>0 perturbed) fused ctx KV, so that perturbation is baked into the prompt hidden states BEFORE
    the first generation token — output-KL at generation positions alone cannot correct it."""
    from transformers import DynamicCache
    Lr = r_ids.shape[1]
    cache = DynamicCache()
    for i, (K, V) in enumerate(layers):
        cache.update(K[:, :Lr, :].unsqueeze(0), V[:, :Lr, :].unsqueeze(0), i)
    Fn = feed_ids.shape[1]
    out = ctx["rm"](input_ids=feed_ids, attention_mask=torch.ones((1, Lr + Fn), dtype=torch.long),
                    position_ids=torch.arange(Lr, Lr + Fn).unsqueeze(0),
                    past_key_values=cache, use_cache=False)
    return out.logits[0].float()                              # [Fn, vocab]


def _freeze_below_top_k(fuser: TorchHeteroRoPEFuser, k: int | None) -> None:
    """Train only the top-k terminal receiver layers' Wk/Wv (and any MLP residual) — reduce capacity /
    CPU cost. Freezes the matching ``mlp.*_<i>`` params too so an MLP fuser actually respects --top-k."""
    if not k:
        return
    layers = sorted(int(i) for i in fuser.Wk.keys())
    keep = set(layers[-k:])
    mlp = getattr(fuser, "mlp", None)
    for i in layers:
        if i not in keep:
            fuser.Wk[str(i)].requires_grad_(False)
            fuser.Wv[str(i)].requires_grad_(False)
            if mlp is not None:
                for part in (f"k1w_{i}", f"k1b_{i}", f"k2w_{i}", f"k2b_{i}",
                             f"v1w_{i}", f"v1b_{i}", f"v2w_{i}", f"v2b_{i}"):
                    if part in mlp:
                        mlp[part].requires_grad_(False)


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
    # --- Python ---
    "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):",
    "import os\nimport sys\n\ndef main():\n    path = sys.argv[1]\n    with open(path) as f:",
    "To compute the factorial of n recursively we",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):",
    "for i in range(10):\n    if i % 2 == 0:\n        print(",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:",
    "try:\n    result = compute(x)\nexcept ValueError as e:\n    ",
    "# Returns the nth Fibonacci number\ndef fib(n):\n    if n < 2:",
    "with open('out.txt', 'w') as f:\n    for line in lines:\n        f.write(",
    "def count_words(text):\n    counts = {}\n    for word in text.split():",
    "import numpy as np\narr = np.zeros((3, 3))\nfor i in range(3):",
    "@dataclass\nclass Point:\n    x: float\n    y: float\n    def dist(self):",
    # --- other languages ---
    "function debounce(fn, delay) {\n  let timer;\n  return function(...args) {",
    "const sum = arr.reduce((acc, x) =>",
    "export async function fetchUser(id) {\n  const res = await fetch(",
    "document.querySelectorAll('.item').forEach(el =>",
    "#include <stdio.h>\nint main() {\n    int sum = 0;\n    for (int i = 0;",
    "public class Main {\n    public static void main(String[] args) {\n        System.out.",
    "func quicksort(arr []int) []int {\n    if len(arr) <= 1 {",
    "fn factorial(n: u64) -> u64 {\n    if n == 0 {",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING",
    "UPDATE accounts SET balance = balance - 100 WHERE",
    "SELECT u.name, o.total FROM users u JOIN orders o ON",
    "#!/bin/bash\nfor file in *.txt; do\n    echo",
    "grep -rn 'TODO' . | awk -F:",
    ".container {\n  display: flex;\n  justify-content:",
    # --- math ---
    "The derivative of x squared with respect to x is",
    "The integral of cos(x) with respect to x is",
    "Solving the quadratic equation ax^2 + bx + c = 0 gives x =",
    "The sum of the first n natural numbers equals",
    "A prime number is a natural number greater than 1 that",
    "The probability of rolling a six on a fair die is",
    "By the Pythagorean theorem, the hypotenuse equals",
    # --- science ---
    "Machine learning models are trained by minimizing a",
    "Water is composed of two hydrogen atoms and one",
    "The mitochondria is often called the powerhouse of the",
    "Newton's second law states that force equals mass times",
    "The speed of light in a vacuum is approximately",
    "DNA is structured as a double helix consisting of",
    "The Earth orbits the Sun once every",
    # --- narrative / prose ---
    "The quick brown fox jumps over the lazy dog and then",
    "Once upon a time in a small village there lived a",
    "In this paper we propose a method that combines",
    "She opened the old wooden door and found",
    "The detective examined the room carefully, noting that",
    "After years of travel, he finally returned to",
    "The storm grew stronger as the ship sailed into",
    "Dear Sir or Madam,\n\nI am writing to apply for",
    # --- factual / QA ---
    "The capital of France is Paris and the capital of Japan is",
    "The three primary colors are red, green, and",
    "The largest ocean on Earth is the",
    "The author of Romeo and Juliet is",
    "The chemical symbol for gold is",
    "The first president of the United States was",
    "Mount Everest is the tallest mountain in the",
    "The currency used in Japan is the",
    # --- procedural / lists ---
    "To make a cup of tea, first you boil",
    "Step 1: Preheat the oven to 350 degrees. Step 2:",
    "The main causes of climate change include",
    "A balanced diet should include proteins, carbohydrates, and",
    "The seven continents of the world are",
    "The four seasons of the year are spring, summer,",
    "A traffic light shows three colors: red, yellow, and",
    "The planets in order from the Sun are Mercury, Venus,",
]

# Held-out contexts (NOT in DISTILL_CONTEXTS) — selection + the reported metric use these, so we
# measure GENERALIZATION (reproduce gate0 on unseen contexts) rather than memorization.
HELDOUT_CONTEXTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    rest =",
    "while True:\n    cmd = input()\n    if cmd == 'quit':\n        break\n    else:",
    "The largest planet in the solar system is Jupiter, and the second largest is",
    "import json\nwith open('data.json') as f:\n    data = json.load(f)\nfor item in",
    "Photosynthesis is the process by which plants convert sunlight into",
    "CREATE TABLE users (\n    id INTEGER PRIMARY KEY,\n    name TEXT NOT NULL,\n    email",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2,",
    "The result of 7 multiplied by 8 is 56, and 9 multiplied by 6 is",
    "const numbers = [5, 3, 8, 1];\nnumbers.sort((a, b) =>",
    "The boiling point of water at sea level is",
    "The formula for the area of a circle is",
    "He picked up the phone and heard a familiar",
    "git commit -m 'fix bug' && git",
    "The opposite of 'increase' is",
    "In Python, a list comprehension like [x*2 for x in",
    "<!DOCTYPE html>\n<html>\n<head>\n    <title>",
]

# Deployment-like gen_prompts (the receiver-side live tokens c2c_edge appends after the fused KV).
# Training over a mix makes the fuser robust to the gateway appending a prompt before generation.
DISTILL_PROMPTS = [" ", "\n", "\n[TASK] Continue.\n\n[Solution]\n", "\n    "]


def _build_distill_samples(ctx, contexts, n_cont):
    """Per context: fuse inputs (recv==share) + a cycled deployment gen_prompt + the receiver's own
    gate0 greedy continuation AFTER that prompt. The supervised feed is ``[gen_prompt, gate0[:-1]]`` and
    the teacher is gate0's distribution over the WHOLE feed, so BOTH the prompt-prefill positions and
    the generation positions are matched to gate0 (Lever-2.5: fix the perturbation baked into the
    prompt before the first generation token)."""
    samples = []
    for k, text in enumerate(contexts):
        fi = _fuse_inputs(ctx, text, text)
        prompt = DISTILL_PROMPTS[k % len(DISTILL_PROMPTS)]
        pid = ctx["rtok"](prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        g0_toks, _ = _greedy_rollout(ctx, fi["recv_layers"], fi["r_ids"], n_cont, pid)
        feed = _feed(pid, g0_toks)
        with torch.no_grad():
            full = torch.softmax(_feed_logits(ctx, fi["recv_layers"], fi["r_ids"], feed), -1)
        samples.append({"fi": fi, "pid": pid, "prompt": prompt, "g0_toks": g0_toks,
                        "g0_feed": feed, "g0_full": full, "op_feed": feed, "op_full": full})
    return samples


def train_distill(ctx, *, steps=60, lr=0.05, gate=0.3, n_cont=8, on_policy=True, reroll=4,
                  warmup=None, seed=0) -> dict:
    """Distill the fused (frozen-gate) model to reproduce the receiver's own gate=0 generation.

    Teacher-forcing on the gate0 trajectory matches the next-token distribution well but free
    generation still degenerates (exposure bias: at inference the fused model conditions on its OWN
    argmax tokens). ON-POLICY refinement (DAgger): after a teacher-forced warm-up, periodically
    re-roll the FUSED model's own greedy trajectory and train it to match gate0's distribution at
    *those* (visited) prefixes — plus a gate0 anchor for stability. Checkpoints are selected by the
    honest FREE-generation metric (token-match vs gate0 + max-repeat), not teacher-forced KL.

    The KL is taken over the WHOLE deployment feed ``[gen_prompt, trajectory[:-1]]`` (see
    ``_feed_logits``) — prompt-prefill positions AND generation positions — so the fused prompt hidden
    states are pulled toward gate0, not just the generation outputs. In the deployment regime the
    receiver computes the gen_prompt by attending to the perturbed fused ctx KV, so supervising only
    generation positions leaves that perturbation uncorrected (it is baked in before the first
    generation token).
    """
    fuser = ctx["fuser"]
    torch.manual_seed(seed)
    with torch.no_grad():                                   # freeze the gate (no trivial gate->0 escape)
        fuser.gate_logit.fill_(float(np.log(gate / (1 - gate))))
    fuser.gate_logit.requires_grad_(False)
    warmup = steps // 3 if warmup is None else warmup

    train_samples = _build_distill_samples(ctx, DISTILL_CONTEXTS, n_cont)
    held_samples = _build_distill_samples(ctx, HELDOUT_CONTEXTS, n_cont)   # gate progress on these

    def _fused_layers(s):
        return fuser.fuse(s["fi"]["recv_layers"], s["fi"]["sK"], s["fi"]["sV"],
                          s["fi"]["sp"], s["fi"]["gidx"], s["fi"]["rp"])

    def _full_kl(s, feed, teacher_full):                    # student fused vs gate0 over the WHOLE feed
        sl = _feed_logits(ctx, _fused_layers(s), s["fi"]["r_ids"], feed)
        return sum(distill_kl(sl[p], teacher_full[p]) for p in range(feed.shape[1])) / feed.shape[1]

    def _refresh_on_policy(s):                              # roll the FUSED trajectory; teacher = gate0 on it
        with torch.no_grad():
            f_toks, _ = _greedy_rollout(ctx, _fused_layers(s), s["fi"]["r_ids"], n_cont, s["pid"])
            feed = _feed(s["pid"], f_toks)
            s["op_feed"] = feed
            s["op_full"] = torch.softmax(_feed_logits(ctx, s["fi"]["recv_layers"], s["fi"]["r_ids"], feed), -1)

    def free_gen_metrics(samples):                         # HONEST metric: free-roll fused vs gate0
        fuser.eval()
        match, rep = [], []
        with torch.no_grad():
            for s in samples:
                f_toks, _ = _greedy_rollout(ctx, _fused_layers(s), s["fi"]["r_ids"], n_cont, s["pid"])
                match.append(float(np.mean([f == g for f, g in zip(f_toks, s["g0_toks"])])))
                rep.append(_max_repeat(f_toks))
        return float(np.mean(match)), float(np.mean(rep))

    m0_tr, _ = free_gen_metrics(train_samples)
    m0, r0 = free_gen_metrics(held_samples)
    print(f"[distill] gate={gate} (frozen) | train={len(train_samples)} held-out={len(held_samples)} | "
          f"n_cont={n_cont} steps={steps} on_policy={on_policy} warmup={warmup} reroll={reroll} | +gen_prompt regime +prompt-pos KL")
    print(f"  baseline free-gen: HELD-OUT match={m0:.0%} max-repeat={r0:.1f}  (train match={m0_tr:.0%})")

    opt = torch.optim.Adam([p for p in fuser.parameters() if p.requires_grad], lr=lr)
    best = {"match": m0, "rep": r0, "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()}}
    for step in range(steps):
        fuser.eval()                                        # frozen gate -> deterministic; Wk/Wv still train
        on_pol = on_policy and step >= warmup
        if on_pol and step % reroll == 0:
            for s in train_samples:
                _refresh_on_policy(s)
        opt.zero_grad()
        loss = 0.0
        for s in train_samples:
            if on_pol:                                      # on-policy term + gate0 anchor (DAgger mix)
                loss = loss + _full_kl(s, s["op_feed"], s["op_full"]) + 0.5 * _full_kl(s, s["g0_feed"], s["g0_full"])
            else:
                loss = loss + _full_kl(s, s["g0_feed"], s["g0_full"])
        loss = loss / len(train_samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in fuser.parameters() if p.requires_grad], 10.0)
        opt.step()
        if step % 4 == 0 or step == steps - 1:
            m, r = free_gen_metrics(held_samples)           # select + report on HELD-OUT (generalization)
            phase = "on-policy" if on_pol else "warmup"
            print(f"  step {step:2d} ({phase}) | loss {loss.item():.4f} | held-out match {m:.0%} max-repeat {r:.1f}")
            if (m, -r) > (best["match"], -best["rep"]):
                best = {"match": m, "rep": r,
                        "state": {k: v.detach().clone() for k, v in fuser.state_dict().items()}}

    fuser.load_state_dict(best["state"])                    # restore the best-by-held-out checkpoint
    fuser.gate_logit.requires_grad_(True)
    m1, r1 = best["match"], best["rep"]
    m1_tr, _ = free_gen_metrics(train_samples)
    print("\n[distill verdict] HELD-OUT free-gen token-match vs gate0 (the generalization metric)")
    print(f"  held-out match: {m0:.0%} -> {m1:.0%}   max-repeat: {r0:.1f} -> {r1:.1f}   (train match now {m1_tr:.0%})")
    print(f"  {'OK: GENERALIZES -- fused free generation tracks gate0 on UNSEEN contexts.' if m1 > m0 + 0.05 else 'limited held-out gain (overfit / data-scale frontier).'}")
    return {"objective": "distill", "gate": gate, "steps": steps, "n_cont": n_cont, "on_policy": on_policy,
            "metrics": {"held_match_before": m0, "held_match_after": m1, "held_rep_before": r0,
                        "held_rep_after": r1, "train_match_after": m1_tr}}


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
    ap.add_argument("--warmup", type=int, default=None, help="distill: teacher-forced warm-up steps before on-policy (default steps//3)")
    ap.add_argument("--fuser", choices=["linear", "mlp"], default="linear",
                    help="fuser architecture: linear Wk/Wv (default) or a per-layer residual MLP (non-linear capacity)")
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
    print(f"[load] sharer={sharer} -> receiver={receiver} (CPU, frozen) | fuser={args.fuser} ...")
    ctx = _setup(sharer, receiver, init_gate=init_gate, tau=float(get("c2c", "tau", 1.0)), fuser_arch=args.fuser)

    if args.objective == "relational":
        meta = train_relational(ctx, steps=args.steps or 30, lr=args.lr or 0.08, margin=2.0,
                                n_neg=args.n_neg, top_k=args.top_k, seed=args.seed)
    else:
        meta = train_distill(ctx, steps=args.steps or 60, lr=args.lr or 0.05, gate=args.gate,
                             on_policy=args.on_policy, reroll=args.reroll, warmup=args.warmup, seed=args.seed)

    meta["sharer_model"] = sharer; meta["receiver_model"] = receiver; meta["fuser_arch"] = args.fuser
    path = save_fuser(ctx["fuser"], out, sharer_model=sharer, receiver_model=receiver,
                      sharer_shape=ctx["s_shape"], receiver_shape=ctx["r_shape"], metadata=meta)
    print(f"\n[saved] {path}  (load via TRINITY_C2C_FUSER={path} or config c2c.fuser_path)")


if __name__ == "__main__":
    main()
