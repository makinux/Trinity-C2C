# Local Trinity-C2C architecture design doc

- Created: 2026-06-23
- Premise: local/open models only (Qwen, GLM, DeepSeek, etc.). No closed cloud AI services.
- Communication layer: built around inter-model KV latent fusion via Cache-to-Cache (C2C).
- Background: through a joint review by Claude and Codex (gpt-5.5), the topology changed from the original chain type to a **star type**.

## Reference papers
- Trinity: An Evolved LLM Coordinator — arXiv:2512.04695
- Cache-to-Cache (C2C): Direct Semantic Communication Between LLMs — arXiv:2510.03215
- Activated LoRA (aLoRA) — arXiv:2504.12397
- Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA — arXiv:2512.17910

---

## 0. Design goals
- Achieve consistency (suppressing interpretation drift) + low cost + error decorrelation (diversity), all locally.
- Replace inter-model communication from "lossy text" with "richer latents (KV)" (= C2C).
- But work around C2C's weaknesses by design (pairwise, one-directional, fragile to history divergence, needs prefill of both models).

---

## 1. Topology (star type)

```
                         ┌───────────────────────────────┐
        Query Q ───────► │  Coordinator (router)         │
                         │  Qwen3-0.6B + head            │
                         │  train: sep-CMA-ES            │
                         │  (terminal reward)            │
                         └──────┬─────────┬─────────┬─────┘
              (control: dotted lines; routes all roles)
                  ┊            ┊                       ┊
                  ▼            ▼                       ▼
        ┌────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
        │ Thinker        │  │ Worker = Receiver ★ │  │ Verifier            │
        │ GLM            │─►│ Qwen3-Coder         │◄─│ DeepSeek-R1-distill │
        │ plan/decompose │  │ the single central  │  │ verify              │
        │ /critique      │  │ integration point   │  │ ACCEPT/REVISE       │
        └────────────────┘  └──────────┬──────────┘  └─────────────────────┘
           plan (latent/C2C) ┘         │   └─ critique (latent/C2C, REVISE)
                                        │      artifact (text) ─► Verifier
                                        ▼
                                ┌────────────────┐
                                │ Final artifact │  ◄ finalized when Verifier = ACCEPT
                                └────────────────┘
```

Legend:
- solid line (C2C latent bus): carries soft intent / plan / critique / uncertainty as latents (KV)
- dashed line (exact text/artifact): carries the "things that must be exact" (code, proofs, artifacts) as text
- dotted line (coordinator control): the Coordinator controls each role's invocation order, looping, and stopping

---

## 2. Components and model assignment (maximize error decorrelation)

| Component | Model | Role / rationale |
|---|---|---|
| Coordinator (router) | Qwen3-0.6B + ~10K-param head | Flow control only. Learns from terminal reward via sep-CMA-ES. Inherits Trinity's "dirt-cheap coordination" |
| Thinker (Sharer) | GLM | Plan / decompose / critique. Inductive bias different from the Qwen family |
| Worker = Receiver ★ (center) | Qwen3-Coder | The single integration point. Generates the artifact. Strongest local code generation = a stable fusion target |
| Verifier (Sharer) | DeepSeek-R1-distill | Verification + ACCEPT/REVISE. A different lineage from both Worker and Thinker = maximal error decorrelation |

Note: do not use GLM for both Thinker and Verifier (it breaks decorrelation).

---

## 3. Why star (why we dropped chain)
C2C is pairwise, one-directional, and fragile to history divergence. Passing latent state in a
chain Thinker->Worker->Verifier->Worker accumulates distribution shift ("latent telephone").
-> Make the central Receiver (Qwen3-Coder) the single canonical integration point, and have the
Sharers inject latents into it radially.

Only two fusers need training:
- GLM (Thinker) -> Qwen3-Coder (Receiver)   ... plan latents
- DeepSeek (Verifier) -> Qwen3-Coder (Receiver) ... critique latents

The Worker->Verifier direction needs no fuser. Passing the artifact as "exact text" is enough.

---

## 4. Dual channel (latent + exact text)
- latent channel (C2C): soft intent / plan prior / critique / uncertainty -> fused into the Receiver
- exact channel (text): code / proofs / artifacts (things that must not ride a lossy fusion)

Principle: carry "intent as latent, the concrete thing as text." Do not insist on a purely latent handoff.

---

## 5. Turn flow (up to 5 turns)
1. Q -> the Coordinator encodes it -> invokes the Thinker
2. Thinker (GLM) prefills Q and generates a plan (KV = Sharer)
3. Coordinator orders integration -> C2C (Thinker->Receiver) injects the plan latents into Qwen3-Coder -> Receiver generates the artifact (exact code)
4. Coordinator orders verification -> Verifier (DeepSeek) reads Q + the exact artifact text and outputs ACCEPT/REVISE + a critique
5. If REVISE (following the mitigation in §6), reconstruct the Receiver with a fresh prefill -> loop
6. On ACCEPT, finish -> return the artifact

---

## 6. The REVISE back-edge trap and its mitigation (most important)
Continuing to reuse the same Worker's history on Verifier->Worker breaks C2C
(injecting Verifier-derived latents from a different trajectory into an already-generated Worker = KV inconsistency).

Mitigation: treat REVISE as a "new integration path," not a "continuation."
- Re-prefill the Receiver from a canonical text state: `prompt + exact artifact + Verifier's exact critique text`
- + optionally inject Verifier->Receiver latents into the fresh prefill

---

## 7. Training order (do not co-train)
1. First build a text-only local Trinity (a baseline without C2C)
2. Train and evaluate the Thinker->Receiver fuser
3. Train the Verifier->Receiver critique fuser with the "fresh-reconstruction protocol"
4. Worker->Verifier last (exact text is likely sufficient)
5. With roles and fusers frozen, finally train the Coordinator with sep-CMA-ES

---

## 8. Infrastructure / GPU
- Practical answer: one node with 2x80GB (A100/H100). Reasonable for running several 7B-32B-class models at once, KV resident, experiments included.
- 1x80GB: possible with quantization + offload, but cramped.
- 1x24GB: prototyping only (small, quantized models only).
- Serving: a vLLM-family engine with raw KV access and injection hooks (the aLoRA-capable engine of 2512.17910 is a close foundation).
- Quantization: AWQ / GPTQ / FP8. C2C assumes both models' prefills stay resident, so budget KV memory accordingly.

---

## 9. Top 3 risks
1. KV inconsistency from history divergence -> avoid via fresh reconstruction on REVISE
2. Weak / noisy latent injection -> keep role models at comparable-or-greater strength + layer gates
3. Fuser fragility when models/roles change -> freeze the role lineup; retrain the fusers on any swap

---

## 10. Phased build-out
- P0: text-only local Trinity (no C2C) for a working baseline of behavior & accuracy
- P1: C2C-ify only the Thinker->Receiver edge and A/B-compare against the text version
- P2: complete it with the two fusers + a frozen Coordinator

---

## Design summary
Work around C2C's weaknesses (history divergence, latent telephone) with
"star + fresh reconstruction + dual channel," and secure error decorrelation through lineage
diversity across Qwen / GLM / DeepSeek.
Everything runs on local/open models; no closed AI needed.
