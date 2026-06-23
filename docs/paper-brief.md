# Briefing: "Trinity: An Evolved LLM Coordinator" (arXiv:2512.04695v3, Sakana AI)

This is a shared briefing so Codex and Claude can discuss the paper. Treat it as ground truth extracted from the paper.

## One-line
A tiny, evolution-trained "coordinator" (Qwen3-0.6B SLM + ~10K-param linear head, <20K trainable params total) orchestrates a pool of frontier LLMs over multiple turns by assigning each a role (Thinker / Worker / Verifier), beating every individual model and prior routing methods. New SOTA 86.2% on LiveCodeBench.

## Problem & motivation
- Combining diverse foundation models is promising, but weight-merging fails across mismatched architectures and closed APIs (can't merge GPT-5, Gemini, Claude weights).
- Proposal: **test-time model composition via coordination** — fuse complementary strengths of multiple SOTA models from different providers WITHOUT touching their weights.
- Hypothesis: contextual hidden-state representations from a *small* LM carry enough semantic signal for a *lightweight head* to coordinate multiple LLMs.

## Method (Trinity)
- **Problem formulation:** sequential decision problem. State s = query + conversation history. SLM maps state to representation h(s) ∈ R^d. Head f_θ outputs logits over agent-role pairs; policy π_θ(a|s) ∝ exp(f_θ(h(s))_a). Objective J(θ)=E_{τ~π_θ}[R(τ)], R(τ)∈{0,1} binary success. Horizon T ≤ B_turn. Strict "atomic evaluation budget" B_env counts individual Bernoulli reward calls.
- **Coordinator parametrization:**
  - Backbone SLM = Qwen3-0.6B (frozen; its generated TEXT is discarded — only hidden states/logits used).
  - Head = single linear layer, hidden dim d → L+3 outputs: L logits to pick which LLM + 3 logits to pick the role. ~10K params.
  - Also trains singular-value scales of selected SLM layers (frozen orthogonal matrices). Total trainable < 20K.
  - Head input = hidden state at the **penultimate output token** (NOT the last token — this matters, see ablation).
- **Tri-role coordination (max K=5 turns):** each turn k, given transcript C_{k-1}=(Q,O_1..O_{k-1}), coordinator picks agent A_k and role R_k:
  - **Thinker** — meta-level guidance: high-level plans, decompositions, critiques of partial solutions.
  - **Worker** — acts directly on the task; concrete progress (derivation, code snippet, numerical result).
  - **Verifier** — checks accumulated solution; outputs u_k ∈ {ACCEPT, REVISE}.
  - Termination: stop at first turn where role=Verifier and u_k=ACCEPT (else continue, cap 5). Final answer = O_τ.
- **Training = sep-CMA-ES** (separable/diagonal-covariance CMA Evolution Strategy), derivative-free.
  - Population λ = ceil(4 + 3 ln n); for n≈10,000 → λ≈32. Sample y = m_t + σ_t D_t z, z~N(0,I). Fitness-weighted recombination of top candidates. Only diagonal scaling D_t maintained.
  - m_CMA = 16 replications per candidate. Total atomic budget B_env ∈ [1.5k, 40k] evals.

## Experiments
- **Coordinator SLM:** Qwen3-0.6B. **LLM pool (7):** GPT-5, Gemini-2.5-pro, Claude-4-Sonnet (closed); Gemma-3-27B-It, DeepSeek-R1-Distill-Qwen-32B, Qwen-3-32B (direct), Qwen-3-32B (reasoning) (open). Max 4096 tokens/LLM, up to 5 turns.
- **In-distribution (4):** MATH500, MMLU, RLPR, LiveCodeBench.
  - LiveCodeBench: Trinity 0.615 vs 0.46 (Gemini, 2nd) → +33.7% relative.
  - MATH500: 0.880 vs 0.78 (random-search baseline).
  - MMLU: 0.916 (top).
  - RLPR: 0.401 vs 0.38 (Gemini).
- **LiveCodeBench SOTA (closed models, no output-length cap), V6:** Trinity **86.2% (±0.5%)** pass@1 vs GPT-5 83.8%, Gemini-2.5-Pro 67.2%, Claude-4-Sonnet 46.5%. Improves 82.3% (2 turns) → 86.3% (6 turns).
- **Zero-shot transfer (held-out: AIME2025, BigCodeBench, MT-Bench, GPQA-D):** Trinity avg 54.21 > Gemini 52.34 > GPT-5 51.07 > Claude-4-Sonnet 46.14. Top on 3/4, tie on BigCodeBench.
- **Budget-matched:** Trinity beats single models even given 5x context tokens ("5x CTX") or 5-turn self-reflection ("5x SR").
- **Ablations (drop = points lost):** w/o singular-value FT −5.78 LCB / −2.15 MATH; w/o Thinker −3.66/−1.80; w/o tri-role −3.18/−6.00; last-token instead of penultimate −10.61/−1.00 (penultimate token critical, esp. coding).
- **Optimizer comparison (LCB / MATH / MMLU / RLPR):** sep-CMA-ES 0.615/0.880/0.916/0.401; SFT 0.592/0.786/0.906/0.360; Random Search 0.374/0.794/0.897/0.345; REINFORCE 0.253/0.459/0.500/0.266. REINFORCE stays near-uniform agent selection (fails to learn); sep-CMA-ES concentrates on strong LLMs.

## Theory
- **Block-ε-separability (Def. 1):** exists diagonal scaling S s.t. scaled Hessian H_S(θ) is uniformly nearly block-diagonal on domain, inter-block coupling ≤ ε_H ∈ [0,1). Most informative signal concentrated within blocks; inter-block interference negligible.
- **Prop. 1 (gain ratio, Trinity scale n≈10k, λ=32, m_CMA=16, m_RS=32):** CMA gain / RS gain ≳ (κ̄_{μ,λ}/2)·T/ln(max{e,⌊16T⌋})·η². T=iterations; κ̄_{μ,λ}=Θ(1) recombination efficiency; η∈(0,1] reliability factor; 16T≈# RS candidates at equal budget. CMA improvement grows ~linearly in T, RS only ~log → CMA wins for modest T.
- **vs REINFORCE:** noisy global returns swamp weak inter-block signals → ill-conditioned gradients, poor credit assignment, unstable.
- **vs SFT/imitation:** multi-turn label generation cost explodes (~8.7×10^10 LLM queries) vs label-free ES.
- **Two key success factors:** (1) coordinator hidden-state reps give rich input contextualization (Sec 4.6: linear SVM near-perfect task classification, chance=0.25; t-SNE clean clusters); (2) under high-dim + strict budget, sep-CMA-ES beats RL/IL/random search by exploiting block-ε-separability (Sec 4.7: block-diagonal-10 head w/ 1,024 params keeps 0.551 LCB vs 0.615 linear).

## Related work positioning
- Micro-level (weight merging): limited to open-source checkpoints, excludes frontier closed models.
- Macro-level (MoA, RouterDC, Smoothie): expensive multi-model inference or static human-designed collaboration. Trinity = lightweight LEARNED coordinator with dynamic role assignment.

## Limitations / future work
- Gap between abstract reasoning and grounded execution: can plan tool use but cannot yet act on tools.
- Future: integrate heterogeneous agents (code interpreters, APIs) to close the plan→act gap.
