# Trinity-C2C

**A fully-local research skeleton combining Trinity-style multi-LLM coordination with
Cache-to-Cache (KV-level latent communication). Runs entirely on local/open models — no closed APIs.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)

---

## ⚠️ Positioning (please read first)

- This is a **research/educational skeleton that independently implements the ideas** of
  published papers. It is an unofficial implementation **unaffiliated** with the papers'
  authors or their institutions.
- Each module's machinery is **verified by model-free self-tests** (`scripts/selftest.sh`).
  That said, it does not ship a trained model that generalizes; **generalization is a
  data-scale WIP** (see [results](#verified-results-honest)).
- Models you use (Qwen/GLM/DeepSeek/SmolLM, etc.) are subject to **their own licenses**.

## Abstract
We propose "Trinity-C2C," a multi-LLM collaborative architecture that relies exclusively on open-source large language models (LLMs).
When coordinating multiple AI models, it is crucial to balance consistency, cost-efficiency, and error decorrelation (diversity) among the models.
While conventional multi-agent systems utilize text for inter-model communication, this research replaces that method with "Cache-to-Cache" (C2C) communication—directly sharing the models' hidden states (KV caches)—to enable the transmission of richer semantic information.

However, applying C2C communication in a serial (chain) configuration presents challenges, such as the accumulation of latent state distribution shifts (akin to the "telephone game") and system failure caused by history branching.
To address this, we redesigned the architecture using a "star topology." Specifically, we position Qwen3-Coder—which possesses powerful code generation capabilities—at the center as the sole integration point (Receiver/Worker), while latent information is injected radially from a GLM (Thinker) responsible for planning and a DeepSeek model (Verifier) ​​responsible for validation. This approach maximizes error decorrelation by combining models with distinct lineages.

Furthermore, to enhance overall system stability, we adopted a "dual-channel" method: "soft" intentions and uncertainties are transmitted via a "KV channel," while precise outputs such as code and proofs are transmitted via a "text channel."
Additionally, to avoid KV inconsistencies (the "history branching trap") that arise when the verification model requests revisions, we introduced a process that reconstructs the state using a new integration path rather than simply continuing generation.
This architecture enables the construction of sophisticated and stable AI collaborative systems in local environments, without reliance on closed cloud APIs.

## Overview

"Weight-merging across models breaks down under architecture mismatches and closed APIs" —
so this implementation provides a skeleton that, on a local-only premise, combines
Trinity's idea of **coordinating models at test time without touching their weights** with
Cache-to-Cache's idea of **communicating latently through the KV cache rather than text**.

- **star-type coordination**: a small Coordinator assigns a role (Thinker / Worker / Verifier) each turn
- **role -> model set in `config.yml`** (GLM / Qwen-Coder / DeepSeek, etc. are swappable)
- **Cache-to-Cache**: latent KV fusion between same/heterogeneous models (tokenizer alignment, GQA, **RoPE-aware** support)

See [docs/design.md](docs/design.md) for design details.

## Architecture

```
                          ┌───────────────────────┐
        Query  ─────────► │ Coordinator (router)  │   small SLM + linear head,
                          │ picks the next role   │   trained with sep-CMA-ES
                          └───────────┬───────────┘
                                      │  (routes all three roles)
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
         ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
         │   Thinker   │  ───► │  Worker ★   │ ◄───  │  Verifier   │
         │  plan /     │       │  implement  │       │   verify    │
         │  critique   │       │  /generate  │       │             │
         └─────────────┘       └─────────────┘       └─────────────┘

   ★ = the single central integration point (Receiver). role -> model in config.yml.
   Channels:  text (P0)  /  KV latent fusion (C2C: Thinker -> Worker passed as latent).
```

## Repository layout

```
.
├── config.yml                 # role -> model and other settings (the main config file)
├── trinity/                   # Python package
│   ├── config.py              # config.yml loader (built-in defaults, env overrides)
│   ├── p0.py                  # orchestration (3 roles + loop control)
│   ├── coordinator.py         # features (SLM hidden state) + linear head + learned coordinator
│   ├── train.py               # coordinator training via sep-CMA-ES
│   ├── eval.py                # execution scorer + policy comparison (bench)
│   ├── datasets.py            # HumanEval+/MBPP+ loader (differential testing)
│   ├── c2c.py                 # C2C fusion core (KVCache, layer alignment, gate)
│   ├── c2c_validate.py        # minimal same-family validation
│   ├── c2c_realrun.py         # on-device C2C transfer (small SLM)
│   ├── c2c_train_fuser.py     # cache fuser training (gradients through a frozen LM)
│   ├── c2c_hetero.py          # heterogeneous support (tokenizer alignment, GQA projection)
│   ├── c2c_rope.py            # RoPE-aware alignment
│   ├── c2c_fuser_hetero.py    # heterogeneous + RoPE + trainable fuser (integrated)
│   ├── c2c_hetero_realrun.py  # on-device heterogeneous 2-model (SmolLM <-> Qwen)
│   └── c2c_train_general.py   # data-scale / generalization training
├── scripts/selftest.sh        # all self-tests at once (no model needed)
├── docs/                      # design doc, paper summary, Docker notes
├── Dockerfile / docker-compose.yml
├── requirements.txt
├── LICENSE                    # Apache License 2.0
└── README.md
```

## Requirements / install

- **Python 3.12 recommended** (stable torch/transformers wheels)
- Self-tests need only `numpy`. On-device C2C / training need `torch`/`transformers`.

```bash
# CPU setup (self-tests + on-device C2C with a small SLM)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# optional: HumanEval+/MBPP+ loader
pip install -r requirements-datasets.txt
```
> Under corporate SSL-inspection environments, pass `--trusted-host pypi.org --trusted-host files.pythonhosted.org`
> to pip, and use `truststore` (bundled) for HuggingFace downloads.

## Quick start

```bash
# 1) All self-tests (no model; checks that the machinery is sound)
sh scripts/selftest.sh        # expected: last line is "===== ALL SELFTESTS PASSED ====="
# individually: e.g. python -m trinity.eval --selftest

# 2) Edit role -> model and other settings
$EDITOR config.yml

# 3) Verify C2C latent transfer on-device (small SLM; tens of seconds even on CPU)
python -m trinity.c2c_realrun          # as the gate rises, sharer-context info transfers into the receiver's generation

# 4) C2C across two heterogeneous models (SmolLM2-135M -> Qwen2.5-0.5B)
python -m trinity.c2c_hetero_realrun

# 5) Generalization training at data scale (with held-out evaluation)
python -m trinity.c2c_train_general
```

### Docker (reproducible environment)

```bash
docker compose build
docker compose run --rm app                       # = scripts/selftest.sh (expect all green)
docker compose run --rm app python -m trinity.c2c_realrun
# Evaluation using large models on GPU (3 vLLM servers):
docker compose --profile gpu up -d --wait
docker compose run --rm app python -m trinity.eval --trials 3
```
See [docs/docker.md](docs/docker.md) for details.

## Configuration (config.yml)

Manages per-role models and key settings in one place.

| Section | Key fields |
|---|---|
| `models.<thinker/worker/verifier>` | `base_url` / `model_id` / `api_key` / `temperature` / `max_tokens` |
| `prompts` | per-role system prompts |
| `orchestration` | `max_turns` / `verbose` |
| `coordinator` | `slm_model` / `featurizer` |
| `training` | sep-CMA-ES `budget` / `sigma0` / `m_reps` / `seed` |
| `eval` | `trials` / `timeout` / `max_inputs` |
| `c2c` | `sharer_model` / `receiver_model` / `init_gate` / `tau` |

- Override precedence: `config.yml` < environment variables (`TRINITY_CONFIG`=path, `THINKER_URL` etc.=base_url, `WORKER_API_KEY` etc.=secret keys)
- If the config file or pyyaml is missing, it runs on the built-in defaults.

## Verified results (honest)

Everything below was actually run and verified on a local CPU machine (we avoid overclaiming).

- **C2C latent transfer (same family)**: fusing France-context KV into a Japan context and
  raising the gate gives `delta-logp(" Paris") = +10.87` (at gate=0 the injection path matches
  a normal forward pass exactly = confirmed non-invasive).
- **Trainability**: gradients flow through the frozen LM into the fuser and the loss decreases.
- **On-device heterogeneous 2-model**: wired SmolLM2-135M (30 layers / 3 KV heads / rope 1e5)
  -> Qwen2.5-0.5B (24 / 2 / 1e6). Verified the machinery through tokenizer alignment, GQA, and
  **RoPE-aware** alignment (our own RoPE matches Qwen's rotary numerically).
- **Generalization (WIP)**: with 24 countries + contrastive training, held-out `learned`
  improves and uses the correct shared content (learned >> shuffled). However `learned < gate0`,
  so **the threshold is barely missed** = scaling data/compute is the next step.

## Reference papers (sources of the ideas)

- Trinity: An Evolved LLM Coordinator — arXiv:2512.04695
- Cache-to-Cache: Direct Semantic Communication Between LLMs — arXiv:2510.03215
- Activated LoRA / Multi-Adapter KV-Cache Reuse — arXiv:2504.12397, 2512.17910

## License

[Apache License 2.0](LICENSE) — Copyright 2026 NSS Co., Ltd.
See [NOTICE](NOTICE) for attribution, the unofficial-status statement, the source papers, and
the handling of referenced models. Licenses for the models and datasets you use are governed
separately by their respective providers.
