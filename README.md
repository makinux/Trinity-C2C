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
│   ├── p0.py                  # orchestration (3 roles + loop control, text channel)
│   ├── p1.py                  # C2C orchestration (Thinker->Worker via KV fusion; reuses p0 + events)
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
│   ├── c2c_edge.py            # reusable heterogeneous C2C edge + multi-token generation from fused KV
│   ├── c2c_train_general.py   # data-scale / generalization training
│   ├── events.py              # structured workflow-trace events (run() on_event hook)
│   ├── mocks.py               # offline mock backend (run the gateway/UI without a GPU)
│   └── gateway/               # OpenAI-compatible API + debug ChatUI (FastAPI + static)
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

## API gateway & debug UI

An OpenAI-compatible gateway fronts the whole pipeline as a single model (`trinity-p0`),
together with a browser **workflow debugger** that traces every Coordinator decision and
role turn (Thinker / Worker / Verifier) live.

Both ways below default to the offline mock backend, so they work with no GPU/models.

```bash
# Option A — Docker (lightweight image, no torch/transformers; builds in seconds):
docker compose up -d --build --wait gateway        # then open http://localhost:8080/  (docker compose down to stop)

# Option B — local Python:
pip install -r requirements.txt                    # adds fastapi + uvicorn
TRINITY_GATEWAY_MOCK=1 python -m trinity.gateway   # -> http://127.0.0.1:8080
```

- **Debug UI** — open <http://localhost:8080/> in a browser **while the gateway is running**
  (opening the raw `index.html` file won't reach the API). Enter a query, keep *Mock mode* on,
  and watch the timeline stream `Coordinator → Thinker → Worker → Verifier (REVISE → ACCEPT)`,
  each step showing the role output, the exact prompt it received, the verdict, and timing.
- **OpenAI API** — point any OpenAI client at `http://localhost:8080/v1`:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"trinity-p0",
       "messages":[{"role":"user","content":"merge two sorted lists"}],
       "extra_body":{"trinity_mock":true,"trinity_trace":true}}'
```

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | lists the virtual model `trinity-p0` |
| `POST /v1/chat/completions` | OpenAI-compatible (`stream:true` for SSE). Final artifact = assistant message; `extra_body.trinity_trace=true` adds the full workflow trace |
| `POST /debug/runs/stream` | live SSE trace consumed by the debug UI |
| `GET /` | the debug ChatUI |

### Try real models via local Ollama (no GPU required)

The bundled `ollama` compose profile runs local OpenAI-compatible models and points all three
roles at them — a one-command way to leave mock mode without a GPU:

```bash
cp .env.ollama.example .env.ollama
docker compose --env-file .env.ollama up -d --build --wait gateway ollama ollama-pull   # first run builds + pulls (minutes)
# open http://localhost:8080/  and UNCHECK "Mock mode (offline)", then Run
docker compose down                                                                     # stop (down -v drops models)
```

`.env.ollama` sets `TRINITY_GATEWAY_MOCK=0`, points `*_URL` at `http://ollama:11434/v1`, and picks
the per-role tags (`THINKER_MODEL_ID` / `WORKER_MODEL_ID` / `VERIFIER_MODEL_ID`, small CPU-friendly
defaults). The `ollama-pull` service pulls exactly those tags into a persistent volume.

For **GPU vLLM** instead, start the `gpu` profile and run the gateway with `TRINITY_GATEWAY_MOCK=0`.
Per-role endpoint/model/key are overridable by env: `THINKER_URL` / `THINKER_MODEL_ID` /
`THINKER_API_KEY` (and the `WORKER_`/`VERIFIER_` equivalents). Model-free check:
`python -m trinity.gateway.selftest` (also wired into `scripts/selftest.sh`). Gateway env knobs:
`TRINITY_GATEWAY_HOST` (default `127.0.0.1`), `TRINITY_GATEWAY_PORT` (default `8080`; the
conventional `PORT` var wins if set), `TRINITY_GATEWAY_MOCK` (`1` = offline mock).

### Try real C2C — heterogeneous KV fusion in-process (no GPU)

The Ollama profile above runs the **text channel (P0)**: Ollama's OpenAI API only returns text, so
no KV cache crosses between roles. To exercise the **actual Cache-to-Cache KV fusion** end-to-end —
SmolLM2-135M (Thinker/Sharer) latently injected into Qwen2.5-0.5B (Worker/Receiver) — the models must
run **in-process** (transformers). The `c2c` compose profile bundles a torch-enabled gateway image
that does exactly this, on CPU:

```bash
docker compose --profile c2c up -d --build --wait gateway-c2c   # first run builds (~2GB) + pulls models
# open http://localhost:8080/  → UNCHECK "Mock mode", CHECK "C2C (KV fusion)", set the gate, Run
docker compose --profile c2c down                               # stop
```

The debug UI's Worker step now shows a **`⚡ C2C fusion`** row (gate, aligned layers, share→recv token
lengths) — the heterogeneous KV edge, distinct from the text turns. Via the API, set
`extra_body.trinity_c2c=true` (and optional `trinity_c2c_gate`); `trinity_trace=true` includes the
`fusion` event.

> **Honest scope:** this wires the C2C *plumbing* through the gateway. The fuser ships **untrained**, so
> `gate=0` ≡ the Worker alone (safe; the plan rides the text channel) and **raising the gate degrades
> quality** until the fuser is trained (see [results](#verified-results-honest)). It demonstrates the
> mechanism live, not a quality gain.

Local (no Docker): `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`,
then `TRINITY_C2C=1 python -m trinity.gateway`. Verify the pieces directly:
`python -m trinity.c2c_edge` (the fused-KV generation gate-0 invariant) and `python -m trinity.p1`
(a full C2C run). C2C env knobs: `TRINITY_C2C` (`1` = default to the in-process fusion backend),
`TRINITY_C2C_GATE` (0..1), `TRINITY_C2C_MAX_NEW_TOKENS`, `SHARER_MODEL_ID` / `RECEIVER_MODEL_ID`.

### Training the fuser (so a checkpoint can be loaded)

The bundled fuser is **untrained** (gate>0 degrades — see [scope](#try-real-c2c--heterogeneous-kv-fusion-in-process-no-gpu)).
`trinity.c2c_train` trains the **engine-compatible** heterogeneous fuser (SmolLM→Qwen) on CPU and
saves a checkpoint the gateway can load. Two objectives:

```bash
# (a) distill — the gateway regime (recv==share): make a forced gate>0 track the receiver-alone
#     output. Default = on-policy/DAgger (re-roll the fused model's own trajectory -> fixes exposure
#     bias); reports free-generation token-match vs gate0. (--no-on-policy = teacher-forced only.)
python -m trinity.c2c_train --objective distill   --out checkpoints/distill.pt
# (b) relational — country->capital with held-out: evidence the C2C mechanism GENERALIZES
#     (reports learned vs gate0 vs shuffled logp on unseen countries).
python -m trinity.c2c_train --objective relational --out checkpoints/relational.pt

# load a checkpoint into the gateway/engine (validated against model ids + shapes + RoPE):
TRINITY_C2C_FUSER=checkpoints/distill.pt python -m trinity.gateway     # or set c2c.fuser_path
```

A checkpoint stores the model ids / shapes / RoPE hashes / `state_dict`, and the engine **refuses to
load a mismatched one** (falling back to the safe untrained path). In the `c2c` Docker profile, mount
`./checkpoints` and set `TRINITY_C2C_FUSER=/app/checkpoints/<name>.pt`.

> **Scope (honest):** a fuser is task-specific. `distill` (default = **on-policy / DAgger**: after a
> teacher-forced warm-up, re-roll the *fused* model's own greedy trajectory and match gate0's
> distribution at those visited states) drives **free-generation token-match vs gate0 from 18%→95%
> and eliminates the repetition collapse** on its 16 training contexts — fixing the exposure bias the
> earlier teacher-forced-only objective left (`--no-on-policy` hits teacher-forced match 97% but its
> *free* generation still degenerates). **But it does not generalize**: on held-out contexts and the
> gateway's gen_prompt regime, free generation still degrades — a data-scale frontier (16 contexts +
> a linear fuser overfit). `relational` only **uses the correct share content** (learned `-8.32` >
> shuffled `-8.71`) but does **not** beat the no-injection baseline (learned < gate0 `-6.07`).
> Neither makes gate>0 *improve* arbitrary code generation; a lower gate injects less and degrades
> less (gate→0 == receiver-alone). Model-free loss checks: `python -m trinity.c2c_train --selftest`.

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
- **Trainable + loadable through the gateway (M2)**: the heterogeneous fuser trains, saves a
  checkpoint (model-ids / shapes / RoPE hashes validated), and the engine loads it (gate>0 then uses
  the trained weights; a mismatched checkpoint is refused -> safe untrained gate-0 fallback).
- **`distill` + on-policy refinement (M2/M3)**: teacher-forcing on the receiver's gate=0 trajectory
  matches its next-token distribution (teacher-forced KL-vs-gate0 0.85→0.11, match 65→97%) but *free*
  generation still degenerated — exposure bias. Adding **on-policy / DAgger** distillation (re-roll
  the fused model's OWN greedy trajectory and match gate0's distribution at those visited states)
  drives **free-generation token-match vs gate0 18%→95% and removes the repetition collapse** on the
  16 training contexts. It does **not** generalize to held-out contexts or the gateway's gen_prompt
  regime (a linear fuser + 16 contexts overfit) — a data-scale frontier, like `relational`.
- **`relational` generalization (still WIP)**: heterogeneous country→capital with a held-out split —
  the trained fuser **uses the correct share content** (learned `-8.32` > shuffled `-8.71`) but does
  **not** beat the no-injection baseline (learned `-8.32` < gate0 `-6.07`). Consistent with prior
  runs; scaling data/compute is the next step. Neither objective improves arbitrary code generation
  (task-specific).

## Reference papers (sources of the ideas)

- Trinity: An Evolved LLM Coordinator — arXiv:2512.04695
- Cache-to-Cache: Direct Semantic Communication Between LLMs — arXiv:2510.03215
- Activated LoRA / Multi-Adapter KV-Cache Reuse — arXiv:2504.12397, 2512.17910

## License

[Apache License 2.0](LICENSE) — Copyright 2026 © Northern system service Co.,Ltd. 

See [NOTICE](NOTICE) for attribution, the unofficial-status statement, the source papers, and
the handling of referenced models. Licenses for the models and datasets you use are governed
separately by their respective providers.
