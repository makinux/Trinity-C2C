# Docker environment (Local Trinity + Cache-to-Cache)

A reproducible environment pinned to Python 3.12. It avoids host-side problems (Python 3.14 /
missing torch / corporate SSL). Large local GPU LLMs (GLM / Qwen-Coder / DeepSeek) are handled
by the vLLM servers in the `gpu` profile, while the app image stays CPU-light (on-device C2C
validation runs on a small SLM).

## 1. Build & self-tests (no model needed)
```bash
docker compose build
docker compose run --rm app            # = sh selftest.sh (--selftest for every module)
```
Expected: `===== ALL SELFTESTS PASSED =====`

## 2. On-device C2C transfer validation (small SLM; CPU is fine)
```bash
docker compose run --rm app python trinity_c2c_realrun.py
# Model downloads persist in the hf-cache volume. If downloads stall under corporate SSL,
# bind-mount the host's ~/.cache/huggingface: -v $HOME/.cache/huggingface:/root/.cache/huggingface
```
Expected: as the gate rises, `dParis>0` and `dTokyo<0` (transfer signal).

## 3. Training (optimize the coordinator theta with sep-CMA-ES)
```bash
docker compose run --rm app python trinity_train.py    # -> coordinator_theta.npy
```

## 4. Full evaluation (requires GPU; 3 vLLM servers)
```bash
cp .env.example .env        # adjust model IDs etc.
docker compose --profile gpu up -d --wait    # start GLM/Qwen-Coder/DeepSeek and wait until healthy
docker compose run --rm app python trinity_eval.py --trials 3 \
    --dataset humaneval --limit 20 --learned coordinator_theta.npy --featurizer qwen3
docker compose --profile gpu down
```
> GPU note: 1 service = 1 GPU reserved. If VRAM is short, use smaller `WORKER_MODEL` etc.,
> adjust `--gpu-memory-utilization` / quantization, or start only some servers. To use
> `evalplus`, build with `WITH_DATASETS=1`.

## About corporate SSL-inspection environments
- **Build (pip)**: the Dockerfile defaults to `--trusted-host`, so it bypasses SSL inspection and works as-is.
- **Runtime (HF download)**: the container has no corporate CA, so either (a) bind-mount the host's
  HF cache, or (b) point `SSL_CERT_FILE` at a mounted corporate CA. `truststore` is already
  installed (it uses the OS certificate store).

## Files
- `Dockerfile` / `requirements*.txt` / `selftest.sh` ... app image
- `docker-compose.yml` / `.env.example` ... app + GPU vLLM servers
- `trinity_*.py` ... the implementation stack (7+1 files)
