# Docker 環境（Local Trinity + Cache-to-Cache）

Python 3.12 固定の再現環境。ホストの Python 3.14 / torch 不在 / 社内SSL の問題を回避する。
GPUの大型ローカルLLM（GLM / Qwen-Coder / DeepSeek）は `gpu` プロファイルの vLLM サーバが担当し、
アプリ像は CPU 軽量（C2C実機検証は小型SLMで動く）。

## 1. ビルド & セルフテスト（モデル不要）
```bash
docker compose build
docker compose run --rm app            # = sh selftest.sh（全モジュールの --selftest）
```
期待: `===== ALL SELFTESTS PASSED =====`

## 2. C2C 実機の転移検証（小型SLM・CPUでOK）
```bash
docker compose run --rm app python trinity_c2c_realrun.py
# モデルDLは hf-cache ボリュームに永続化。社内SSLでDLが詰まる場合はホストの
# ~/.cache/huggingface を bind mount する: -v $HOME/.cache/huggingface:/root/.cache/huggingface
```
期待: gate↑で `ΔParis>0` かつ `ΔTokyo<0`（転移シグナル）。

## 3. 学習（sep-CMA-ES でコーディネータ θ を最適化）
```bash
docker compose run --rm app python trinity_train.py    # → coordinator_theta.npy
```

## 4. フル評価（要 GPU・vLLM 3サーバ）
```bash
cp .env.example .env        # モデルID等を調整
docker compose --profile gpu up -d --wait    # GLM/Qwen-Coder/DeepSeek を起動し healthy まで待つ
docker compose run --rm app python trinity_eval.py --trials 3 \
    --dataset humaneval --limit 20 --learned coordinator_theta.npy --featurizer qwen3
docker compose --profile gpu down
```
> GPU注意: 1サービス=1GPU確保。VRAM不足なら `WORKER_MODEL` 等を小型化、`--gpu-memory-utilization`/
> 量子化を調整、または一部サーバのみ起動。`evalplus` を使うなら `WITH_DATASETS=1` でビルド。

## 社内SSL検査環境について
- **ビルド(pip)**: Dockerfile は `--trusted-host` 既定で SSL 検査を回避するため、そのまま通る。
- **実行時(HF DL)**: コンテナ内は企業CAを持たないため、(a) ホストの HF キャッシュを bind mount するか、
  (b) 企業CAを `SSL_CERT_FILE` でマウント指定する。`truststore` は導入済み（OS証明書ストア利用）。

## ファイル
- `Dockerfile` / `requirements*.txt` / `selftest.sh` … アプリ像
- `docker-compose.yml` / `.env.example` … app + GPU vLLM サーバ
- `trinity_*.py` … 実装スタック（7+1ファイル）
