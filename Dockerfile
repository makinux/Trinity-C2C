# syntax=docker/dockerfile:1
# Local Trinity + Cache-to-Cache スタック（CPUアプリ像）。GPUの大型LLMは vLLM サーバ側(compose)が担当。
# Python 3.12 固定（ホストの 3.14 + torch 問題を回避）。社内SSL検査は pip --trusted-host で通過。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

# SSL検査環境でも pip が通るよう trusted-host を既定化（必要なら --build-arg で上書き）
ARG TRUSTED="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org"

# torch は CPU 専用 index から（GPUは使わない＝アプリ像は軽量）
RUN pip install $TRUSTED torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-datasets.txt ./
RUN pip install $TRUSTED -r requirements.txt

# 任意: HumanEval+/MBPP+ ローダ用（--build-arg WITH_DATASETS=1）
ARG WITH_DATASETS=0
RUN if [ "$WITH_DATASETS" = "1" ]; then pip install $TRUSTED -r requirements-datasets.txt; fi

COPY trinity/ ./trinity/
COPY config.yml ./
COPY scripts/ ./scripts/

# 既定はモデル不要のセルフテスト一括（ビルド健全性の確認に使える）
CMD ["sh", "scripts/selftest.sh"]
