# syntax=docker/dockerfile:1
# Local Trinity + Cache-to-Cache stack (CPU app image). Large GPU LLMs are handled by the vLLM servers (compose).
# Pinned to Python 3.12 (avoids the host 3.14 + torch issues). Corporate SSL inspection is passed via pip --trusted-host.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

WORKDIR /app

# Default trusted-host so pip works even under SSL inspection (override via --build-arg if needed)
ARG TRUSTED="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org"

# Install torch from the CPU-only index (no GPU = lightweight app image)
RUN pip install $TRUSTED torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-datasets.txt ./
RUN pip install $TRUSTED -r requirements.txt

# Optional: for the HumanEval+/MBPP+ loader (--build-arg WITH_DATASETS=1)
ARG WITH_DATASETS=0
RUN if [ "$WITH_DATASETS" = "1" ]; then pip install $TRUSTED -r requirements-datasets.txt; fi

COPY trinity/ ./trinity/
COPY config.yml ./
COPY scripts/ ./scripts/

# Default is the full model-free self-test suite (useful as a build sanity check)
CMD ["sh", "scripts/selftest.sh"]
