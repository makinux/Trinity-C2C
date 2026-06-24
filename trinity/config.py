"""
trinity/config.py — config.yml loader (built-in defaults, robust)
============================================================
- By default reads ./config.yml (TRINITY_CONFIG env var for a different path).
- Works via DEFAULTS even without config.yml or pyyaml (self-tests need no config).
- *_URL env vars override base_url (backward compatible).
Usage:
  from trinity.config import CONFIG, get
  CONFIG["models"]["worker"]["model_id"]
  get("orchestration", "max_turns", 5)
"""
from __future__ import annotations

import copy
import os

# Defaults with the same structure as config.yml (works on these values even without the file/yaml)
DEFAULTS = {
    "models": {
        "thinker":  {"name": "GLM",          "base_url": "http://localhost:8001/v1", "model_id": "glm-4",
                     "api_key": "EMPTY", "temperature": 0.6, "max_tokens": 4096},
        "worker":   {"name": "Qwen3-Coder",  "base_url": "http://localhost:8002/v1", "model_id": "qwen3-coder",
                     "api_key": "EMPTY", "temperature": 0.6, "max_tokens": 4096},
        "verifier": {"name": "DeepSeek-R1",  "base_url": "http://localhost:8003/v1", "model_id": "deepseek-r1-distill",
                     "api_key": "EMPTY", "temperature": 0.6, "max_tokens": 4096},
    },
    "prompts": {
        "thinker": "You are the Thinker. Analyze the problem and return only high-level plans, decompositions, and critiques of existing solutions. Do not write code or the final solution.",
        "worker": "You are the Worker. Building on the plans and critiques so far, produce the concrete final solution (complete code/derivation/numbers).",
        "verifier": ("You are the Verifier. Check whether the latest Worker artifact correctly and completely satisfies the query. "
                     "Always output 'VERDICT: ACCEPT' or 'VERDICT: REVISE' on the last line. If REVISE, also state the specific fixes."),
    },
    "orchestration": {"max_turns": 5, "verbose": True},
    "coordinator": {"slm_model": "Qwen/Qwen3-0.6B", "featurizer": "qwen3", "mask_no_artifact": True},
    "training": {"budget": 8000, "sigma0": 0.3, "m_reps": 8, "seed": 0},
    "eval": {"trials": 1, "timeout": 10.0, "max_inputs": None},
    "c2c": {"sharer_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "receiver_model": "Qwen/Qwen2.5-0.5B-Instruct", "init_gate": 0.05, "tau": 1.0},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = path or os.getenv("TRINITY_CONFIG", "config.yml")
    if os.path.exists(p):
        try:
            import yaml
            with open(p, encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, user)
        except Exception as e:                          # no pyyaml / parse failure -> continue with defaults
            print(f"[config] failed to load {p}, using defaults: {e}")
    # Override base_url via *_URL env vars (backward compatible)
    for role, env in (("thinker", "THINKER_URL"), ("worker", "WORKER_URL"), ("verifier", "VERIFIER_URL")):
        if os.getenv(env):
            cfg["models"][role]["base_url"] = os.environ[env]
    # Prefer the <ROLE>_API_KEY env var for secrets (recommended; avoids plaintext YAML)
    for role in cfg.get("models", {}):
        key = os.getenv(f"{role.upper()}_API_KEY")
        if key:
            cfg["models"][role]["api_key"] = key
    # Override model_id via <ROLE>_MODEL_ID env vars (e.g. to match an Ollama tag like qwen2.5-coder:3b)
    for role in cfg.get("models", {}):
        mid = os.getenv(f"{role.upper()}_MODEL_ID")
        if mid:
            cfg["models"][role]["model_id"] = mid
    return cfg


CONFIG = load_config()


def get(section: str, key: str | None = None, default=None):
    s = CONFIG.get(section, {})
    return s if key is None else s.get(key, default)


if __name__ == "__main__":
    import json
    src = os.getenv("TRINITY_CONFIG", "config.yml")
    print(f"[config] source = {src} ({'found' if os.path.exists(src) else 'DEFAULTS'})")
    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
