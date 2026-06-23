"""
trinity/config.py — config.yml ローダ（デフォルト内蔵・堅牢）
============================================================
- 既定では ./config.yml を読む（環境変数 TRINITY_CONFIG で別パス）。
- config.yml や pyyaml が無くても DEFAULTS で動作（セルフテストは設定不要）。
- *_URL 環境変数で base_url を上書き（後方互換）。
使い方:
  from trinity.config import CONFIG, get
  CONFIG["models"]["worker"]["model_id"]
  get("orchestration", "max_turns", 5)
"""
from __future__ import annotations

import copy
import os

# config.yml と同一構造の既定値（ファイル/yaml が無くてもこの値で動く）
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
        "thinker": "あなたはThinker。問題を分析し、高レベルの計画・分解・既存解への批評のみを返す。コードや最終解は書かない。",
        "worker": "あなたはWorker。これまでの計画と批評を踏まえ、最終解（完全なコード/導出/数値）を具体的に作る。",
        "verifier": ("あなたはVerifier。最新のWorker成果物がクエリを正しく・完全に満たすか点検する。"
                     "最終行に必ず 'VERDICT: ACCEPT' か 'VERDICT: REVISE' を出力。REVISEなら具体的修正点も述べる。"),
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
        except Exception as e:                          # pyyaml無し/パース失敗 → 既定で続行
            print(f"[config] {p} 読み込み失敗、デフォルト使用: {e}")
    # *_URL 環境変数で base_url を上書き（後方互換）
    for role, env in (("thinker", "THINKER_URL"), ("worker", "WORKER_URL"), ("verifier", "VERIFIER_URL")):
        if os.getenv(env):
            cfg["models"][role]["base_url"] = os.environ[env]
    # 秘密鍵は <ROLE>_API_KEY 環境変数を優先（YAML平文を避ける推奨運用）
    for role in cfg.get("models", {}):
        key = os.getenv(f"{role.upper()}_API_KEY")
        if key:
            cfg["models"][role]["api_key"] = key
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
