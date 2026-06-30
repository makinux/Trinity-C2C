"""
trinity/manifest.py — model / fuser manifests + ABI hashes (provenance)
=======================================================================
"Which model build and which fuser produced this KV?" is the question every safe KV reuse must
answer first. C2C KV is bound to a precise ABI: tokenizer, chat template, model revision, RoPE
config, engine/KV layout, dtype/quant, parallel layout — and, for fused Receiver KV, the fuser's
checkpoint, layer map and injection semantics. Mix two of these up and you silently corrupt
generation (the exact failure ``c2c_fuser_hetero.load_fuser_into`` already guards against).

This module makes that ABI a first-class, hashable object so the rest of the system (the event
log's ``model_manifest_id``, and the Phase-C ``kvstore`` ABI gate) can compare compatibility by a
single hash instead of ad-hoc field checks:

  - :func:`model_manifest` — build a manifest for a role from ``config.yml`` + an ``abi_hash``.
  - :func:`fuser_manifest` — read a fuser checkpoint's compatibility metadata (reusing the fields
    ``c2c_fuser_hetero.save_fuser`` already writes) + an ``abi_hash``.
  - :func:`compatibility_abi_hash` — the static (model [+ fuser]) ABI portion of a KV cache key.
    ``kvstore`` folds the dynamic per-entry parts (token-ids hash, positions, branch/epoch) on top.

``model_manifest`` / ``compatibility_abi_hash`` are pure-stdlib (model-free). ``fuser_manifest``
lazily imports ``torch`` only to deserialize a checkpoint's metadata; the self-test skips that part
when torch is unavailable, mirroring the gateway self-test's optional-dependency pattern.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from trinity.config import CONFIG

# Engine / KV-layout identity. Bump KV_LAYOUT_VERSION whenever the on-disk/in-memory KV byte
# layout changes in a way that makes older cached KV unsafe to reuse.
ENGINE = "hf-transformers"
KV_LAYOUT_VERSION = 1

# The fields that actually affect KV byte-compatibility (a difference in any of these = hard miss).
# Sampling params (temperature/max_tokens) change *outputs* but not KV *layout*, so they are part of
# the full manifest identity but NOT of the ABI hash used to gate KV reuse.
_MODEL_ABI_FIELDS = (
    "model_id", "model_revision", "tokenizer_hash", "chat_template_hash",
    "special_token_config", "rope_config", "engine", "kv_layout_version",
    "dtype", "quantization", "parallel_layout",
)
_FUSER_ABI_FIELDS = (
    "format_version", "arch_name", "fuser_class", "sharer_model", "receiver_model",
    "sharer_shape", "receiver_shape", "sh_inv_hash", "rc_inv_hash",
    # The trained weights ARE the fuser's behavior: a fused_receiver_kv blob is only reusable if the
    # SAME fuser would reproduce it. Two checkpoints with identical metadata but different weights
    # must therefore get different ABI hashes (fuser ABI drift, invariant 4).
    "weights_hash",
)

# Model ABI fields that genuinely pin KV byte-compatibility. If any of these is UNKNOWN, the
# manifest's ABI is *incomplete* and KV reuse must be disabled (see ``model_manifest.abi_complete``).
_MODEL_ABI_CRITICAL = ("tokenizer_hash", "chat_template_hash", "rope_config", "dtype")

# Sentinel for ABI fields not yet captured by the local OpenAI-compatible config. Recording them
# explicitly (rather than omitting) keeps the hash honest: "unknown" is a distinct ABI state, and
# it can be tightened later (e.g. once we read the tokenizer/chat-template from the served model).
UNKNOWN = "unknown"


def _hash(obj: Any, n: int = 16) -> str:
    """Deterministic short hash of a JSON-serializable object (sorted keys -> stable)."""
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def _abi_hash(manifest: dict, fields: tuple) -> str:
    return _hash({k: manifest.get(k, UNKNOWN) for k in fields}, n=32)


def _state_dict_hash(sd: dict, n: int = 16) -> str:
    """Stable hash of a fuser's trained weights (the bytes that determine what KV it produces)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode("utf-8"))
        v = sd[k]
        try:
            import numpy as np
            h.update(np.ascontiguousarray(v.detach().cpu().to_dense().float().numpy()).tobytes())
        except Exception:
            h.update(repr(v).encode("utf-8"))      # last-resort: still deterministic per-checkpoint
    return h.hexdigest()[:n]


# ============================================================
# 1. Model manifest
# ============================================================
def model_manifest(role: str, cfg: Optional[dict] = None) -> dict:
    """Build the ABI/provenance manifest for one role from config.

    Many true-ABI fields (tokenizer/chat-template/dtype/parallel layout) are not yet exposed by the
    local OpenAI-compatible endpoints, so they are recorded as ``UNKNOWN`` and contribute a stable
    (if conservative) ABI hash. ``manifest_id`` identifies the full config (incl. sampling);
    ``abi_hash`` identifies only KV-byte-compatibility.
    """
    cfg = cfg or CONFIG
    m = (cfg.get("models", {}).get(role, {}) or {})
    manifest = {
        "role": role,
        "name": m.get("name", role),
        "model_id": m.get("model_id", UNKNOWN),
        "base_url": m.get("base_url", UNKNOWN),
        "model_revision": m.get("model_revision", UNKNOWN),
        "tokenizer_hash": m.get("tokenizer_hash", UNKNOWN),
        "chat_template_hash": m.get("chat_template_hash", UNKNOWN),
        "special_token_config": m.get("special_token_config", UNKNOWN),
        "rope_config": m.get("rope_config", UNKNOWN),
        "engine": m.get("engine", ENGINE),
        "kv_layout_version": m.get("kv_layout_version", KV_LAYOUT_VERSION),
        "dtype": m.get("dtype", UNKNOWN),
        "quantization": m.get("quantization", UNKNOWN),
        "parallel_layout": m.get("parallel_layout", UNKNOWN),
        # sampling: part of full identity, not of the ABI hash
        "temperature": m.get("temperature", UNKNOWN),
        "max_tokens": m.get("max_tokens", UNKNOWN),
    }
    manifest["abi_hash"] = _abi_hash(manifest, _MODEL_ABI_FIELDS)
    manifest["manifest_id"] = _hash(manifest, n=16)
    # Honesty flag: with the local OpenAI-compatible config most ABI fields are UNKNOWN, so two
    # genuinely different model builds can share an abi_hash. KV reuse must be gated on this being
    # True (the kvstore ABI gate is only *safe* once these fields are populated from the served model).
    manifest["abi_complete"] = all(manifest.get(k, UNKNOWN) != UNKNOWN for k in _MODEL_ABI_CRITICAL)
    return manifest


def all_model_manifests(cfg: Optional[dict] = None) -> dict[str, dict]:
    """Manifest per role (the snapshot written to ``runs/<run_id>/manifest.json``)."""
    cfg = cfg or CONFIG
    return {role: model_manifest(role, cfg) for role in (cfg.get("models", {}) or {})}


def write_run_manifest(run_id: str, runs_dir: Optional[str] = None,
                       cfg: Optional[dict] = None) -> str:
    """Freeze the per-role manifests for a run to ``runs/<run_id>/manifest.json`` and return its path.

    Call once at run start; pairs with ``trinity.persist`` (which owns the same ``runs/<run_id>/``
    dir). Tags every event downstream via the per-role ``manifest_id``.
    """
    import os

    from trinity.persist import RUNS_DIR
    run_dir = os.path.join(runs_dir or RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    snapshot = {"run_id": run_id, "engine": ENGINE, "kv_layout_version": KV_LAYOUT_VERSION,
                "models": all_model_manifests(cfg)}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


# ============================================================
# 2. Fuser manifest (reuses the checkpoint metadata from c2c_fuser_hetero.save_fuser)
# ============================================================
def fuser_manifest(ckpt_path: str) -> dict:
    """Read a fuser checkpoint's compatibility metadata (no model load) + an ``abi_hash``.

    Lazily imports torch to deserialize; the stored fields (format_version / arch_name /
    sharer_model / receiver_model / shapes / inv_freq hashes) are exactly what
    ``load_fuser_into`` validates, so this manifest is the auditable provenance of a fusion.
    """
    import torch                       # lazy: keeps model_manifest path torch-free
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    manifest = {
        "ckpt_path": ckpt_path,
        "format_version": ckpt.get("format_version"),
        "fuser_version": ckpt.get("format_version"),
        "arch_name": ckpt.get("arch_name"),
        "fuser_class": ckpt.get("fuser_class"),
        "sharer_model": ckpt.get("sharer_model"),
        "receiver_model": ckpt.get("receiver_model"),
        "sharer_shape": ckpt.get("sharer_shape"),
        "receiver_shape": ckpt.get("receiver_shape"),
        "tau": ckpt.get("tau"),
        "sh_inv_hash": ckpt.get("sh_inv_hash"),
        "rc_inv_hash": ckpt.get("rc_inv_hash"),
        "weights_hash": _state_dict_hash(ckpt.get("state_dict", {})),
        "training": ckpt.get("training", {}),
    }
    manifest["abi_hash"] = _abi_hash(manifest, _FUSER_ABI_FIELDS)
    manifest["manifest_id"] = _hash(manifest, n=16)
    return manifest


# ============================================================
# 3. Compatibility ABI hash (the static portion of a KV cache key)
# ============================================================
def compatibility_abi_hash(model_manifest_obj: dict,
                           fuser_manifest_obj: Optional[dict] = None) -> str:
    """The model (+ optional fuser) ABI portion of a KV cache key.

    For a plain ``prefix_kv`` entry pass only the model manifest; for a ``fused_receiver_kv`` entry
    also pass the fuser manifest. ``kvstore`` folds the dynamic per-entry parts (token-ids hash,
    position span, branch/state_epoch) on top of this to form the full cache key — so a mismatch in
    *any* ABI field forces a hard miss before the dynamic parts are even compared.
    """
    parts = {"model_abi": model_manifest_obj.get("abi_hash", UNKNOWN)}
    if fuser_manifest_obj is not None:
        parts["fuser_abi"] = fuser_manifest_obj.get("abi_hash", UNKNOWN)
    return _hash(parts, n=32)


# ============================================================
# 4. Model-free self-test
# ============================================================
def _selftest() -> None:
    import copy

    # Determinism: same config -> same hashes.
    a = model_manifest("worker")
    b = model_manifest("worker")
    assert a["abi_hash"] == b["abi_hash"], "abi_hash not deterministic"
    assert a["manifest_id"] == b["manifest_id"], "manifest_id not deterministic"

    # An ABI-relevant change (tokenizer) moves abi_hash; a sampling-only change does not.
    cfg = copy.deepcopy(CONFIG)
    cfg["models"]["worker"]["tokenizer_hash"] = "abc123"
    c = model_manifest("worker", cfg)
    assert c["abi_hash"] != a["abi_hash"], "tokenizer change did not move abi_hash"

    cfg2 = copy.deepcopy(CONFIG)
    cfg2["models"]["worker"]["temperature"] = 0.99
    d = model_manifest("worker", cfg2)
    assert d["abi_hash"] == a["abi_hash"], "sampling change wrongly moved abi_hash"
    assert d["manifest_id"] != a["manifest_id"], "sampling change should move full manifest_id"

    # abi_complete is False under the default (metadata-less) config -> KV reuse must stay disabled;
    # populating the critical fields flips it True.
    assert a["abi_complete"] is False, "default config should report an incomplete ABI"
    cfg3 = copy.deepcopy(CONFIG)
    cfg3["models"]["worker"].update(tokenizer_hash="tk", chat_template_hash="ct",
                                    rope_config="r", dtype="bf16")
    assert model_manifest("worker", cfg3)["abi_complete"] is True, "complete ABI not detected"

    # Distinct roles -> distinct ABI (different model_id).
    assert model_manifest("thinker")["abi_hash"] != model_manifest("verifier")["abi_hash"], \
        "distinct roles share an abi_hash"

    # compatibility_abi_hash: model-only vs model+fuser differ; deterministic.
    fake_fuser = {"abi_hash": "fuserabi0001"}
    h_model = compatibility_abi_hash(a)
    h_fused = compatibility_abi_hash(a, fake_fuser)
    assert h_model != h_fused, "fused ABI hash collided with model-only"
    assert compatibility_abi_hash(a, fake_fuser) == h_fused, "compatibility hash not deterministic"

    # all_model_manifests covers every configured role.
    manifests = all_model_manifests()
    assert set(manifests) == set(CONFIG["models"]), "manifest snapshot missing a role"

    # write_run_manifest freezes a JSON snapshot under runs/<run_id>/.
    import os
    import shutil
    import tempfile
    tmp_runs = tempfile.mkdtemp(prefix="trinity-runmani-")
    try:
        path = write_run_manifest("run-mani-0001", runs_dir=tmp_runs)
        assert os.path.exists(path), "manifest.json not written"
        snap = json.load(open(path, encoding="utf-8"))
        assert set(snap["models"]) == set(CONFIG["models"]), "snapshot roles incomplete"
    finally:
        shutil.rmtree(tmp_runs, ignore_errors=True)

    # fuser_manifest: round-trip a real checkpoint if torch is available, else skip (optional dep).
    fuser_ok = "skipped (torch unavailable)"
    try:
        import torch  # noqa: F401
        from trinity.c2c_fuser_hetero import KVShape, TorchHeteroRoPEFuser, save_fuser
        from trinity.c2c_rope import rope_inv_freq
        import os
        import tempfile

        sh = KVShape(2, 8, 16)
        rh = KVShape(2, 16, 8)
        fuser = TorchHeteroRoPEFuser(sh, rh, rope_inv_freq(8, 10000.0), rope_inv_freq(16, 1e6))
        tmp = tempfile.mkdtemp(prefix="trinity-manifest-")
        try:
            ckpt = os.path.join(tmp, "f.pt")
            save_fuser(fuser, ckpt, sharer_model="s-x", receiver_model="r-y",
                       sharer_shape=sh, receiver_shape=rh, metadata={"objective": "selftest"})
            fm = fuser_manifest(ckpt)
            assert fm["sharer_model"] == "s-x" and fm["receiver_model"] == "r-y", "fuser meta wrong"
            assert fm["abi_hash"] and fm["manifest_id"], "fuser manifest missing hashes"
            assert compatibility_abi_hash(a, fm) != h_model, "real fuser ABI did not change cache key"

            # Blocker fix: two checkpoints with IDENTICAL metadata but DIFFERENT trained weights
            # must get different ABI hashes (the weights are the fuser's behavior).
            with torch.no_grad():
                for p in fuser.parameters():
                    p.add_(0.5)                       # perturb the weights, keep all metadata
            ckpt2 = os.path.join(tmp, "f2.pt")
            save_fuser(fuser, ckpt2, sharer_model="s-x", receiver_model="r-y",
                       sharer_shape=sh, receiver_shape=rh, metadata={"objective": "selftest"})
            fm2 = fuser_manifest(ckpt2)
            assert fm2["arch_name"] == fm["arch_name"] and fm2["sharer_model"] == fm["sharer_model"], \
                "metadata should be identical between the two checkpoints"
            assert fm2["weights_hash"] != fm["weights_hash"], "weights_hash did not change with weights"
            assert fm2["abi_hash"] != fm["abi_hash"], "different fuser weights collided on abi_hash (Blocker)"
            fuser_ok = f"ok (arch={fm['arch_name']}, weights in ABI)"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    except ImportError:
        pass

    print(f"[manifest] selftest OK - model abi/manifest deterministic, fuser_manifest {fuser_ok}")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
