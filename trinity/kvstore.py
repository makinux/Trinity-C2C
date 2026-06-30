"""
trinity/kvstore.py — KV cache: namespaces + ABI gate + tiering + eviction (Phase C)
===================================================================================
An **optional accelerator**. KV / C2C latents are never state (see ``trinity.replay``): this store
only makes prefills cheaper to *re-create*; correctness must survive with the store switched off
(``KVStore`` absent -> every path is a text-only reconstruction). Everything here enforces the
invariants from the CC + Codex design consultation so a cache hit can never smuggle hidden latent
state into a result:

  - **Namespaces** (invariant 5): ``prefix_kv`` | ``agent_latent_kv`` | ``fused_receiver_kv`` |
    ``decode_tail_kv``. C2C-derived *fused* KV is never reused as an ordinary prefix cache.
  - **ABI gate** (invariant 4): a lookup hits only if the stored entry's ``compatibility_abi_hash``
    (from ``trinity.manifest``) exactly matches the caller's. Any model/tokenizer/template/fuser/
    layout difference -> the entry is a *stale* mismatch, hard miss, and is evicted.
  - **Branch quarantine** (invariant 6): rejected-branch fused KV is quarantined and is the first
    thing evicted; a lookup that would cross a rejected/accepted branch boundary misses.
  - **State epoch** (invariant 2): a REVISE bumps the epoch; fused/agent/decode KV from old epochs
    is invalidated (the "fresh reconstruction" rule, in cache form).
  - **Tiering**: GPU -> host RAM -> local SSD, behind an abstract :class:`KVBackend` so a real
    backend (LMCache / Mooncake / NIXL) can be dropped in later without touching call sites. Only a
    local in-memory + on-disk backend ships here.

Model-free: the store logic is exercised with synthetic blobs (bytes), so it runs under the repo's
``--selftest`` discipline with no torch and no GPU. The real C2C wiring is a documented, opt-in hook
(``c2c_edge._encode`` -> :meth:`KVStore.put`, ``_generate_from_fused`` -> :meth:`KVStore.lookup`);
with no store passed, behavior is byte-for-byte unchanged.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

# --- namespaces (prefer these constants over bare strings) ---
PREFIX_KV = "prefix_kv"               # stable Receiver prompt prefix (the only long-term-persistable class)
AGENT_LATENT_KV = "agent_latent_kv"   # Thinker/Verifier latent-extraction KV
FUSED_RECEIVER_KV = "fused_receiver_kv"  # C2C-fused Receiver KV (branch/epoch-scoped, never a prefix)
DECODE_TAIL_KV = "decode_tail_kv"     # generated decode tail (cheapest to drop)

NAMESPACES = (PREFIX_KV, AGENT_LATENT_KV, FUSED_RECEIVER_KV, DECODE_TAIL_KV)

# Eviction priority: lower rank = evicted first (Codex D-3). Quarantined entries jump to the front.
_EVICT_RANK = {
    FUSED_RECEIVER_KV: 1,             # rejected-branch fused first (see _evict_rank for the quarantine bump)
    DECODE_TAIL_KV: 2,
    AGENT_LATENT_KV: 3,
    PREFIX_KV: 6,                     # stable Receiver prefix evicted last
}

# Only stable prefix KV is reusable across branches (a "stable-prefix proof" = identical token span,
# branch-independent). Everything else is branch/epoch-scoped.
_BRANCH_AGNOSTIC = {PREFIX_KV}


def _hash(*parts: object, n: int = 32) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


# ============================================================
# 1. Records
# ============================================================
@dataclass
class KVCacheRecord:
    cache_key: str
    namespace: str
    compatibility_abi_hash: str
    token_ids_hash: str
    position_span: tuple
    branch_id: str
    state_epoch: int
    model_manifest_id: str = ""
    fuser_application_id: Optional[str] = None
    tier: str = "host_ram"
    blob_uri: str = ""
    byte_size: int = 0
    checksum: str = ""
    quarantined: bool = False
    seq: int = 0                       # insertion order: LRU/FIFO tiebreak within a rank


@dataclass
class KVMetrics:
    hits_by_namespace: dict = field(default_factory=lambda: {ns: 0 for ns in NAMESPACES})
    misses_by_namespace: dict = field(default_factory=lambda: {ns: 0 for ns in NAMESPACES})
    stale_kv_rejection_count: int = 0
    branch_crossing_cache_attempts: int = 0
    fuser_abi_mismatch_count: int = 0
    evictions: int = 0

    def hit_rate_by_namespace(self) -> dict:
        out = {}
        for ns in NAMESPACES:
            tot = self.hits_by_namespace[ns] + self.misses_by_namespace[ns]
            out[ns] = (self.hits_by_namespace[ns] / tot) if tot else None
        return out


# ============================================================
# 2. Backend (tiering substrate; swap for LMCache/Mooncake/NIXL later)
# ============================================================
class KVBackend:
    """Abstract blob substrate. Implementations move bytes between tiers; they hold no policy."""
    def put(self, cache_key: str, blob: bytes, tier: str) -> str: ...
    def get(self, cache_key: str) -> Optional[bytes]: ...
    def delete(self, cache_key: str) -> None: ...


class LocalKVBackend(KVBackend):
    """In-memory (host RAM) + optional local-SSD spill. The only backend that ships here."""
    def __init__(self, ssd_dir: Optional[str] = None):
        self._mem: dict[str, bytes] = {}
        self.ssd_dir = ssd_dir
        if ssd_dir:
            os.makedirs(ssd_dir, exist_ok=True)

    def _ssd_path(self, cache_key: str) -> str:
        return os.path.join(self.ssd_dir, cache_key)

    def put(self, cache_key: str, blob: bytes, tier: str) -> str:
        if tier == "local_ssd" and self.ssd_dir:
            path = self._ssd_path(cache_key)
            with open(path, "wb") as f:
                f.write(blob)
            return f"ssd://{path}"
        self._mem[cache_key] = blob
        return f"mem://{cache_key}"

    def get(self, cache_key: str) -> Optional[bytes]:
        if cache_key in self._mem:
            return self._mem[cache_key]
        if self.ssd_dir:
            path = self._ssd_path(cache_key)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
        return None

    def delete(self, cache_key: str) -> None:
        self._mem.pop(cache_key, None)
        if self.ssd_dir:
            try:
                os.remove(self._ssd_path(cache_key))
            except OSError:
                pass


# ============================================================
# 3. The store
# ============================================================
class KVStore:
    """Policy layer: namespaced index, ABI gate, branch quarantine, epoch invalidation, eviction.

    ``capacity`` is a per-store entry cap (a stand-in for a real byte budget); exceeding it evicts by
    :data:`_EVICT_RANK`. Pass ``backend=None`` for an in-memory store.
    """

    def __init__(self, backend: Optional[KVBackend] = None, capacity: int = 1024):
        self.backend = backend or LocalKVBackend()
        self.capacity = capacity
        self.metrics = KVMetrics()
        self._index: dict[str, KVCacheRecord] = {}   # logical_key -> current record for that slot
        self._content: dict[str, set] = {}           # content_key -> {logical_key,...} (cross-branch probe)
        self._seq = 0

    # --- key construction ---
    @staticmethod
    def content_key(namespace: str, token_ids_hash: str, position_span: tuple) -> str:
        """Branch/epoch-independent identity of the cached *content* (used to probe for crossings)."""
        return _hash(namespace, token_ids_hash, position_span)

    @staticmethod
    def logical_key(namespace: str, token_ids_hash: str, position_span: tuple,
                    branch_id: str, state_epoch: int) -> str:
        """The slot a lookup addresses. Prefix KV is branch/epoch-agnostic (stable-prefix proof);
        fused/agent/decode KV is scoped to (branch, epoch)."""
        if namespace in _BRANCH_AGNOSTIC:
            return _hash(namespace, token_ids_hash, position_span)
        return _hash(namespace, token_ids_hash, position_span, branch_id, state_epoch)

    # --- writes ---
    def put(self, namespace: str, blob: bytes, *, compatibility_abi_hash: str,
            token_ids_hash: str, position_span: tuple, branch_id: str, state_epoch: int,
            model_manifest_id: str = "", fuser_application_id: Optional[str] = None,
            tier: str = "host_ram") -> KVCacheRecord:
        assert namespace in NAMESPACES, f"unknown namespace {namespace!r}"
        lkey = self.logical_key(namespace, token_ids_hash, position_span, branch_id, state_epoch)
        cache_key = _hash(lkey, compatibility_abi_hash)        # ABI folded into the physical key
        self._seq += 1
        rec = KVCacheRecord(
            cache_key=cache_key, namespace=namespace,
            compatibility_abi_hash=compatibility_abi_hash, token_ids_hash=token_ids_hash,
            position_span=tuple(position_span), branch_id=branch_id, state_epoch=state_epoch,
            model_manifest_id=model_manifest_id, fuser_application_id=fuser_application_id,
            tier=tier, byte_size=len(blob), checksum=_hash(blob, n=16), seq=self._seq,
        )
        # Overwriting a logical slot (e.g. the ABI changed for the same content) must release the
        # previous blob, or the old backend object leaks — defeating the byte-budget the store exists
        # to enforce. Evict the prior record first unless it is byte-identical (same physical key).
        prior = self._index.get(lkey)
        if prior is not None and prior.cache_key != cache_key:
            self._evict_key(lkey)
        rec.blob_uri = self.backend.put(cache_key, blob, tier)
        self._index[lkey] = rec
        self._content.setdefault(
            self.content_key(namespace, token_ids_hash, position_span), set()).add(lkey)
        self._maybe_evict()
        return rec

    # --- reads (the ABI gate) ---
    def lookup(self, namespace: str, *, compatibility_abi_hash: str, token_ids_hash: str,
               position_span: tuple, branch_id: str, state_epoch: int,
               is_fuser_related: bool = False) -> Optional[bytes]:
        """Return the cached blob iff it passes the ABI gate, branch and epoch checks; else None.

        Side effects (the point of the gate): a stale ABI mismatch is *counted and evicted*; a
        branch-crossing attempt is counted and refused. A real hit returns bytes from the backend.
        """
        lkey = self.logical_key(namespace, token_ids_hash, position_span, branch_id, state_epoch)
        rec = self._index.get(lkey)

        if rec is not None:
            # Exact slot (same branch+epoch for non-prefix, same content for prefix).
            # ABI gate: identical slot, different ABI = stale. Hard miss + evict the stale entry.
            if rec.compatibility_abi_hash != compatibility_abi_hash:
                self.metrics.stale_kv_rejection_count += 1
                if is_fuser_related or rec.namespace in (FUSED_RECEIVER_KV, AGENT_LATENT_KV):
                    self.metrics.fuser_abi_mismatch_count += 1
                self._evict_key(lkey)
                return self._miss(namespace)
            if rec.quarantined:
                return self._miss(namespace)
            blob = self.backend.get(rec.cache_key)
            if blob is None:                          # tier lost the blob -> treat as miss
                return self._miss(namespace)
            self.metrics.hits_by_namespace[namespace] += 1
            return blob

        # No exact slot: probe same-content candidates from *other* branches/epochs. Finding one is
        # an explicit ABI-mismatch or branch-crossing attempt to refuse (invariants 4 & 6), not a
        # plain miss — count it so it shows up in the metrics.
        ckey = self.content_key(namespace, token_ids_hash, position_span)
        for cand_lkey in self._content.get(ckey, ()):  # prefix is always exact-matched above
            cand = self._index.get(cand_lkey)
            if cand is None:
                continue
            if cand.compatibility_abi_hash != compatibility_abi_hash:
                self.metrics.stale_kv_rejection_count += 1
                if is_fuser_related or cand.namespace in (FUSED_RECEIVER_KV, AGENT_LATENT_KV):
                    self.metrics.fuser_abi_mismatch_count += 1
                break
            if cand.branch_id != branch_id:
                self.metrics.branch_crossing_cache_attempts += 1
                break
            if cand.state_epoch != state_epoch:        # stale epoch (older draft of this branch)
                self.metrics.stale_kv_rejection_count += 1
                break
        return self._miss(namespace)

    def _miss(self, namespace: str) -> None:
        self.metrics.misses_by_namespace[namespace] += 1
        return None

    # --- invalidation policy ---
    def quarantine_branch(self, branch_id: str) -> int:
        """Quarantine all fused/agent/decode KV of a rejected branch (-> first to be evicted)."""
        n = 0
        for rec in self._index.values():
            if rec.branch_id == branch_id and rec.namespace != PREFIX_KV:
                rec.quarantined = True
                n += 1
        return n

    def invalidate_epoch(self, branch_id: str, keep_epoch: int) -> int:
        """On REVISE: drop fused/agent/decode KV from older epochs of a branch (fresh reconstruction)."""
        victims = [k for k, r in self._index.items()
                   if r.branch_id == branch_id and r.namespace != PREFIX_KV and r.state_epoch < keep_epoch]
        for k in victims:
            self._evict_key(k)
        return len(victims)

    # --- eviction ---
    def _evict_rank(self, rec: KVCacheRecord) -> tuple:
        # Quarantined entries are always evicted first (rank -1), then by namespace rank, then LRU (seq).
        base = -1 if rec.quarantined else _EVICT_RANK.get(rec.namespace, 5)
        return (base, rec.seq)

    def _maybe_evict(self) -> None:
        while len(self._index) > self.capacity:
            victim = min(self._index.items(), key=lambda kv: self._evict_rank(kv[1]))
            self._evict_key(victim[0])

    def _evict_key(self, lkey: str) -> None:
        rec = self._index.pop(lkey, None)
        if rec is not None:
            ckey = self.content_key(rec.namespace, rec.token_ids_hash, rec.position_span)
            slot = self._content.get(ckey)
            if slot is not None:
                slot.discard(lkey)
                if not slot:
                    self._content.pop(ckey, None)
            self.backend.delete(rec.cache_key)
            self.metrics.evictions += 1

    def __len__(self) -> int:
        return len(self._index)


# ============================================================
# 4. Model-free self-test
# ============================================================
def _selftest() -> None:
    import shutil
    import tempfile

    abi_a = "abi-model-A"
    abi_b = "abi-model-B"               # a different model build / fuser ABI
    span = (0, 32)
    tih = "tokids-deadbeef"

    tmp = tempfile.mkdtemp(prefix="trinity-kvstore-")
    try:
        store = KVStore(LocalKVBackend(ssd_dir=os.path.join(tmp, "ssd")), capacity=8)

        # 1. Round-trip hit under matching ABI.
        store.put(PREFIX_KV, b"prefix-bytes", compatibility_abi_hash=abi_a,
                  token_ids_hash=tih, position_span=span, branch_id="b0", state_epoch=0)
        hit = store.lookup(PREFIX_KV, compatibility_abi_hash=abi_a, token_ids_hash=tih,
                           position_span=span, branch_id="b0", state_epoch=0)
        assert hit == b"prefix-bytes", "matching-ABI lookup did not hit"

        # 2. ABI gate: same slot, different ABI = stale rejection + eviction (hard miss).
        miss = store.lookup(PREFIX_KV, compatibility_abi_hash=abi_b, token_ids_hash=tih,
                            position_span=span, branch_id="b0", state_epoch=0)
        assert miss is None, "stale ABI must not hit"
        assert store.metrics.stale_kv_rejection_count == 1, "stale rejection not counted"

        # 3. Namespace isolation: a fused entry is never reachable via the prefix namespace.
        store.put(FUSED_RECEIVER_KV, b"fused-bytes", compatibility_abi_hash=abi_a,
                  token_ids_hash=tih, position_span=span, branch_id="b1", state_epoch=0,
                  fuser_application_id="fa-1")
        as_prefix = store.lookup(PREFIX_KV, compatibility_abi_hash=abi_a, token_ids_hash=tih,
                                 position_span=span, branch_id="b1", state_epoch=0)
        assert as_prefix is None, "fused KV leaked into the prefix namespace"
        as_fused = store.lookup(FUSED_RECEIVER_KV, compatibility_abi_hash=abi_a, token_ids_hash=tih,
                                position_span=span, branch_id="b1", state_epoch=0)
        assert as_fused == b"fused-bytes", "fused KV not retrievable in its own namespace"

        # 4. Branch crossing: fused KV from b1 is not reusable from b2.
        cross = store.lookup(FUSED_RECEIVER_KV, compatibility_abi_hash=abi_a, token_ids_hash=tih,
                             position_span=span, branch_id="b2", state_epoch=0)
        assert cross is None, "fused KV crossed a branch boundary"
        assert store.metrics.branch_crossing_cache_attempts == 1, "branch crossing not counted"

        # 5. Branch quarantine: rejecting b1 quarantines its fused KV -> misses + first to evict.
        assert store.quarantine_branch("b1") == 1, "quarantine count wrong"
        q = store.lookup(FUSED_RECEIVER_KV, compatibility_abi_hash=abi_a, token_ids_hash=tih,
                         position_span=span, branch_id="b1", state_epoch=0)
        assert q is None, "quarantined KV must not hit"

        # 6. Epoch invalidation on REVISE: old-epoch fused KV is dropped, new-epoch survives.
        store.put(FUSED_RECEIVER_KV, b"fused-e0", compatibility_abi_hash=abi_a,
                  token_ids_hash="t-e", position_span=span, branch_id="b3", state_epoch=0)
        store.put(FUSED_RECEIVER_KV, b"fused-e1", compatibility_abi_hash=abi_a,
                  token_ids_hash="t-e", position_span=span, branch_id="b3", state_epoch=1)
        dropped = store.invalidate_epoch("b3", keep_epoch=1)
        assert dropped == 1, "old epoch not invalidated"
        assert store.lookup(FUSED_RECEIVER_KV, compatibility_abi_hash=abi_a, token_ids_hash="t-e",
                            position_span=span, branch_id="b3", state_epoch=1) == b"fused-e1", \
            "new epoch wrongly dropped"

        # 7. Eviction priority: at capacity, quarantined/fused/decode go before stable prefix.
        small = KVStore(LocalKVBackend(), capacity=2)
        small.put(PREFIX_KV, b"P", compatibility_abi_hash=abi_a, token_ids_hash="p",
                  position_span=span, branch_id="b", state_epoch=0)
        small.put(DECODE_TAIL_KV, b"D", compatibility_abi_hash=abi_a, token_ids_hash="d",
                  position_span=span, branch_id="b", state_epoch=0)
        small.put(FUSED_RECEIVER_KV, b"F", compatibility_abi_hash=abi_a, token_ids_hash="f",
                  position_span=span, branch_id="b", state_epoch=0)   # overflow -> evict lowest rank
        # The stable prefix must still be resident; a lower-rank entry was evicted instead.
        assert small.lookup(PREFIX_KV, compatibility_abi_hash=abi_a, token_ids_hash="p",
                            position_span=span, branch_id="b", state_epoch=0) == b"P", \
            "stable prefix was evicted before lower-priority KV"
        assert small.metrics.evictions >= 1, "no eviction happened at capacity"

        # 8. Overwriting a logical slot with a new ABI releases the old backend blob (no leak).
        leak = KVStore(LocalKVBackend(), capacity=8)
        r1 = leak.put(PREFIX_KV, b"old-blob", compatibility_abi_hash=abi_a, token_ids_hash="x",
                      position_span=span, branch_id="b", state_epoch=0)
        assert leak.backend.get(r1.cache_key) == b"old-blob", "first blob not stored"
        leak.put(PREFIX_KV, b"new-blob", compatibility_abi_hash=abi_b, token_ids_hash="x",
                 position_span=span, branch_id="b", state_epoch=0)
        assert leak.backend.get(r1.cache_key) is None, "stale blob leaked after overwrite"
        assert len(leak) == 1, "overwrite should keep exactly one record for the slot"

        rates = store.metrics.hit_rate_by_namespace()
        print(f"[kvstore] selftest OK - ABI gate, namespaces, branch quarantine, epoch invalidation,"
              f" eviction priority. hit_rate={ {k: v for k, v in rates.items() if v is not None} }")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
