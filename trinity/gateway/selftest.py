"""
Model-free self-test for the gateway (mock mode; no network / no GPU).

Run:  python -m trinity.gateway.selftest

Exercises the OpenAI-compatible endpoints and the debug SSE trace end-to-end using the
offline mock backend, and asserts the workflow ordering the debug UI relies on.
"""
from __future__ import annotations

import json
import os


def _events_from_sse(text: str) -> list[dict]:
    events = []
    for frame in text.split("\n\n"):
        data = ""
        for ln in frame.split("\n"):
            if ln.startswith("data:"):
                data += ln[5:].strip()
        if data:
            try:
                events.append(json.loads(data))
            except Exception:
                pass
    return events


def main() -> None:
    from fastapi.testclient import TestClient

    from trinity.gateway.app import app
    from trinity.mocks import build_mock_pool
    from trinity.p0 import Config, ScriptedCoordinator, run

    client = TestClient(app)

    # 1) health + models
    assert client.get("/healthz").json()["status"] == "ok"
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "trinity-p0", models
    print("[ok] /healthz and /v1/models")

    # 2) OpenAI chat completion (mock) — final artifact + trace
    r = client.post("/v1/chat/completions", json={
        "model": "trinity-p0",
        "messages": [{"role": "user", "content": "merge two sorted lists"}],
        "trinity_mock": True, "trinity_trace": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion", body
    content = body["choices"][0]["message"]["content"]
    assert content and "MOCK Worker" in content, content
    assert body["trinity"]["accepted"] is True, body["trinity"]
    assert isinstance(body["trinity"]["trace"], list) and body["trinity"]["trace"], body["trinity"]
    print("[ok] POST /v1/chat/completions (mock) -> accepted artifact + trace")

    # 2b) streaming variant yields chunks + [DONE]
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "trinity_mock": True, "stream": True,
    })
    assert r.status_code == 200, r.text
    assert "chat.completion.chunk" in r.text and "[DONE]" in r.text, r.text[:200]
    print("[ok] streaming /v1/chat/completions -> chunks + [DONE]")

    # 2c) the response is pure-OpenAI (no non-standard fields) unless trinity_trace is requested
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "trinity_mock": True,
    })
    assert "trinity" not in r.json(), r.json()
    print("[ok] response is pure-OpenAI without trinity_trace")

    # 3) debug SSE trace (mock, no delay) — assert the workflow ordering the UI relies on
    r = client.post("/debug/runs/stream", json={
        "query": "merge two sorted lists", "mock": True, "mock_delay": 0.0, "include_prompts": True,
    })
    assert r.status_code == 200, r.text
    events = _events_from_sse(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "final", types
    roles = [e.get("role") for e in events if e["type"] == "turn_end"]
    assert roles == ["thinker", "worker", "verifier", "worker", "verifier"], roles
    verdicts = [(e["verdict"], e["accepted"]) for e in events if e["type"] == "verdict"]
    assert verdicts == [("REVISE", False), ("ACCEPT", True)], verdicts
    final = next(e for e in events if e["type"] == "final")
    assert final["accepted"] is True, final
    ts = next(e for e in events if e["type"] == "turn_start")
    assert ts.get("system") and ts.get("user"), ts
    print("[ok] /debug/runs/stream -> Thinker->Worker->Verifier(REVISE->ACCEPT)->final")

    # 3b) include_prompts=False strips the prompt text
    r = client.post("/debug/runs/stream", json={
        "query": "x", "mock": True, "mock_delay": 0.0, "include_prompts": False,
    })
    ts2 = next(e for e in _events_from_sse(r.text) if e["type"] == "turn_start")
    assert "system" not in ts2 and "user" not in ts2, ts2
    print("[ok] include_prompts=False strips prompt text")

    # 4) p0.run() stays backward-compatible when nobody is listening
    res = run("x", ScriptedCoordinator(), build_mock_pool(0.0), Config(verbose=False))
    assert res["accepted"] is True and res["final"], res
    print("[ok] p0.run() backward-compatible (on_event=None)")

    # 5) resolve_mode truth table (explicit request flags + per-backend env defaults)
    from trinity.gateway import service
    saved = {k: os.environ.get(k) for k in ("TRINITY_C2C", "TRINITY_GATEWAY_MOCK")}
    try:
        os.environ.pop("TRINITY_C2C", None)
        os.environ.pop("TRINITY_GATEWAY_MOCK", None)
        rm = service.resolve_mode
        assert rm(None, None) == "text" and rm(True, None) == "mock" and rm(False, None) == "text"
        assert rm(None, True) == "c2c" and rm(None, False) == "text"
        assert rm(False, True) == "c2c" and rm(True, False) == "mock", "explicit-flag precedence"
        os.environ["TRINITY_C2C"] = "1"
        assert rm(None, None) == "c2c" and rm(True, None) == "mock" and rm(None, False) == "text"
        os.environ.pop("TRINITY_C2C", None)
        os.environ["TRINITY_GATEWAY_MOCK"] = "1"
        assert rm(None, None) == "mock" and rm(None, True) == "c2c"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    print("[ok] resolve_mode truth table (mock | text | c2c)")

    # 6) run_c2c event ordering with a model-free MockC2CEngine (no torch / no models)
    from trinity.mocks import build_mock_c2c_engine
    from trinity.p1 import run_c2c
    evs: list[dict] = []
    res_c2c = run_c2c("merge two sorted lists", build_mock_c2c_engine(),
                      Config(max_turns=6, verbose=False), on_event=evs.append)
    types = [e["type"] for e in evs]
    assert types[0] == "run_start" and types[-1] == "final", types
    assert evs[0].get("mode") == "c2c" and "gate" in evs[0], evs[0]
    te_roles = [e["role"] for e in evs if e["type"] == "turn_end"]
    assert te_roles == ["thinker", "worker", "verifier", "worker", "verifier"], te_roles
    fusions = [e for e in evs if e["type"] == "fusion"]
    assert len(fusions) == 2 and all(f["role"] == "worker" for f in fusions), fusions
    assert all({"gate", "aligned_layers", "share_len", "recv_len"} <= set(f) for f in fusions), fusions
    for f in fusions:   # each fusion sits inside a worker turn (between its turn_start and turn_end)
        before = [e for e in evs if e["type"] == "turn_start" and e["role"] == "worker" and e["seq"] < f["seq"]]
        after = [e for e in evs if e["type"] == "turn_end" and e["role"] == "worker" and e["seq"] > f["seq"]]
        assert before and after, f
    verdicts = [(e["verdict"], e["accepted"]) for e in evs if e["type"] == "verdict"]
    assert verdicts == [("REVISE", False), ("ACCEPT", True)], verdicts
    assert res_c2c["accepted"] is True and res_c2c["final"], res_c2c
    print("[ok] run_c2c -> Thinker->[fusion]->Worker->Verifier(REVISE->ACCEPT)->final (mock engine)")

    # 7) gateway c2c mode end-to-end via a monkeypatched singleton engine (still no models)
    service._c2c_engine = build_mock_c2c_engine()
    try:
        assert service.get_c2c_engine() is service._c2c_engine, "engine should be a cached singleton"
        r = client.post("/debug/runs/stream", json={
            "query": "merge two sorted lists", "c2c": True, "mock_delay": 0.0,
        })
        cev = _events_from_sse(r.text)
        assert cev and cev[0].get("mode") == "c2c", cev[:1]
        assert any(e["type"] == "fusion" for e in cev), [e["type"] for e in cev]
        # OpenAI endpoint with trinity_c2c -> artifact + a fusion event in the trace
        service._c2c_engine = build_mock_c2c_engine()        # fresh REVISE/ACCEPT counter
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "merge two sorted lists"}],
            "trinity_c2c": True, "trinity_trace": True,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert "MOCK Worker" in body["choices"][0]["message"]["content"], body
        assert any(e["type"] == "fusion" for e in body["trinity"]["trace"]), body["trinity"]
    finally:
        service._c2c_engine = None
    print("[ok] gateway c2c mode -> fusion event via /debug + /v1 (monkeypatched engine)")

    print("\n===== GATEWAY SELFTEST PASSED =====")


if __name__ == "__main__":
    main()
