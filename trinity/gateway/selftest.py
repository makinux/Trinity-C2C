"""
Model-free self-test for the gateway (mock mode; no network / no GPU).

Run:  python -m trinity.gateway.selftest

Exercises the OpenAI-compatible endpoints and the debug SSE trace end-to-end using the
offline mock backend, and asserts the workflow ordering the debug UI relies on.
"""
from __future__ import annotations

import json


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

    print("\n===== GATEWAY SELFTEST PASSED =====")


if __name__ == "__main__":
    main()
