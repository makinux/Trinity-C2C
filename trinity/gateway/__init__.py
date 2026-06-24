"""
trinity.gateway — OpenAI-compatible API gateway + workflow-trace debug UI
=========================================================================
Fronts the Trinity orchestration (``trinity.p0.run``) with:
  - ``GET  /v1/models``            — lists the single virtual model ``trinity-p0``
  - ``POST /v1/chat/completions``  — OpenAI-compatible (non-streaming + SSE streaming)
  - ``POST /debug/runs/stream``    — live SSE trace of the workflow (Coordinator + roles)
  - ``GET  /``                     — the debug ChatUI (static, no build step)

Run it with::

    python -m trinity.gateway          # http://127.0.0.1:8080

Set ``TRINITY_GATEWAY_MOCK=1`` to default to the offline mock backend (no vLLM/GPU needed).
"""
from trinity.gateway.app import app, create_app

__all__ = ["app", "create_app"]
