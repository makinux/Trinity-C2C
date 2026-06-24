"""FastAPI app: OpenAI-compatible endpoints + debug SSE + static ChatUI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from trinity import __version__
from trinity.gateway import service
from trinity.gateway.schemas import ChatCompletionRequest, DebugRunRequest

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Trinity-C2C Gateway", version=__version__)
    # Dev convenience: a debug UI served from the same origin doesn't need this, but it
    # lets you point an external OpenAI client / page at the gateway without CORS pain.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "model": service.MODEL_ID,
                "mock_default": service.env_mock_default()}

    @app.get("/v1/models")
    async def v1_models():
        return service.list_models()

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: ChatCompletionRequest):
        query = service.flatten_messages(req.messages)
        mode = service.resolve_mode(req.trinity_mock, req.trinity_c2c)
        if req.stream:
            gen = service.stream_openai(query, mode=mode, max_turns=req.trinity_max_turns,
                                        gate=req.trinity_c2c_gate)
            return StreamingResponse(gen, media_type="text/event-stream",
                                     headers=service.SSE_HEADERS)
        result, trace = await service.run_collect(query, mode=mode,
                                                  max_turns=req.trinity_max_turns,
                                                  gate=req.trinity_c2c_gate)
        return JSONResponse(service.build_completion(
            query, result, trace, include_trace=bool(req.trinity_trace)))

    @app.post("/debug/runs/stream")
    async def debug_runs_stream(req: DebugRunRequest):
        # The debug UI sends explicit toggles; c2c overrides mock.
        mode = "c2c" if req.c2c else ("mock" if req.mock else "text")
        gen = service.stream_debug_sse(
            req.query, mode=mode, max_turns=req.max_turns, gate=req.gate,
            mock_delay=req.mock_delay, include_prompts=req.include_prompts,
        )
        return StreamingResponse(gen, media_type="text/event-stream",
                                 headers=service.SSE_HEADERS)

    # Serve the debug UI (index.html + its sibling assets) at the root. Mounted LAST so the
    # explicit API routes above always take precedence; html=True makes "/" return index.html.
    # The UI references its assets with RELATIVE paths (styles.css / app.js) so it renders
    # correctly whether served from "/" or opened as a sibling file.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
    return app


app = create_app()
