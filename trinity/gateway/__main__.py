"""``python -m trinity.gateway`` — launch the gateway with uvicorn.

Env:
  TRINITY_GATEWAY_HOST  (default 127.0.0.1; use 0.0.0.0 in containers)
  TRINITY_GATEWAY_PORT  (default 8080)
  TRINITY_GATEWAY_MOCK  (1/true -> default to the offline mock backend)
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("TRINITY_GATEWAY_HOST", "127.0.0.1")
    port = int(os.getenv("TRINITY_GATEWAY_PORT", "8080"))
    uvicorn.run("trinity.gateway.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
