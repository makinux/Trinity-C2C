"""Pydantic request models for the gateway (OpenAI chat-completions + debug run)."""
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """An OpenAI chat message. ``content`` may be a string or a list of content parts."""
    model_config = ConfigDict(extra="allow")
    role: str = "user"
    content: Optional[Union[str, List[Any]]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI ``/v1/chat/completions`` request.

    ``extra="allow"`` tolerates standard OpenAI fields we don't act on (top_p, n, ...).
    Trinity-specific knobs arrive at the top level via the OpenAI SDK's ``extra_body``.
    """
    model_config = ConfigDict(extra="allow")
    model: str = "trinity-p0"
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # --- Trinity extras (pass via extra_body={...}) ---
    trinity_mock: Optional[bool] = None      # force mock backend on/off (default: env)
    trinity_trace: Optional[bool] = None     # include the full workflow trace in the response
    trinity_max_turns: Optional[int] = None  # override orchestration.max_turns


class DebugRunRequest(BaseModel):
    """Body for the debug UI's live-trace endpoint."""
    model_config = ConfigDict(extra="allow")
    query: str = ""
    max_turns: Optional[int] = None
    mock: bool = True               # default ON so the UI works fully offline
    mock_delay: float = 0.4         # per-turn delay so the live trace is visibly progressive
    include_prompts: bool = True    # include each role's system+user prompt in turn_start
