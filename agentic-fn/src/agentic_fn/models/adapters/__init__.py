"""Optional vendor adapters (import lazily; SDKs are extras)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "HttpJsonDecideClient",
    "LayaClient",
    "OpenAIClient",
    "OpenAICompatibleClient",
]


def __getattr__(name: str) -> Any:
    """Lazy export adapters."""

    if name == "OpenAICompatibleClient":
        from agentic_fn.models.adapters.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient
    if name == "OpenAIClient":
        from agentic_fn.models.adapters.openai import OpenAIClient

        return OpenAIClient
    if name == "LayaClient":
        from agentic_fn.models.adapters.laya import LayaClient

        return LayaClient
    if name == "HttpJsonDecideClient":
        from agentic_fn.models.adapters.http_json import HttpJsonDecideClient

        return HttpJsonDecideClient
    raise AttributeError(name)
