"""Generic HTTP JSON decide adapter for custom decision services."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.content import json_ops
from agentic_fn.core.context import Context
from agentic_fn.models.adapters._http import post_json
from agentic_fn.models.errors import ModelCapabilityError
from agentic_fn.models.kinds import CAP_DECIDE
from agentic_fn.models.protocols import SecretProvider
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import DecideResponse, ModelResponse, Usage


@dataclass(slots=True)
class HttpJsonDecideClient:
    """
    Configurable decide client for arbitrary REST decision APIs.

    ``request_mapper`` / ``response_mapper`` let enterprises adapt proprietary
    schemas without forking the framework.
    """

    base_url: str
    provider: str = "http_json"
    path: str = "/decide"
    api_key_secret: str | None = None
    secrets: SecretProvider | None = None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({CAP_DECIDE}))
    request_mapper: Callable[[DecideRequest], dict[str, Any]] | None = None
    response_mapper: Callable[[dict[str, Any], DecideRequest], DecideResponse] | None = None

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        _ = request, ctx
        raise ModelCapabilityError(f"{self.provider} does not implement complete()")

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        return self.complete(request, ctx)

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        _ = ctx
        physical = request.physical_model or "default"
        if self.request_mapper is not None:
            payload = self.request_mapper(request)
        else:
            payload = {
                "model": physical,
                "features": json_ops.loads(request.features.body),
            }
        headers = dict(self.default_headers)
        if self.api_key_secret and self.secrets is not None:
            headers["Authorization"] = f"Bearer {self.secrets.get(self.api_key_secret)}"
        url = self.base_url.rstrip("/") + self.path
        data = post_json(url, payload, headers=headers, timeout_s=request.timeout_s or 30.0)
        if self.response_mapper is not None:
            return self.response_mapper(data, request)
        scores_raw = data.get("scores") or {}
        return DecideResponse(
            logical_model=request.model,
            provider=self.provider,
            physical_model=physical,
            action=data.get("action"),
            scores={str(k): float(v) for k, v in dict(scores_raw).items()},
            confidence=data.get("confidence"),
            explanation=data.get("explanation"),
            payload=json_ops.from_value(data),
            usage=Usage.from_dict(data.get("usage")),
            request_digest=request.digest(),
        )

    async def adecide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        return self.decide(request, ctx)

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        _ = request, ctx
        raise ModelCapabilityError(f"{self.provider} does not stream")
