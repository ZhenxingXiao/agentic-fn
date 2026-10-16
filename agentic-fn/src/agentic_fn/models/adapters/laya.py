"""Laya decision-model adapter (HTTP JSON contract)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.content import json_ops
from agentic_fn.core.context import Context
from agentic_fn.models.adapters._http import post_json
from agentic_fn.models.errors import ModelCapabilityError, ModelPermanentError
from agentic_fn.models.kinds import CAP_DECIDE
from agentic_fn.models.protocols import SecretProvider
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import DecideResponse, ModelResponse, Usage


@dataclass(slots=True)
class LayaClient:
    """
    Decision model client for Laya-style APIs.

    Expected request body::

        {"model": "...", "features": {...}, "context": [...]}

    Expected response body::

        {"action": "...", "scores": {...}, "confidence": 0.9, "explanation": "..."}
    """

    base_url: str
    api_key_secret: str = "laya.api_key"
    provider: str = "laya"
    secrets: SecretProvider | None = None
    path: str = "/v1/decide"
    default_headers: Mapping[str, str] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({CAP_DECIDE}))

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Chat complete is not supported."""

        _ = request, ctx
        raise ModelCapabilityError("laya client does not implement complete()")

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Async complete stub."""

        return self.complete(request, ctx)

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Call Laya decide endpoint."""

        _ = ctx
        if CAP_DECIDE not in self.capabilities:
            raise ModelCapabilityError("laya decide capability disabled")
        physical = request.physical_model or "policy"
        features = json_ops.loads(request.features.body)
        payload: dict[str, Any] = {
            "model": physical,
            "features": features,
            "context": [m.to_dict() for m in request.context_messages],
            "params": request.params.to_dict(),
        }
        headers = {**dict(self.default_headers)}
        if self.secrets is not None:
            try:
                headers["Authorization"] = f"Bearer {self.secrets.get(self.api_key_secret)}"
            except KeyError:
                pass
        url = self.base_url.rstrip("/") + self.path
        data = post_json(url, payload, headers=headers, timeout_s=request.timeout_s or 30.0)
        return self._parse(data, request=request, physical=physical)

    async def adecide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Async decide."""

        return self.decide(request, ctx)

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        """Streaming unsupported."""

        _ = request, ctx
        raise ModelCapabilityError("laya client does not stream")

    def _parse(
        self,
        data: dict[str, Any],
        *,
        request: DecideRequest,
        physical: str,
    ) -> DecideResponse:
        """Normalize Laya JSON."""

        if not isinstance(data, dict):
            raise ModelPermanentError("laya response must be a JSON object")
        scores_raw = data.get("scores") or {}
        usage_raw = data.get("usage") or {}
        return DecideResponse(
            logical_model=request.model,
            provider=self.provider,
            physical_model=physical,
            action=data.get("action"),
            scores={str(k): float(v) for k, v in dict(scores_raw).items()},
            confidence=data.get("confidence"),
            explanation=data.get("explanation"),
            payload=json_ops.from_value(data),
            usage=Usage(
                input_tokens=usage_raw.get("input_tokens"),
                output_tokens=usage_raw.get("output_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
                cost=usage_raw.get("cost"),
            ),
            request_digest=request.digest(),
        )
