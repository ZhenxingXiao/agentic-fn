"""Model and decide request value objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from agentic_fn.content.codec import content_from_dict, content_to_dict, is_content_dict
from agentic_fn.content.model import Content
from agentic_fn.models.kinds import (
    CAP_JSON_SCHEMA,
    CAP_TOOLS,
    MODEL_SCHEMA_VERSION,
    REQUEST_TYPE_MARKER,
    Modality,
    ResponseFormatKind,
)
from agentic_fn.models.message import Message, ToolSpec, messages_require_capabilities


@dataclass(frozen=True, slots=True)
class ModelParams:
    """Sampling / generation parameters (vendor-agnostic subset)."""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize params."""

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> ModelParams:
        """Deserialize params."""

        if not data:
            return ModelParams()
        stop = data.get("stop") or ()
        return ModelParams(
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_tokens=data.get("max_tokens"),
            stop=tuple(stop),
            seed=data.get("seed"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """Requested response shape."""

    kind: ResponseFormatKind = ResponseFormatKind.TEXT
    json_schema: Mapping[str, Any] | None = None
    schema_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize response format."""

        return {
            "kind": self.kind.value,
            "json_schema": dict(self.json_schema) if self.json_schema else None,
            "schema_name": self.schema_name,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> ResponseFormat:
        """Deserialize response format."""

        if not data:
            return ResponseFormat()
        return ResponseFormat(
            kind=ResponseFormatKind(str(data.get("kind") or "text")),
            json_schema=dict(data["json_schema"]) if data.get("json_schema") else None,
            schema_name=data.get("schema_name"),
        )


@dataclass(frozen=True, slots=True)
class RoutingHints:
    """Optional per-request overrides applied after route strategy selection."""

    prefer_provider: str | None = None
    exclude_providers: tuple[str, ...] = ()
    sticky_key: str | None = None
    force_physical_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing hints."""

        return {
            "prefer_provider": self.prefer_provider,
            "exclude_providers": list(self.exclude_providers),
            "sticky_key": self.sticky_key,
            "force_physical_model": self.force_physical_model,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> RoutingHints | None:
        """Deserialize routing hints."""

        if not data:
            return None
        return RoutingHints(
            prefer_provider=data.get("prefer_provider"),
            exclude_providers=tuple(data.get("exclude_providers") or ()),
            sticky_key=data.get("sticky_key"),
            force_physical_model=data.get("force_physical_model"),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """
    Vendor-agnostic model invocation request.

    ``model`` is a *logical* name resolved by ``ModelRouter``.
    """

    model: str
    messages: tuple[Message, ...]
    modality: Modality = Modality.CHAT
    tools: tuple[ToolSpec, ...] = ()
    response_format: ResponseFormat = field(default_factory=ResponseFormat)
    params: ModelParams = field(default_factory=ModelParams)
    routing: RoutingHints | None = None
    idempotency_key: str | None = None
    timeout_s: float | None = None
    stream: bool = False
    labels: Mapping[str, str] = field(default_factory=dict)
    # Filled by Runtime after routing — adapters should read these.
    provider: str | None = None
    physical_model: str | None = None

    def required_capabilities(self) -> frozenset[str]:
        """Capabilities implied by this request."""

        needed = set(messages_require_capabilities(self.messages))
        if self.tools:
            needed.add(CAP_TOOLS)
        if self.response_format.kind is ResponseFormatKind.JSON_SCHEMA:
            needed.add(CAP_JSON_SCHEMA)
        if self.modality is Modality.DECIDE:
            from agentic_fn.models.kinds import CAP_DECIDE

            needed.add(CAP_DECIDE)
        if self.modality is Modality.MULTIMODAL:
            from agentic_fn.models.kinds import CAP_VISION

            needed.add(CAP_VISION)
        if self.stream:
            from agentic_fn.models.kinds import CAP_STREAM

            needed.add(CAP_STREAM)
        return frozenset(needed)

    def bind_route(self, *, provider: str, physical_model: str) -> ModelRequest:
        """Return a copy bound to a concrete provider/model."""

        return replace(self, provider=provider, physical_model=physical_model)

    def digest(self, *, include_route: bool = False) -> str:
        """Stable digest for replay / idempotency (secrets must never appear)."""

        payload = {
            "model": self.model,
            "modality": self.modality.value,
            "messages": [m.to_dict() for m in self.messages],
            "tools": [t.to_dict() for t in self.tools],
            "response_format": self.response_format.to_dict(),
            "params": self.params.to_dict(),
            "idempotency_key": self.idempotency_key,
            "stream": self.stream,
            "labels": dict(self.labels),
        }
        if include_route:
            payload["provider"] = self.provider
            payload["physical_model"] = self.physical_model
        blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request."""

        return {
            "_type": REQUEST_TYPE_MARKER,
            "schema_version": MODEL_SCHEMA_VERSION,
            "model": self.model,
            "modality": self.modality.value,
            "messages": [m.to_dict() for m in self.messages],
            "tools": [t.to_dict() for t in self.tools],
            "response_format": self.response_format.to_dict(),
            "params": self.params.to_dict(),
            "routing": self.routing.to_dict() if self.routing else None,
            "idempotency_key": self.idempotency_key,
            "timeout_s": self.timeout_s,
            "stream": self.stream,
            "labels": dict(self.labels),
            "provider": self.provider,
            "physical_model": self.physical_model,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ModelRequest:
        """Deserialize a request."""

        return ModelRequest(
            model=str(data["model"]),
            messages=tuple(Message.from_dict(m) for m in data.get("messages") or ()),
            modality=Modality(str(data.get("modality") or "chat")),
            tools=tuple(ToolSpec.from_dict(t) for t in data.get("tools") or ()),
            response_format=ResponseFormat.from_dict(data.get("response_format")),
            params=ModelParams.from_dict(data.get("params")),
            routing=RoutingHints.from_dict(data.get("routing")),
            idempotency_key=data.get("idempotency_key"),
            timeout_s=data.get("timeout_s"),
            stream=bool(data.get("stream") or False),
            labels=dict(data.get("labels") or {}),
            provider=data.get("provider"),
            physical_model=data.get("physical_model"),
        )

    @staticmethod
    def chat(
        model: str,
        *messages: Message,
        **kwargs: Any,
    ) -> ModelRequest:
        """Convenience constructor for chat modality."""

        kwargs.setdefault("modality", Modality.CHAT)
        return ModelRequest(
            model=model,
            messages=tuple(messages),
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class DecideRequest:
    """Structured decision / scoring request (e.g. Laya)."""

    model: str
    features: Content
    context_messages: tuple[Message, ...] = ()
    params: ModelParams = field(default_factory=ModelParams)
    routing: RoutingHints | None = None
    idempotency_key: str | None = None
    timeout_s: float | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    provider: str | None = None
    physical_model: str | None = None

    def required_capabilities(self) -> frozenset[str]:
        """Capabilities required for decide calls."""

        from agentic_fn.models.kinds import CAP_DECIDE

        needed = {CAP_DECIDE}
        needed |= set(messages_require_capabilities(self.context_messages))
        return frozenset(needed)

    def bind_route(self, *, provider: str, physical_model: str) -> DecideRequest:
        """Bind concrete route metadata."""

        return replace(self, provider=provider, physical_model=physical_model)

    def as_model_request(self) -> ModelRequest:
        """Project into a ModelRequest for shared routing machinery."""

        from agentic_fn.models.kinds import Modality as _M

        user = Message.user(self.features.body)
        return ModelRequest(
            model=self.model,
            messages=tuple(self.context_messages) + (user,),
            modality=_M.DECIDE,
            params=self.params,
            routing=self.routing,
            idempotency_key=self.idempotency_key,
            timeout_s=self.timeout_s,
            labels=dict(self.labels),
            provider=self.provider,
            physical_model=self.physical_model,
            response_format=ResponseFormat(kind=ResponseFormatKind.JSON_OBJECT),
        )

    def digest(self, *, include_route: bool = False) -> str:
        """Stable digest for decide requests."""

        return self.as_model_request().digest(include_route=include_route)

    def to_dict(self) -> dict[str, Any]:
        """Serialize decide request."""

        return {
            "model": self.model,
            "features": content_to_dict(self.features),
            "context_messages": [m.to_dict() for m in self.context_messages],
            "params": self.params.to_dict(),
            "routing": self.routing.to_dict() if self.routing else None,
            "idempotency_key": self.idempotency_key,
            "timeout_s": self.timeout_s,
            "labels": dict(self.labels),
            "provider": self.provider,
            "physical_model": self.physical_model,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> DecideRequest:
        """Deserialize decide request."""

        features_raw = data["features"]
        if is_content_dict(features_raw):
            features = content_from_dict(features_raw)
        elif isinstance(features_raw, Content):
            features = features_raw
        else:
            from agentic_fn.content import json_ops

            features = json_ops.from_value(features_raw)
        return DecideRequest(
            model=str(data["model"]),
            features=features,
            context_messages=tuple(
                Message.from_dict(m) for m in data.get("context_messages") or ()
            ),
            params=ModelParams.from_dict(data.get("params")),
            routing=RoutingHints.from_dict(data.get("routing")),
            idempotency_key=data.get("idempotency_key"),
            timeout_s=data.get("timeout_s"),
            labels=dict(data.get("labels") or {}),
            provider=data.get("provider"),
            physical_model=data.get("physical_model"),
        )
