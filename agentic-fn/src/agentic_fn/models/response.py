"""Model and decide response value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.content.codec import content_from_dict, content_to_dict, is_content_dict
from agentic_fn.content.model import Content
from agentic_fn.content.refs import ContentRef
from agentic_fn.models.kinds import (
    DECIDE_TYPE_MARKER,
    MODEL_SCHEMA_VERSION,
    MODEL_TYPE_MARKER,
    FinishReason,
)
from agentic_fn.models.message import Message


@dataclass(frozen=True, slots=True)
class Usage:
    """Token / cost accounting for a single model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    unknown: bool = False

    @staticmethod
    def unknown_usage() -> Usage:
        """Build a marker usage when the provider omitted counters."""

        return Usage(unknown=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize usage."""

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "unknown": self.unknown,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> Usage:
        """Deserialize usage."""

        if not data:
            return Usage.unknown_usage()
        return Usage(
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            total_tokens=data.get("total_tokens"),
            cost=data.get("cost"),
            unknown=bool(data.get("unknown") or False),
        )


@dataclass(frozen=True, slots=True)
class RouteAttemptTrace:
    """One attempted candidate in a route decision."""

    provider: str
    physical_model: str
    reason: str
    error_type: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class RouteTrace:
    """Explainability payload for how a request was routed."""

    logical_model: str
    strategy: str
    attempts: tuple[RouteAttemptTrace, ...] = ()
    selected_provider: str | None = None
    selected_physical_model: str | None = None
    filtered: tuple[str, ...] = ()  # human-readable filter reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialize route trace."""

        return {
            "logical_model": self.logical_model,
            "strategy": self.strategy,
            "attempts": [
                {
                    "provider": a.provider,
                    "physical_model": a.physical_model,
                    "reason": a.reason,
                    "error_type": a.error_type,
                    "latency_ms": a.latency_ms,
                }
                for a in self.attempts
            ],
            "selected_provider": self.selected_provider,
            "selected_physical_model": self.selected_physical_model,
            "filtered": list(self.filtered),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> RouteTrace:
        """Deserialize route trace."""

        if not data:
            return RouteTrace(logical_model="", strategy="")
        attempts = tuple(
            RouteAttemptTrace(
                provider=str(a["provider"]),
                physical_model=str(a["physical_model"]),
                reason=str(a.get("reason") or ""),
                error_type=a.get("error_type"),
                latency_ms=a.get("latency_ms"),
            )
            for a in data.get("attempts") or ()
        )
        return RouteTrace(
            logical_model=str(data.get("logical_model") or ""),
            strategy=str(data.get("strategy") or ""),
            attempts=attempts,
            selected_provider=data.get("selected_provider"),
            selected_physical_model=data.get("selected_physical_model"),
            filtered=tuple(data.get("filtered") or ()),
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized model completion result."""

    logical_model: str
    provider: str
    physical_model: str
    message: Message
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = field(default_factory=Usage.unknown_usage)
    route: RouteTrace = field(default_factory=lambda: RouteTrace(logical_model="", strategy=""))
    raw_ref: ContentRef | None = None
    latency_ms: float = 0.0
    request_digest: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for State."""

        return {
            "_type": MODEL_TYPE_MARKER,
            "schema_version": MODEL_SCHEMA_VERSION,
            "logical_model": self.logical_model,
            "provider": self.provider,
            "physical_model": self.physical_model,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "route": self.route.to_dict(),
            "raw_ref": self.raw_ref.to_dict() if self.raw_ref else None,
            "latency_ms": self.latency_ms,
            "request_digest": self.request_digest,
            "labels": dict(self.labels),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ModelResponse:
        """Deserialize a model response."""

        raw = data.get("raw_ref")
        return ModelResponse(
            logical_model=str(data["logical_model"]),
            provider=str(data["provider"]),
            physical_model=str(data["physical_model"]),
            message=Message.from_dict(data["message"]),
            finish_reason=FinishReason(str(data.get("finish_reason") or "unknown")),
            usage=Usage.from_dict(data.get("usage")),
            route=RouteTrace.from_dict(data.get("route")),
            raw_ref=ContentRef.from_dict(raw) if raw else None,
            latency_ms=float(data.get("latency_ms") or 0.0),
            request_digest=str(data.get("request_digest") or ""),
            labels=dict(data.get("labels") or {}),
        )


@dataclass(frozen=True, slots=True)
class DecideResponse:
    """Normalized decision / scoring result."""

    logical_model: str
    provider: str
    physical_model: str
    action: str | None = None
    scores: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    explanation: str | None = None
    payload: Content | None = None
    usage: Usage = field(default_factory=Usage.unknown_usage)
    route: RouteTrace = field(default_factory=lambda: RouteTrace(logical_model="", strategy=""))
    raw_ref: ContentRef | None = None
    latency_ms: float = 0.0
    request_digest: str = ""
    labels: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize decide response."""

        return {
            "_type": DECIDE_TYPE_MARKER,
            "schema_version": MODEL_SCHEMA_VERSION,
            "logical_model": self.logical_model,
            "provider": self.provider,
            "physical_model": self.physical_model,
            "action": self.action,
            "scores": dict(self.scores),
            "confidence": self.confidence,
            "explanation": self.explanation,
            "payload": content_to_dict(self.payload) if self.payload else None,
            "usage": self.usage.to_dict(),
            "route": self.route.to_dict(),
            "raw_ref": self.raw_ref.to_dict() if self.raw_ref else None,
            "latency_ms": self.latency_ms,
            "request_digest": self.request_digest,
            "labels": dict(self.labels),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> DecideResponse:
        """Deserialize decide response."""

        payload_raw = data.get("payload")
        payload: Content | None
        if payload_raw is None:
            payload = None
        elif is_content_dict(payload_raw):
            payload = content_from_dict(payload_raw)
        else:
            from agentic_fn.content import json_ops

            payload = json_ops.from_value(payload_raw)
        raw = data.get("raw_ref")
        return DecideResponse(
            logical_model=str(data["logical_model"]),
            provider=str(data["provider"]),
            physical_model=str(data["physical_model"]),
            action=data.get("action"),
            scores={str(k): float(v) for k, v in dict(data.get("scores") or {}).items()},
            confidence=data.get("confidence"),
            explanation=data.get("explanation"),
            payload=payload,
            usage=Usage.from_dict(data.get("usage")),
            route=RouteTrace.from_dict(data.get("route")),
            raw_ref=ContentRef.from_dict(raw) if raw else None,
            latency_ms=float(data.get("latency_ms") or 0.0),
            request_digest=str(data.get("request_digest") or ""),
            labels=dict(data.get("labels") or {}),
        )
