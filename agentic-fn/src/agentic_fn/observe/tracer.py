"""Tracing abstractions with a no-op default implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(slots=True)
class NoOpSpan:
    """Span implementation that discards all operations."""

    name: str
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        """Store an attribute in-memory without exporting it."""

        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        """Ignore exceptions in the no-op tracer."""

        _ = exc

    def end(self) -> None:
        """No-op end."""


@dataclass(frozen=True, slots=True)
class NoOpTracer:
    """Default tracer used when no OpenTelemetry (or other) backend is configured."""

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, str] | None = None,
    ) -> NoOpSpan:
        """Create a no-op span."""

        span = NoOpSpan(name=name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span
