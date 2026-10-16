"""Output safety filter hooks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agentic_fn.models.errors import ModelSafetyError
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import DecideResponse, ModelResponse


@dataclass(slots=True)
class DenySubstringFilter:
    """
    Blocks responses containing configured substrings (case-insensitive).

    Production deployments should inject stronger classifiers; this ships as a
    deterministic, dependency-free baseline hook.
    """

    denied: Sequence[str] = ()
    check_decide_enabled: bool = True

    def check_response(self, response: ModelResponse, request: ModelRequest) -> None:
        """Raise when assistant text contains a denied substring."""

        _ = request
        text = response.message.text().lower()
        for token in self.denied:
            if token and token.lower() in text:
                raise ModelSafetyError(f"output blocked by deny filter: {token!r}")

    def check_decide(self, response: DecideResponse, request: DecideRequest) -> None:
        """Optionally block decide explanations / actions."""

        _ = request
        if not self.check_decide_enabled:
            return
        hay = " ".join(
            part
            for part in (response.action, response.explanation)
            if part
        ).lower()
        for token in self.denied:
            if token and token.lower() in hay:
                raise ModelSafetyError(f"decide output blocked by deny filter: {token!r}")


@dataclass(slots=True)
class CompositeOutputFilter:
    """Fan-out to multiple filters."""

    filters: tuple[object, ...] = ()

    def check_response(self, response: ModelResponse, request: ModelRequest) -> None:
        """Run all filters."""

        for filt in self.filters:
            filt.check_response(response, request)  # type: ignore[attr-defined]

    def check_decide(self, response: DecideResponse, request: DecideRequest) -> None:
        """Run all decide filters."""

        for filt in self.filters:
            filt.check_decide(response, request)  # type: ignore[attr-defined]


@dataclass(slots=True)
class NoOpOutputFilter:
    """Pass-through filter."""

    def check_response(self, response: ModelResponse, request: ModelRequest) -> None:
        _ = response, request

    def check_decide(self, response: DecideResponse, request: DecideRequest) -> None:
        _ = response, request
