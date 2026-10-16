"""Protocols for model clients, routers, secrets, and safety filters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentic_fn.core.context import Context
    from agentic_fn.models.request import DecideRequest, ModelRequest
    from agentic_fn.models.response import DecideResponse, ModelResponse
    from agentic_fn.models.router.config import RouteTable
    from agentic_fn.models.router.decision import RouteDecision


@runtime_checkable
class SecretProvider(Protocol):
    """Resolves secrets without placing them into State."""

    def get(self, key: str) -> str:
        """Return a secret value or raise KeyError."""


@runtime_checkable
class OutputFilter(Protocol):
    """Optional safety / policy filter applied to model outputs."""

    def check_response(self, response: ModelResponse, request: ModelRequest) -> None:
        """Raise ``ModelSafetyError`` when the response must be blocked."""

    def check_decide(self, response: DecideResponse, request: DecideRequest) -> None:
        """Raise ``ModelSafetyError`` when a decide response must be blocked."""


@runtime_checkable
class ModelClient(Protocol):
    """
    Vendor adapter contract.

    Implementations live in extras. Core only depends on this protocol.
    """

    @property
    def provider(self) -> str:
        """Stable provider key used in the client registry / route table."""

    @property
    def capabilities(self) -> frozenset[str]:
        """Capability tokens advertised by this client."""

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Execute a unary completion (chat / multimodal)."""

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Async completion; may delegate to a thread for sync SDKs."""

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Execute a decision / scoring call."""

    async def adecide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Async decide."""

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        """
        Stream partial responses.

        Final yielded item should be the complete normalized response.
        """


@runtime_checkable
class ModelRouter(Protocol):
    """Production traffic control plane for logical model names."""

    def resolve(self, request: ModelRequest, ctx: Context) -> RouteDecision:
        """Produce an ordered attempt list for ``request``."""

    def reload(self, table: RouteTable) -> None:
        """Atomically replace the route table (hot reload)."""

    def snapshot(self) -> RouteTable:
        """Return a copy of the current route table."""


@runtime_checkable
class CostModel(Protocol):
    """Estimates monetary cost from usage / model identity."""

    def estimate(
        self,
        *,
        provider: str,
        physical_model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        """Return estimated cost in account currency, or None if unknown."""


class EnvSecretProvider:
    """Simple secret provider backed by environment variables."""

    def __init__(self, *, prefix: str = "", mapping: Mapping[str, str] | None = None) -> None:
        """Optional prefix and explicit mapping for tests."""

        self._prefix = prefix
        self._mapping = dict(mapping or {})

    def get(self, key: str) -> str:
        """Resolve ``prefix+key`` from mapping, then ``os.environ``."""

        import os

        if key in self._mapping:
            return self._mapping[key]
        env_key = f"{self._prefix}{key}" if self._prefix else key
        # Also try common uppercase forms: openai.api_key -> OPENAI_API_KEY
        candidates = [env_key, env_key.upper(), env_key.replace(".", "_").upper()]
        for candidate in candidates:
            value = os.environ.get(candidate)
            if value:
                return value
        raise KeyError(key)
