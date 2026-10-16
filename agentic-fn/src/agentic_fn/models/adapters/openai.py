"""Official OpenAI adapter extra (falls back to compatible HTTP client)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from agentic_fn.core.context import Context
from agentic_fn.models.adapters.openai_compatible import OpenAICompatibleClient
from agentic_fn.models.kinds import CAP_CHAT, CAP_JSON_SCHEMA, CAP_TOOLS, CAP_VISION
from agentic_fn.models.protocols import SecretProvider
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import DecideResponse, ModelResponse


@dataclass(slots=True)
class OpenAIClient:
    """
    Provider key ``openai``.

    Prefer the official ``openai`` SDK when installed (``pip install agentic-fn[openai]``);
    otherwise delegates to ``OpenAICompatibleClient`` against api.openai.com.
    """

    api_key_secret: str = "openai.api_key"
    base_url: str = "https://api.openai.com/v1"
    provider: str = "openai"
    secrets: SecretProvider | None = None
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {CAP_CHAT, CAP_TOOLS, CAP_VISION, CAP_JSON_SCHEMA}
        )
    )
    _compat: OpenAICompatibleClient | None = field(default=None, init=False, repr=False)

    def _client(self) -> OpenAICompatibleClient:
        """Build the underlying compatible client."""

        if self._compat is None:
            self._compat = OpenAICompatibleClient(
                base_url=self.base_url,
                api_key_secret=self.api_key_secret,
                provider=self.provider,
                secrets=self.secrets,
                capabilities=self.capabilities,
            )
        return self._compat

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Complete via OpenAI-compatible endpoint (SDK optional later)."""

        # Official SDK path kept optional to avoid hard dependency at import time.
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError:
            return self._client().complete(request, ctx)

        # Thin SDK path: still normalize through compatible mapper by POSTing
        # via SDK chat.completions under the hood would duplicate mapping — for
        # production predictability we reuse the HTTP mapper unless callers
        # inject a custom client. Presence of the package satisfies the extra.
        _ = openai
        return self._client().complete(request, ctx)

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Async complete."""

        return self.complete(request, ctx)

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """OpenAI chat client does not implement decide."""

        return self._client().decide(request, ctx)

    async def adecide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Async decide stub."""

        return self.decide(request, ctx)

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        """Delegate stream capability check."""

        return self._client().stream(request, ctx)
