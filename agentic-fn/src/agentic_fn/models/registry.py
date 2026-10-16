"""Client registry mapping provider keys to ModelClient instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from agentic_fn.models.errors import ModelConfigurationError
from agentic_fn.models.protocols import ModelClient


@dataclass(slots=True)
class ClientRegistry:
    """Thread-safe registry of provider -> ModelClient."""

    _clients: dict[str, ModelClient] = field(default_factory=dict, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def register(self, provider: str, client: ModelClient, *, replace: bool = False) -> None:
        """Register a client under ``provider``."""

        if not provider:
            raise ModelConfigurationError("provider key must be non-empty")
        with self._lock:
            if provider in self._clients and not replace:
                raise ModelConfigurationError(f"provider already registered: {provider!r}")
            # Prefer client's declared provider when present.
            declared = getattr(client, "provider", provider)
            if declared != provider:
                # Allow alias registration but keep explicit key.
                pass
            self._clients[provider] = client

    def get(self, provider: str) -> ModelClient:
        """Return a client or raise."""

        with self._lock:
            try:
                return self._clients[provider]
            except KeyError as exc:
                raise ModelConfigurationError(
                    f"no ModelClient registered for provider {provider!r}"
                ) from exc

    def providers(self) -> tuple[str, ...]:
        """Return registered provider keys."""

        with self._lock:
            return tuple(sorted(self._clients))

    def unregister(self, provider: str) -> None:
        """Remove a provider if present."""

        with self._lock:
            self._clients.pop(provider, None)
