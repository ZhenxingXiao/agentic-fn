"""Event hook bus that isolates callback failures from the main run path."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agentic_fn.observe.events import Event
from agentic_fn.type.aliases import AsyncOnEvent, OnEvent

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HookBus:
    """Synchronous/asynchronous event fan-out with failure isolation."""

    sync_handlers: list[OnEvent] = field(default_factory=list)
    async_handlers: list[AsyncOnEvent] = field(default_factory=list)

    def add(self, handler: OnEvent) -> None:
        """Register a synchronous event handler."""

        self.sync_handlers.append(handler)

    def add_async(self, handler: AsyncOnEvent) -> None:
        """Register an asynchronous event handler."""

        self.async_handlers.append(handler)

    def emit(self, event: Event) -> None:
        """Emit an event to all synchronous handlers, swallowing handler errors."""

        for handler in self.sync_handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - callbacks must not break the run
                _LOGGER.exception("sync event handler failed for %s", event.kind)

    async def emit_async(self, event: Event) -> None:
        """Emit an event to sync and async handlers, swallowing handler errors."""

        self.emit(event)
        for handler in self.async_handlers:
            try:
                await handler(event)
            except Exception:  # noqa: BLE001 - callbacks must not break the run
                _LOGGER.exception("async event handler failed for %s", event.kind)
