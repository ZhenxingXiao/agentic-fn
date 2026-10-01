"""Bounded executors for synchronous parallel fan-out."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from agentic_fn.core.errors import ConfigurationError

T = TypeVar("T")


@dataclass(slots=True)
class BoundedExecutor:
    """
    Thin wrapper over ``ThreadPoolExecutor`` with an explicit concurrency cap.

    Sync parallel execution is opt-in at the pipeline level; this helper ensures
    the underlying pool cannot grow without bound.
    """

    max_workers: int = 8
    _pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate worker configuration."""

        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be >= 1")

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Submit a zero-argument callable to the pool."""

        return self._ensure_pool().submit(fn)

    def map(self, fn: Callable[[T], object], items: list[T]) -> list[object]:
        """Map ``fn`` over ``items`` using the bounded pool, preserving order."""

        pool = self._ensure_pool()
        return list(pool.map(fn, items))

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut down the underlying pool if it was created."""

        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
            self._pool = None

    def _ensure_pool(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool."""

        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._pool
