"""Budget accounting for production-safe pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock

from agentic_fn.core.errors import BudgetExceeded


@dataclass(frozen=True, slots=True)
class BudgetView:
    """Read-only snapshot of remaining budget counters."""

    max_calls: int | None
    remaining_calls: int | None
    max_tokens: int | None
    remaining_tokens: int | None
    max_cost: float | None
    remaining_cost: float | None

    @property
    def exhausted(self) -> bool:
        """Return whether any finite budget dimension has reached zero."""

        if self.remaining_calls is not None and self.remaining_calls <= 0:
            return True
        if self.remaining_tokens is not None and self.remaining_tokens <= 0:
            return True
        if self.remaining_cost is not None and self.remaining_cost <= 0:
            return True
        return False


@dataclass(slots=True)
class Budget:
    """
    Mutable, thread-safe budget tracker shared through ``Context``.

    Any dimension left as ``None`` is treated as unlimited.
    """

    max_calls: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    _remaining_calls: int | None = field(init=False, default=None)
    _remaining_tokens: int | None = field(init=False, default=None)
    _remaining_cost: float | None = field(init=False, default=None)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize remaining counters from configured maxima."""

        self._remaining_calls = self.max_calls
        self._remaining_tokens = self.max_tokens
        self._remaining_cost = self.max_cost

    @property
    def exhausted(self) -> bool:
        """Return whether the budget is exhausted."""

        return self.view().exhausted

    def view(self) -> BudgetView:
        """Return a consistent read-only view of remaining resources."""

        with self._lock:
            return BudgetView(
                max_calls=self.max_calls,
                remaining_calls=self._remaining_calls,
                max_tokens=self.max_tokens,
                remaining_tokens=self._remaining_tokens,
                max_cost=self.max_cost,
                remaining_cost=self._remaining_cost,
            )

    def consume(
        self,
        *,
        calls: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> BudgetView:
        """
        Consume budget units and return the updated view.

        Raises
        ------
        BudgetExceeded
            If consumption would exceed any configured limit.
        """

        with self._lock:
            if calls and self._remaining_calls is not None:
                if self._remaining_calls < calls:
                    raise BudgetExceeded("call budget exceeded")
                self._remaining_calls -= calls
            if tokens and self._remaining_tokens is not None:
                if self._remaining_tokens < tokens:
                    raise BudgetExceeded("token budget exceeded")
                self._remaining_tokens -= tokens
            if cost and self._remaining_cost is not None:
                if self._remaining_cost < cost:
                    raise BudgetExceeded("cost budget exceeded")
                self._remaining_cost -= cost
            return BudgetView(
                max_calls=self.max_calls,
                remaining_calls=self._remaining_calls,
                max_tokens=self.max_tokens,
                remaining_tokens=self._remaining_tokens,
                max_cost=self.max_cost,
                remaining_cost=self._remaining_cost,
            )

    def ensure_available(self) -> None:
        """Raise ``BudgetExceeded`` when no remaining budget exists."""

        if self.exhausted:
            raise BudgetExceeded("budget exhausted")

    def copy(self) -> Budget:
        """Create an independent budget with the same limits and remaining values."""

        view = self.view()
        clone = Budget(
            max_calls=self.max_calls,
            max_tokens=self.max_tokens,
            max_cost=self.max_cost,
        )
        with clone._lock:
            clone._remaining_calls = view.remaining_calls
            clone._remaining_tokens = view.remaining_tokens
            clone._remaining_cost = view.remaining_cost
        return clone


def unlimited_budget() -> Budget:
    """Return a budget with all dimensions unlimited."""

    return Budget()
