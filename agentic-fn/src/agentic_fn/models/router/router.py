"""Production ``DefaultModelRouter`` implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock

from agentic_fn.core.context import Context
from agentic_fn.models.errors import ModelConfigurationError, ModelRouteError
from agentic_fn.models.kinds import RouteStrategy
from agentic_fn.models.request import ModelRequest
from agentic_fn.models.router.config import (
    Candidate,
    RouteTable,
    validate_route_table,
)
from agentic_fn.models.router.decision import RouteAttempt, RouteDecision
from agentic_fn.models.router.health import ProviderHealthView
from agentic_fn.models.router.policies import (
    order_latency_aware,
    order_priority,
    order_sticky,
    order_weighted,
)


@dataclass(slots=True)
class DefaultModelRouter:
    """
    Logical-model router with capability / health filtering and failover chains.

    Thread-safe hot reload via ``reload``. In-flight ``resolve`` calls use a
    snapshot reference and are unaffected mid-flight.
    """

    table: RouteTable
    health: ProviderHealthView = field(default_factory=ProviderHealthView)
    known_providers: tuple[str, ...] = ()
    validate_on_init: bool = True
    _lock: RLock = field(default_factory=RLock, repr=False)

    def __post_init__(self) -> None:
        """Optionally validate the initial table."""

        if self.validate_on_init:
            validate_route_table(self.table, known_providers=self.known_providers or None)

    def resolve(self, request: ModelRequest, ctx: Context) -> RouteDecision:
        """Build an ordered attempt list for ``request``."""

        with self._lock:
            table = self.table
            health = self.health

        try:
            entry = table.get(request.model)
        except ModelConfigurationError as exc:
            raise ModelRouteError(str(exc), cause=exc) from exc

        required = request.required_capabilities()
        filtered_reasons: list[str] = []
        eligible: list[Candidate] = []

        for candidate in entry.candidates:
            if not candidate.enabled:
                filtered_reasons.append(
                    f"{candidate.provider}/{candidate.physical_model}:disabled"
                )
                continue
            missing = required - candidate.capabilities
            if missing:
                filtered_reasons.append(
                    f"{candidate.provider}/{candidate.physical_model}:"
                    f"missing_caps={sorted(missing)}"
                )
                continue
            breaker = candidate.resolved_breaker_key()
            if health.is_open(breaker):
                filtered_reasons.append(
                    f"{candidate.provider}/{candidate.physical_model}:circuit_open"
                )
                continue
            eligible.append(candidate)

        eligible = self._apply_hints(eligible, request, filtered_reasons)
        if not eligible:
            raise ModelRouteError(
                f"no eligible candidates for logical model {request.model!r}; "
                f"filtered={filtered_reasons}"
            )

        ordered = self._order(eligible, entry.strategy, request, ctx, health)
        max_attempts = max(1, entry.failover.max_attempts)
        ordered = ordered[:max_attempts]

        attempts = tuple(
            RouteAttempt(
                provider=c.provider,
                physical_model=c.physical_model,
                breaker_key=c.resolved_breaker_key(),
                reason=self._reason(entry.strategy, index=i),
                candidate=c,
            )
            for i, c in enumerate(ordered)
        )
        return RouteDecision(
            logical_model=request.model,
            strategy=entry.strategy.value,
            attempts=attempts,
            entry=entry,
            filtered=tuple(filtered_reasons),
            failover=entry.failover,
        )

    def reload(self, table: RouteTable) -> None:
        """Atomically replace the route table after validation."""

        validate_route_table(table, known_providers=self.known_providers or None)
        with self._lock:
            self.table = table

    def snapshot(self) -> RouteTable:
        """Return a shallow copy of the current table."""

        with self._lock:
            return RouteTable(
                version=self.table.version,
                entries=dict(self.table.entries),
            )

    def set_known_providers(self, providers: Sequence[str]) -> None:
        """Update provider allow-list used on subsequent reloads."""

        with self._lock:
            self.known_providers = tuple(providers)

    def _apply_hints(
        self,
        candidates: list[Candidate],
        request: ModelRequest,
        filtered_reasons: list[str],
    ) -> list[Candidate]:
        """Apply prefer/exclude/force hints."""

        hints = request.routing
        if hints is None:
            return candidates
        remaining = candidates
        if hints.exclude_providers:
            excluded = set(hints.exclude_providers)
            kept: list[Candidate] = []
            for candidate in remaining:
                if candidate.provider in excluded:
                    filtered_reasons.append(
                        f"{candidate.provider}/{candidate.physical_model}:excluded_by_hint"
                    )
                else:
                    kept.append(candidate)
            remaining = kept
        if hints.force_physical_model:
            forced = [
                c for c in remaining if c.physical_model == hints.force_physical_model
            ]
            if forced:
                remaining = forced
            else:
                filtered_reasons.append(
                    f"force_physical_model={hints.force_physical_model!r}:unmatched"
                )
        if hints.prefer_provider:
            preferred = [c for c in remaining if c.provider == hints.prefer_provider]
            others = [c for c in remaining if c.provider != hints.prefer_provider]
            remaining = preferred + others
        return remaining

    def _order(
        self,
        candidates: list[Candidate],
        strategy: RouteStrategy,
        request: ModelRequest,
        ctx: Context,
        health: ProviderHealthView,
    ) -> list[Candidate]:
        """Dispatch to strategy implementation."""

        if strategy is RouteStrategy.PRIORITY:
            return order_priority(candidates)
        if strategy is RouteStrategy.WEIGHTED:
            return order_weighted(candidates, ctx)
        if strategy is RouteStrategy.STICKY:
            return order_sticky(candidates, request)
        if strategy is RouteStrategy.LATENCY_AWARE:
            return order_latency_aware(candidates, health)
        raise ModelConfigurationError(f"unknown route strategy: {strategy!r}")

    @staticmethod
    def _reason(strategy: RouteStrategy, *, index: int) -> str:
        """Human-readable attempt reason."""

        if index == 0:
            return f"selected:{strategy.value}"
        return f"failover:{strategy.value}:#{index}"


def build_simple_table(
    mapping: Mapping[str, Sequence[tuple[str, str, Sequence[str]]]],
    *,
    modality: str = "chat",
    strategy: str = "priority",
) -> RouteTable:
    """
    Helper for tests: logical -> [(provider, physical, caps), ...].
    """

    from agentic_fn.models.kinds import Modality, RouteStrategy
    from agentic_fn.models.router.config import Candidate, RouteEntry

    entries = {}
    for logical, rows in mapping.items():
        candidates = tuple(
            Candidate(
                provider=provider,
                physical_model=physical,
                priority=10 + idx * 10,
                capabilities=frozenset(caps),
            )
            for idx, (provider, physical, caps) in enumerate(rows)
        )
        entries[logical] = RouteEntry(
            modality=Modality(modality),
            strategy=RouteStrategy(strategy),
            candidates=candidates,
        )
    return RouteTable(entries=entries)
