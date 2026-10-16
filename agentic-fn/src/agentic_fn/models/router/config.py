"""Route table configuration models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from agentic_fn.models.errors import ModelConfigurationError
from agentic_fn.models.kinds import Modality, RouteStrategy
from agentic_fn.models.request import ModelParams


@dataclass(frozen=True, slots=True)
class FailoverPolicy:
    """Controls when and how deeply to failover across candidates."""

    on: tuple[str, ...] = ("transient", "timeout", "circuit_open")
    max_attempts: int = 3
    same_candidate_retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize failover policy."""

        return {
            "on": list(self.on),
            "max_attempts": self.max_attempts,
            "same_candidate_retries": self.same_candidate_retries,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any] | None) -> FailoverPolicy:
        """Deserialize failover policy."""

        if not data:
            return FailoverPolicy()
        return FailoverPolicy(
            on=tuple(data.get("on") or ("transient", "timeout", "circuit_open")),
            max_attempts=int(data.get("max_attempts") or 3),
            same_candidate_retries=int(data.get("same_candidate_retries") or 0),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    """One concrete provider/model option for a logical route."""

    provider: str
    physical_model: str
    weight: float = 1.0
    priority: int = 100
    capabilities: frozenset[str] = field(default_factory=frozenset)
    enabled: bool = True
    cost_class: str | None = None
    breaker_key: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def resolved_breaker_key(self) -> str:
        """Circuit breaker key for this candidate."""

        return self.breaker_key or f"{self.provider}:{self.physical_model}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize candidate."""

        return {
            "provider": self.provider,
            "physical_model": self.physical_model,
            "weight": self.weight,
            "priority": self.priority,
            "capabilities": sorted(self.capabilities),
            "enabled": self.enabled,
            "cost_class": self.cost_class,
            "breaker_key": self.breaker_key,
            "labels": dict(self.labels),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Candidate:
        """Deserialize candidate."""

        caps = data.get("capabilities") or ()
        return Candidate(
            provider=str(data["provider"]),
            physical_model=str(data["physical_model"]),
            weight=float(data.get("weight") or 1.0),
            priority=int(data.get("priority") or 100),
            capabilities=frozenset(str(c) for c in caps),
            enabled=bool(data.get("enabled", True)),
            cost_class=data.get("cost_class"),
            breaker_key=data.get("breaker_key"),
            labels=dict(data.get("labels") or {}),
        )


@dataclass(frozen=True, slots=True)
class RouteEntry:
    """Routing configuration for one logical model name."""

    modality: Modality
    strategy: RouteStrategy = RouteStrategy.PRIORITY
    candidates: tuple[Candidate, ...] = ()
    failover: FailoverPolicy = field(default_factory=FailoverPolicy)
    defaults: ModelParams = field(default_factory=ModelParams)

    def to_dict(self) -> dict[str, Any]:
        """Serialize route entry."""

        return {
            "modality": self.modality.value,
            "strategy": self.strategy.value,
            "candidates": [c.to_dict() for c in self.candidates],
            "failover": self.failover.to_dict(),
            "defaults": self.defaults.to_dict(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> RouteEntry:
        """Deserialize route entry."""

        return RouteEntry(
            modality=Modality(str(data.get("modality") or "chat")),
            strategy=RouteStrategy(str(data.get("strategy") or "priority")),
            candidates=tuple(Candidate.from_dict(c) for c in data.get("candidates") or ()),
            failover=FailoverPolicy.from_dict(data.get("failover")),
            defaults=ModelParams.from_dict(data.get("defaults")),
        )


@dataclass(frozen=True, slots=True)
class RouteTable:
    """Versioned map of logical model name -> route entry."""

    version: str = "1"
    entries: Mapping[str, RouteEntry] = field(default_factory=dict)

    def get(self, logical_model: str) -> RouteEntry:
        """Return an entry or raise."""

        try:
            return self.entries[logical_model]
        except KeyError as exc:
            raise ModelConfigurationError(
                f"unknown logical model: {logical_model!r}"
            ) from exc

    def with_entry(self, name: str, entry: RouteEntry) -> RouteTable:
        """Return a copy with one entry upserted."""

        merged = dict(self.entries)
        merged[name] = entry
        return replace(self, entries=merged)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the table."""

        return {
            "version": self.version,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> RouteTable:
        """Deserialize a table."""

        entries_raw = data.get("entries") or {}
        return RouteTable(
            version=str(data.get("version") or "1"),
            entries={
                str(name): RouteEntry.from_dict(entry)
                for name, entry in dict(entries_raw).items()
            },
        )

    @staticmethod
    def from_mapping(entries: Mapping[str, RouteEntry], *, version: str = "1") -> RouteTable:
        """Build from an already-typed mapping."""

        return RouteTable(version=version, entries=dict(entries))


def validate_route_table(
    table: RouteTable,
    *,
    known_providers: Sequence[str] | None = None,
) -> None:
    """
    Fail-fast validation used at startup and on hot reload.

    Raises ``ModelConfigurationError`` when the table is unusable.
    """

    if not table.entries:
        raise ModelConfigurationError("route table has no entries")
    providers = set(known_providers or ())
    for name, entry in table.entries.items():
        if not name.strip():
            raise ModelConfigurationError("logical model name must be non-empty")
        enabled = [c for c in entry.candidates if c.enabled]
        if not enabled:
            raise ModelConfigurationError(
                f"route {name!r} has no enabled candidates"
            )
        for candidate in enabled:
            if not candidate.provider or not candidate.physical_model:
                raise ModelConfigurationError(
                    f"route {name!r} has candidate with empty provider/model"
                )
            if candidate.weight < 0:
                raise ModelConfigurationError(
                    f"route {name!r} candidate weight must be >= 0"
                )
            if providers and candidate.provider not in providers:
                raise ModelConfigurationError(
                    f"route {name!r} references unknown provider {candidate.provider!r}"
                )
