"""Model routing subsystem."""

from agentic_fn.models.router.config import (
    Candidate,
    FailoverPolicy,
    RouteEntry,
    RouteTable,
    validate_route_table,
)
from agentic_fn.models.router.decision import RouteAttempt, RouteDecision
from agentic_fn.models.router.health import ProviderHealthView
from agentic_fn.models.router.router import DefaultModelRouter, build_simple_table

__all__ = [
    "Candidate",
    "DefaultModelRouter",
    "FailoverPolicy",
    "ProviderHealthView",
    "RouteAttempt",
    "RouteDecision",
    "RouteEntry",
    "RouteTable",
    "build_simple_table",
    "validate_route_table",
]
