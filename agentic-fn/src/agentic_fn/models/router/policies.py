"""Candidate ordering strategies for the model router."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from agentic_fn.core.context import Context
from agentic_fn.models.request import ModelRequest
from agentic_fn.models.router.config import Candidate
from agentic_fn.models.router.health import ProviderHealthView


def order_priority(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Stable sort by ascending priority, then provider/model name."""

    return sorted(
        candidates,
        key=lambda c: (c.priority, c.provider, c.physical_model),
    )


def order_weighted(
    candidates: Sequence[Candidate],
    ctx: Context,
) -> list[Candidate]:
    """
    Weighted random shuffle using context RNG.

    Returns a full permutation biased by weight (higher weight earlier).
    """

    remaining = list(candidates)
    ordered: list[Candidate] = []
    while remaining:
        total = sum(max(0.0, c.weight) for c in remaining)
        if total <= 0:
            ordered.extend(remaining)
            break
        pick = ctx.rng.random() * total
        acc = 0.0
        chosen_idx = len(remaining) - 1
        for idx, candidate in enumerate(remaining):
            acc += max(0.0, candidate.weight)
            if pick <= acc:
                chosen_idx = idx
                break
        ordered.append(remaining.pop(chosen_idx))
    return ordered


def order_sticky(
    candidates: Sequence[Candidate],
    request: ModelRequest,
) -> list[Candidate]:
    """
    Sticky routing: hash sticky_key (or run labels) to prefer one candidate.

    Remaining candidates follow priority order as failover.
    """

    if not candidates:
        return []
    key = ""
    if request.routing and request.routing.sticky_key:
        key = request.routing.sticky_key
    else:
        key = request.labels.get("sticky_key") or request.labels.get("user_id") or ""
    if not key:
        return order_priority(candidates)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(candidates)
    primary = candidates[idx]
    rest = order_priority([c for c in candidates if c is not primary])
    return [primary, *rest]


def order_latency_aware(
    candidates: Sequence[Candidate],
    health: ProviderHealthView,
) -> list[Candidate]:
    """
    Prefer lower EWMA latency; unknown latency sorts after known healthy ones.

    Open circuits should already be filtered out before calling this.
    """

    def score(candidate: Candidate) -> tuple[int, float, int, str]:
        key = candidate.resolved_breaker_key()
        latency = health.latency_ms(key)
        # 0 = known latency first, 1 = unknown
        known = 0 if latency is not None else 1
        return (known, latency if latency is not None else 1e12, candidate.priority, candidate.provider)

    return sorted(candidates, key=score)
