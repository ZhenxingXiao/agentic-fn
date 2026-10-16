"""Call policies for ModelRuntime."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_fn.models.errors import ModelTransientError
from agentic_fn.core.errors import CircuitOpenError, TimeoutError, TransientError


@dataclass(frozen=True, slots=True)
class ModelCallPolicy:
    """
    Defaults for model invocations.

    Failover across route candidates is preferred over hammering one candidate.
    """

    timeout_s: float = 60.0
    digest_includes_route: bool = False
    externalize_raw: bool = True
    consume_budget_calls: bool = True
    consume_budget_tokens: bool = True
    consume_budget_cost: bool = True
    same_candidate_retries: int = 0
    failover_on: tuple[type[BaseException], ...] = field(
        default_factory=lambda: (
            ModelTransientError,
            TransientError,
            TimeoutError,
            CircuitOpenError,
        )
    )
    record_responses: bool = True
    apply_output_filter: bool = True


DEFAULT_MODEL_CALL_POLICY = ModelCallPolicy()
