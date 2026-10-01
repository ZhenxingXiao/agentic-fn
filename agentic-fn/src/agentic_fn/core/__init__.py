"""Core building blocks: state, context, steps, merge, and results."""

from agentic_fn.core.context import Context
from agentic_fn.core.errors import (
    BudgetExceeded,
    CancelledError,
    CircuitOpenError,
    ConfigurationError,
    FrameworkError,
    HarnessDeniedError,
    MergeConflictError,
    PermanentError,
    ReplayMissError,
    StateValidationError,
    TimeoutError,
    TransientError,
)
from agentic_fn.core.fn import (
    as_capability,
    as_effect,
    capability,
    effect,
    get_marked_kind,
    is_coroutine_fn,
    is_marked_idempotent,
    to_step_spec,
)
from agentic_fn.core.merge import MergeAll, MergeByKeys, ReduceMerge
from agentic_fn.core.result import PipelineResult, RunMetrics, StepRecord
from agentic_fn.core.retry import ExponentialBackoff, RetryPolicy, default_retry_policy
from agentic_fn.core.state import ERRORS_KEY, State, StateMeta, validate_state
from agentic_fn.core.step import StepPolicy, StepSpec

__all__ = [
    "ERRORS_KEY",
    "BudgetExceeded",
    "CancelledError",
    "CircuitOpenError",
    "ConfigurationError",
    "Context",
    "ExponentialBackoff",
    "FrameworkError",
    "HarnessDeniedError",
    "MergeAll",
    "MergeByKeys",
    "MergeConflictError",
    "PermanentError",
    "PipelineResult",
    "ReduceMerge",
    "ReplayMissError",
    "RetryPolicy",
    "RunMetrics",
    "State",
    "StateMeta",
    "StateValidationError",
    "StepPolicy",
    "StepRecord",
    "StepSpec",
    "TimeoutError",
    "TransientError",
    "as_capability",
    "as_effect",
    "capability",
    "default_retry_policy",
    "effect",
    "get_marked_kind",
    "is_coroutine_fn",
    "is_marked_idempotent",
    "to_step_spec",
    "validate_state",
]
