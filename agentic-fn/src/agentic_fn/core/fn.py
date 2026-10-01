"""Decorators and helpers that mark capability vs effect functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

from agentic_fn.core.step import StepPolicy, StepSpec
from agentic_fn.type.enums import StepKind

F = TypeVar("F", bound=Callable[..., Any])

_KIND_ATTR = "__agentic_fn_kind__"
_IDEMPOTENT_ATTR = "__agentic_fn_idempotent__"


def _mark(fn: F, *, kind: StepKind, idempotent: bool) -> F:
    """Attach framework metadata attributes to ``fn``."""

    setattr(fn, _KIND_ATTR, kind)
    setattr(fn, _IDEMPOTENT_ATTR, idempotent)
    return fn


@overload
def capability(fn: F) -> F: ...


@overload
def capability(*, name: str | None = None) -> Callable[[F], F]: ...


def capability(
    fn: F | None = None,
    *,
    name: str | None = None,
) -> F | Callable[[F], F]:
    """
    Mark a callable as a capability (pure / local) function.

    Can be used as ``@capability`` or ``@capability()``.
    """

    def decorator(target: F) -> F:
        _ = name
        return _mark(target, kind=StepKind.CAPABILITY, idempotent=True)

    if fn is not None:
        return decorator(fn)
    return decorator


@overload
def effect(fn: F) -> F: ...


@overload
def effect(*, idempotent: bool = False, name: str | None = None) -> Callable[[F], F]: ...


def effect(
    fn: F | None = None,
    *,
    idempotent: bool = False,
    name: str | None = None,
) -> F | Callable[[F], F]:
    """
    Mark a callable as an effect (IO / external) function.

    Parameters
    ----------
    idempotent:
        When ``True``, the pipeline may automatically retry the effect.
    """

    def decorator(target: F) -> F:
        _ = name
        return _mark(target, kind=StepKind.EFFECT, idempotent=idempotent)

    if fn is not None:
        return decorator(fn)
    return decorator


def get_marked_kind(fn: Callable[..., Any]) -> StepKind | None:
    """Return the marked step kind when present."""

    value = getattr(fn, _KIND_ATTR, None)
    return value if isinstance(value, StepKind) else None


def is_marked_idempotent(fn: Callable[..., Any]) -> bool:
    """Return whether the callable was marked idempotent."""

    return bool(getattr(fn, _IDEMPOTENT_ATTR, False))


def is_coroutine_fn(fn: Callable[..., Any]) -> bool:
    """Return whether ``fn`` is an async function."""

    return inspect.iscoroutinefunction(fn)


def to_step_spec(
    name: str,
    fn: Callable[..., Any],
    *,
    kind: StepKind | None = None,
    policy: StepPolicy | None = None,
    tags: tuple[str, ...] = (),
) -> StepSpec:
    """
    Build a ``StepSpec`` from a callable, inferring kind/async when possible.

    Raises
    ------
    ValueError
        If kind cannot be inferred and was not provided explicitly.
    """

    resolved_kind = kind or get_marked_kind(fn)
    if resolved_kind is None:
        raise ValueError(
            f"Cannot infer step kind for {name!r}; use @capability/@effect or pass kind="
        )
    resolved_policy = policy or StepPolicy()
    if resolved_kind is StepKind.EFFECT and is_marked_idempotent(fn):
        resolved_policy = StepPolicy(
            timeout=resolved_policy.timeout,
            retry=resolved_policy.retry,
            on_error=resolved_policy.on_error,
            circuit_breaker=resolved_policy.circuit_breaker,
            idempotent=True,
            critical=resolved_policy.critical,
            fallback=resolved_policy.fallback,
        )
    return StepSpec(
        name=name,
        kind=resolved_kind,
        fn=fn,
        policy=resolved_policy,
        tags=tags,
        is_async=is_coroutine_fn(fn),
    )


def as_capability(
    name: str,
    fn: Callable[..., Any],
    **policy_kwargs: Any,
) -> StepSpec:
    """Convenience helper to wrap a sync/async capability callable as ``StepSpec``."""

    policy = StepPolicy(**policy_kwargs) if policy_kwargs else StepPolicy()
    return to_step_spec(name, fn, kind=StepKind.CAPABILITY, policy=policy)


def as_effect(
    name: str,
    fn: Callable[..., Any],
    **policy_kwargs: Any,
) -> StepSpec:
    """Convenience helper to wrap a sync/async effect callable as ``StepSpec``."""

    policy = StepPolicy(**policy_kwargs) if policy_kwargs else StepPolicy()
    return to_step_spec(name, fn, kind=StepKind.EFFECT, policy=policy)
