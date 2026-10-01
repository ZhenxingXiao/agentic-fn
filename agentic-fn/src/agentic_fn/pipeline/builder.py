"""Fluent builders for sync and async pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agentic_fn.core.fn import is_coroutine_fn
from agentic_fn.core.merge import MergeAll
from agentic_fn.core.retry import RetryPolicy
from agentic_fn.core.step import StepPolicy, StepSpec
from agentic_fn.pipeline.async_ import AsyncPipeline
from agentic_fn.pipeline.defaults import PipelineDefaults
from agentic_fn.pipeline.nodes import (
    Branch,
    Loop,
    MapNode,
    Node,
    Parallel,
    Seq,
    branch,
    loop,
    map_node,
    parallel,
    seq,
)
from agentic_fn.pipeline.sync import SyncPipeline
from agentic_fn.type.aliases import BodyBuilderFn, PredicateFn
from agentic_fn.type.enums import ErrorAction, StepKind
from agentic_fn.type.protocols import MergeStrategy


def _is_node(value: object) -> bool:
    """Return whether ``value`` is a pipeline AST node."""

    return isinstance(value, (StepSpec, Seq, Parallel, Branch, Loop, MapNode))


class PipelineBodyBuilder:
    """
    Helper used to assemble a sequential body for loop/map/nested segments.

    Methods return ``self`` to support fluent chaining.
    """

    def __init__(self, name: str = "body") -> None:
        """Create an empty body builder."""

        self._name = name
        self._nodes: list[Node] = []

    def capability(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        on_error: ErrorAction = ErrorAction.FAIL,
        tags: Sequence[str] = (),
    ) -> PipelineBodyBuilder:
        """Append a capability step."""

        self._nodes.append(
            StepSpec(
                name=name,
                kind=StepKind.CAPABILITY,
                fn=fn,
                policy=StepPolicy(timeout=timeout, retry=retry, on_error=on_error),
                tags=tuple(tags),
                is_async=is_coroutine_fn(fn),
            )
        )
        return self

    def effect(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        on_error: ErrorAction = ErrorAction.FAIL,
        idempotent: bool = False,
        circuit_breaker: str | None = None,
        tags: Sequence[str] = (),
    ) -> PipelineBodyBuilder:
        """Append an effect step."""

        self._nodes.append(
            StepSpec(
                name=name,
                kind=StepKind.EFFECT,
                fn=fn,
                policy=StepPolicy(
                    timeout=timeout,
                    retry=retry,
                    on_error=on_error,
                    idempotent=idempotent,
                    circuit_breaker=circuit_breaker,
                ),
                tags=tuple(tags),
                is_async=is_coroutine_fn(fn),
            )
        )
        return self

    def node(self, value: Node) -> PipelineBodyBuilder:
        """Append an arbitrary node."""

        self._nodes.append(value)
        return self

    def build_node(self) -> Node:
        """Materialize the body as a single node (seq when multiple)."""

        if not self._nodes:
            raise ValueError("body builder is empty")
        if len(self._nodes) == 1:
            return self._nodes[0]
        return seq(self._name, *self._nodes)


class _PipelineBuilderBase:
    """Shared fluent construction logic for sync/async pipelines."""

    def __init__(self, *, defaults: PipelineDefaults) -> None:
        """Initialize builder state."""

        self._defaults = defaults
        self._nodes: list[Node] = []

    def capability(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        on_error: ErrorAction = ErrorAction.FAIL,
        tags: Sequence[str] = (),
    ) -> _PipelineBuilderBase:
        """Append a capability step to the root sequence."""

        self._nodes.append(
            StepSpec(
                name=name,
                kind=StepKind.CAPABILITY,
                fn=fn,
                policy=StepPolicy(timeout=timeout, retry=retry, on_error=on_error),
                tags=tuple(tags),
                is_async=is_coroutine_fn(fn),
            )
        )
        return self

    def effect(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
        on_error: ErrorAction = ErrorAction.FAIL,
        idempotent: bool = False,
        circuit_breaker: str | None = None,
        tags: Sequence[str] = (),
    ) -> _PipelineBuilderBase:
        """Append an effect step to the root sequence."""

        self._nodes.append(
            StepSpec(
                name=name,
                kind=StepKind.EFFECT,
                fn=fn,
                policy=StepPolicy(
                    timeout=timeout,
                    retry=retry,
                    on_error=on_error,
                    idempotent=idempotent,
                    circuit_breaker=circuit_breaker,
                ),
                tags=tuple(tags),
                is_async=is_coroutine_fn(fn),
            )
        )
        return self

    def node(self, value: Node) -> _PipelineBuilderBase:
        """Append an arbitrary pre-built node."""

        self._nodes.append(value)
        return self

    def parallel(
        self,
        name: str,
        branches: Mapping[str, Node | BodyBuilderFn],
        *,
        merge: MergeStrategy | None = None,
        enable_sync_parallel: bool = False,
    ) -> _PipelineBuilderBase:
        """Append a parallel fan-out node."""

        resolved = {
            branch_name: self._resolve_body(f"{name}.{branch_name}", value)
            for branch_name, value in branches.items()
        }
        self._nodes.append(
            parallel(
                name,
                resolved,
                merge or MergeAll(),
                enable_sync_parallel=enable_sync_parallel,
            )
        )
        return self

    def branch(
        self,
        name: str,
        predicate: PredicateFn,
        then: Node | BodyBuilderFn,
        else_: Node | BodyBuilderFn | None = None,
    ) -> _PipelineBuilderBase:
        """Append a conditional branch node."""

        then_node = self._resolve_body(f"{name}.then", then)
        else_node = None if else_ is None else self._resolve_body(f"{name}.else", else_)
        self._nodes.append(branch(name, predicate, then_node, else_node))
        return self

    def loop(
        self,
        name: str,
        body: Node | BodyBuilderFn,
        until: PredicateFn,
        *,
        max_iter: int = 8,
        on_max_iter: ErrorAction = ErrorAction.FAIL,
    ) -> _PipelineBuilderBase:
        """Append a loop node with a hard iteration ceiling."""

        body_node = self._resolve_body(f"{name}.body", body)
        self._nodes.append(
            loop(
                name,
                body_node,
                until,
                max_iter=max_iter,
                on_max_iter=on_max_iter,
            )
        )
        return self

    def map(
        self,
        name: str,
        over: str,
        body: Node | BodyBuilderFn,
        *,
        concurrency: int = 1,
        item_key: str = "item",
        results_key: str = "map_results",
        enable_sync_parallel: bool = False,
    ) -> _PipelineBuilderBase:
        """Append a map node over a sequence field."""

        body_node = self._resolve_body(f"{name}.body", body)
        self._nodes.append(
            map_node(
                name,
                over,
                body_node,
                concurrency=concurrency,
                item_key=item_key,
                results_key=results_key,
                enable_sync_parallel=enable_sync_parallel,
            )
        )
        return self

    def _resolve_body(self, name: str, value: Node | BodyBuilderFn) -> Node:
        """Resolve either a node or a body builder callback into a node."""

        if _is_node(value):
            return value  # type: ignore[return-value]
        body = PipelineBodyBuilder(name=name)
        configured = value(body)  # type: ignore[operator]
        return configured.build_node()

    def _build_root(self) -> Node:
        """Materialize the root node from collected children."""

        if not self._nodes:
            raise ValueError("pipeline builder has no nodes")
        if len(self._nodes) == 1:
            return self._nodes[0]
        return seq(self._defaults.name, *self._nodes)


class SyncPipelineBuilder(_PipelineBuilderBase):
    """Fluent builder that produces a ``SyncPipeline``."""

    def capability(self, *args: Any, **kwargs: Any) -> SyncPipelineBuilder:
        """Append a capability step."""

        super().capability(*args, **kwargs)
        return self

    def effect(self, *args: Any, **kwargs: Any) -> SyncPipelineBuilder:
        """Append an effect step."""

        super().effect(*args, **kwargs)
        return self

    def node(self, value: Node) -> SyncPipelineBuilder:
        """Append an arbitrary node."""

        super().node(value)
        return self

    def parallel(self, *args: Any, **kwargs: Any) -> SyncPipelineBuilder:
        """Append a parallel node."""

        super().parallel(*args, **kwargs)
        return self

    def branch(self, *args: Any, **kwargs: Any) -> SyncPipelineBuilder:
        """Append a branch node."""

        super().branch(*args, **kwargs)
        return self

    def loop(self, *args: Any, **kwargs: Any) -> SyncPipelineBuilder:
        """Append a loop node."""

        super().loop(*args, **kwargs)
        return self

    def map(self, *args: Any, **kwargs: Any) -> SyncPipelineBuilder:
        """Append a map node."""

        super().map(*args, **kwargs)
        return self

    def build(self) -> SyncPipeline:
        """Build the synchronous pipeline."""

        return SyncPipeline(self._build_root(), defaults=self._defaults)


class AsyncPipelineBuilder(_PipelineBuilderBase):
    """Fluent builder that produces an ``AsyncPipeline``."""

    def capability(self, *args: Any, **kwargs: Any) -> AsyncPipelineBuilder:
        """Append a capability step."""

        super().capability(*args, **kwargs)
        return self

    def effect(self, *args: Any, **kwargs: Any) -> AsyncPipelineBuilder:
        """Append an effect step."""

        super().effect(*args, **kwargs)
        return self

    def node(self, value: Node) -> AsyncPipelineBuilder:
        """Append an arbitrary node."""

        super().node(value)
        return self

    def parallel(self, *args: Any, **kwargs: Any) -> AsyncPipelineBuilder:
        """Append a parallel node."""

        super().parallel(*args, **kwargs)
        return self

    def branch(self, *args: Any, **kwargs: Any) -> AsyncPipelineBuilder:
        """Append a branch node."""

        super().branch(*args, **kwargs)
        return self

    def loop(self, *args: Any, **kwargs: Any) -> AsyncPipelineBuilder:
        """Append a loop node."""

        super().loop(*args, **kwargs)
        return self

    def map(self, *args: Any, **kwargs: Any) -> AsyncPipelineBuilder:
        """Append a map node."""

        super().map(*args, **kwargs)
        return self

    def build(self) -> AsyncPipeline:
        """Build the asynchronous pipeline."""

        return AsyncPipeline(self._build_root(), defaults=self._defaults)
