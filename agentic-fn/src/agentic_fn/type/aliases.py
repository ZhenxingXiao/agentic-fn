"""Callable and literal type aliases for the public API surface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from agentic_fn.core.context import Context
    from agentic_fn.core.state import State
    from agentic_fn.observe.events import Event
    from agentic_fn.pipeline.nodes import Node

# Sync step callables: (state, context) -> state
CapabilityFn: TypeAlias = Callable[["State", "Context"], "State"]
EffectFn: TypeAlias = Callable[["State", "Context"], "State"]

# Async step callables: (state, context) -> awaitable state
AsyncCapabilityFn: TypeAlias = Callable[["State", "Context"], Awaitable["State"]]
AsyncEffectFn: TypeAlias = Callable[["State", "Context"], Awaitable["State"]]

# Predicates and reducers used by control-flow nodes
PredicateFn: TypeAlias = Callable[["State", "Context"], bool]
AsyncPredicateFn: TypeAlias = Callable[["State", "Context"], Awaitable[bool]]
ReduceFn: TypeAlias = Callable[["State", list["State"]], "State"]

# Event callbacks (must not break the main execution path)
OnEvent: TypeAlias = Callable[["Event"], None]
AsyncOnEvent: TypeAlias = Callable[["Event"], Awaitable[None]]

# Builder helper that configures a nested sequence of nodes
BodyBuilderFn: TypeAlias = Callable[["PipelineBodyBuilder"], "PipelineBodyBuilder"]

# JSON-ish mapping used for serializable state payloads
JSONMapping: TypeAlias = Mapping[str, Any]

# Forward declaration placeholder resolved in pipeline.builder
if TYPE_CHECKING:
    from agentic_fn.pipeline.builder import PipelineBodyBuilder

NodeFactory: TypeAlias = Callable[[], "Node"]
