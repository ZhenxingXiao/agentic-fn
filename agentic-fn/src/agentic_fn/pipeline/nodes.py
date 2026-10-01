"""Algebraic node types that compose a pipeline graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_fn.core.step import StepSpec
from agentic_fn.type.aliases import PredicateFn
from agentic_fn.type.enums import ErrorAction
from agentic_fn.type.protocols import MergeStrategy


@dataclass(frozen=True, slots=True)
class Seq:
    """Strict sequential composition of child nodes."""

    name: str
    children: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Parallel:
    """Fan-out composition that merges branch results."""

    name: str
    branches: tuple[tuple[str, Node], ...]
    merge: MergeStrategy
    enable_sync_parallel: bool = False


@dataclass(frozen=True, slots=True)
class Branch:
    """Conditional composition selecting ``then_node`` or ``else_node``."""

    name: str
    predicate: PredicateFn
    then_node: Node
    else_node: Node | None = None


@dataclass(frozen=True, slots=True)
class Loop:
    """
    Iterative composition with a hard ``max_iter`` ceiling.

    ``until`` returning True stops the loop before executing another body pass.
    """

    name: str
    body: Node
    until: PredicateFn
    max_iter: int = 8
    on_max_iter: ErrorAction = ErrorAction.FAIL


@dataclass(frozen=True, slots=True)
class MapNode:
    """
    Map a nested body over a sequence stored in ``state.data[over]``.

    Each item is placed under ``item_key`` for the body; results are collected
    into ``results_key``.
    """

    name: str
    over: str
    body: Node
    concurrency: int = 1
    item_key: str = "item"
    results_key: str = "map_results"
    enable_sync_parallel: bool = False


# Recursive alias for the pipeline AST.
Node = StepSpec | Seq | Parallel | Branch | Loop | MapNode


def seq(name: str, *children: Node) -> Seq:
    """Create a sequential node."""

    if not children:
        raise ValueError("seq() requires at least one child node")
    return Seq(name=name, children=tuple(children))


def parallel(
    name: str,
    branches: dict[str, Node],
    merge: MergeStrategy,
    *,
    enable_sync_parallel: bool = False,
) -> Parallel:
    """Create a parallel fan-out node from a name→node mapping."""

    if not branches:
        raise ValueError("parallel() requires at least one branch")
    return Parallel(
        name=name,
        branches=tuple(branches.items()),
        merge=merge,
        enable_sync_parallel=enable_sync_parallel,
    )


def branch(
    name: str,
    predicate: PredicateFn,
    then_node: Node,
    else_node: Node | None = None,
) -> Branch:
    """Create a conditional branch node."""

    return Branch(name=name, predicate=predicate, then_node=then_node, else_node=else_node)


def loop(
    name: str,
    body: Node,
    until: PredicateFn,
    *,
    max_iter: int = 8,
    on_max_iter: ErrorAction = ErrorAction.FAIL,
) -> Loop:
    """Create a loop node with a mandatory iteration ceiling."""

    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    return Loop(
        name=name,
        body=body,
        until=until,
        max_iter=max_iter,
        on_max_iter=on_max_iter,
    )


def map_node(
    name: str,
    over: str,
    body: Node,
    *,
    concurrency: int = 1,
    item_key: str = "item",
    results_key: str = "map_results",
    enable_sync_parallel: bool = False,
) -> MapNode:
    """Create a map node over a sequence field."""

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    return MapNode(
        name=name,
        over=over,
        body=body,
        concurrency=concurrency,
        item_key=item_key,
        results_key=results_key,
        enable_sync_parallel=enable_sync_parallel,
    )
