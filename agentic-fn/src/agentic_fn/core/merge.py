"""Merge strategies for parallel branch joins."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.core.errors import MergeConflictError
from agentic_fn.core.state import State
from agentic_fn.type.enums import ConflictMode


@dataclass(frozen=True, slots=True)
class MergeByKeys:
    """
    Merge selected keys from each named branch into the base state.

    ``mapping`` maps branch name → iterable of keys to copy from that branch.
    """

    mapping: Mapping[str, Sequence[str]]
    conflict_mode: ConflictMode = ConflictMode.ERROR

    def merge(self, base: State, parts: Sequence[tuple[str, State]]) -> State:
        """Merge configured keys from each branch into ``base``."""

        updates: dict[str, Any] = {}
        owners: dict[str, str] = {}
        part_map = dict(parts)

        for branch_name, keys in self.mapping.items():
            if branch_name not in part_map:
                raise MergeConflictError(f"missing parallel branch {branch_name!r}")
            branch_state = part_map[branch_name]
            for key in keys:
                if key in updates:
                    if self.conflict_mode is ConflictMode.ERROR:
                        raise MergeConflictError(
                            f"key conflict on {key!r} between {owners[key]!r} "
                            f"and {branch_name!r}"
                        )
                    if self.conflict_mode is ConflictMode.PREFER_LEFT:
                        continue
                updates[key] = branch_state.get(key)
                owners[key] = branch_name

        return base.update(**updates)


@dataclass(frozen=True, slots=True)
class MergeAll:
    """
    Merge all keys from all branches into base with a conflict policy.

    Keys listed in ``exclude`` are ignored.
    """

    conflict_mode: ConflictMode = ConflictMode.ERROR
    exclude: frozenset[str] = field(default_factory=frozenset)

    def merge(self, base: State, parts: Sequence[tuple[str, State]]) -> State:
        """Merge every non-excluded key from branch states."""

        updates: dict[str, Any] = {}
        owners: dict[str, str] = {}
        for branch_name, branch_state in parts:
            for key, value in branch_state.data.items():
                if key in self.exclude:
                    continue
                if key in updates:
                    if self.conflict_mode is ConflictMode.ERROR:
                        raise MergeConflictError(
                            f"key conflict on {key!r} between {owners[key]!r} "
                            f"and {branch_name!r}"
                        )
                    if self.conflict_mode is ConflictMode.PREFER_LEFT:
                        continue
                updates[key] = value
                owners[key] = branch_name
        return base.update(**updates)


@dataclass(frozen=True, slots=True)
class ReduceMerge:
    """Delegate merging to a user-supplied reducer."""

    fn: Callable[[State, list[State]], State]

    def merge(self, base: State, parts: Sequence[tuple[str, State]]) -> State:
        """Reduce branch states with the configured function."""

        return self.fn(base, [state for _, state in parts])
