"""Minimal harness implementations for permissions and budget hooks."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass, field

from agentic_fn.core.errors import HarnessDeniedError


@dataclass(slots=True)
class AllowAllHarness:
    """Permissive harness used as a safe default in development."""

    def allow_step(self, step_name: str, kind: str) -> bool:
        """Allow every step."""

        _ = step_name, kind
        return True

    def on_budget(self, remaining: Mapping[str, float | int | None]) -> None:
        """Ignore budget updates."""

        _ = remaining


@dataclass(slots=True)
class AllowlistHarness:
    """
    Harness that permits only explicitly listed step names.

    Useful as a starting point for production tool/step sandboxing.
    """

    allowed_steps: Set[str] = field(default_factory=set)
    allowed_kinds: Set[str] = field(default_factory=lambda: {"capability", "effect"})

    def allow_step(self, step_name: str, kind: str) -> bool:
        """Return whether the step name and kind are allow-listed."""

        if kind not in self.allowed_kinds:
            return False
        if not self.allowed_steps:
            return True
        return step_name in self.allowed_steps

    def on_budget(self, remaining: Mapping[str, float | int | None]) -> None:
        """Ignore budget updates in the base allowlist harness."""

        _ = remaining

    def require_step(self, step_name: str, kind: str) -> None:
        """Raise when a step is not allowed."""

        if not self.allow_step(step_name, kind):
            raise HarnessDeniedError(f"step not allowed: {step_name!r} kind={kind!r}")
