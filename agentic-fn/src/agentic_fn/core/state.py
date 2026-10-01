"""Immutable pipeline state passed between steps."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from agentic_fn.core.errors import StateValidationError


@dataclass(frozen=True, slots=True)
class StateMeta:
    """Framework metadata attached to every state instance."""

    schema_version: str = "1"
    run_id: str = ""
    parent_run_id: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def with_run_id(self, run_id: str) -> StateMeta:
        """Return a copy with an updated run identifier."""

        return replace(self, run_id=run_id)

    def with_labels(self, **labels: str) -> StateMeta:
        """Return a copy with merged labels."""

        merged = dict(self.labels)
        merged.update(labels)
        return replace(self, labels=merged)


@dataclass(frozen=True, slots=True)
class State:
    """
    Explicit, serializable business payload shared by pipeline steps.

    Secrets must never be stored in ``data``. Large blobs should be referenced
    indirectly (object refs) rather than inlined.
    """

    data: Mapping[str, Any] = field(default_factory=dict)
    meta: StateMeta = field(default_factory=StateMeta)

    @staticmethod
    def empty(
        *,
        run_id: str | None = None,
        schema_version: str = "1",
        labels: Mapping[str, str] | None = None,
    ) -> State:
        """Create an empty state with freshly allocated metadata."""

        return State(
            data={},
            meta=StateMeta(
                schema_version=schema_version,
                run_id=run_id or str(uuid4()),
                labels=dict(labels or {}),
            ),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Read a business field from ``data``."""

        return self.data.get(key, default)

    def update(self, **values: Any) -> State:
        """
        Return a new state with ``data`` shallow-updated by ``values``.

        Nested containers are deep-copied to reduce accidental shared mutation.
        """

        merged = copy.deepcopy(dict(self.data))
        for key, value in values.items():
            merged[key] = copy.deepcopy(value)
        return replace(self, data=merged)

    def remove(self, *keys: str) -> State:
        """Return a new state without the given keys."""

        merged = {k: copy.deepcopy(v) for k, v in self.data.items() if k not in keys}
        return replace(self, data=merged)

    def with_meta(self, meta: StateMeta) -> State:
        """Return a new state with replacement metadata."""

        return replace(self, meta=meta)

    def digest(self) -> str:
        """
        Compute a stable digest of business data for tracing and replay keys.

        Metadata is intentionally excluded so digest identity tracks payload only.
        """

        payload = json.dumps(self.data, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_state(value: object) -> State:
    """
    Validate that ``value`` is a usable ``State`` instance.

    Raises
    ------
    StateValidationError
        If the value is not a ``State`` or its fields are malformed.
    """

    if not isinstance(value, State):
        raise StateValidationError(f"Step must return State, got {type(value)!r}")
    if not isinstance(value.data, Mapping):
        raise StateValidationError("State.data must be a Mapping")
    if not isinstance(value.meta, StateMeta):
        raise StateValidationError("State.meta must be a StateMeta instance")
    if not isinstance(value.meta.schema_version, str) or not value.meta.schema_version:
        raise StateValidationError("State.meta.schema_version must be a non-empty string")
    return value


# Reserved key used by ErrorAction.CONTINUE to accumulate soft failures.
ERRORS_KEY = "_errors"
