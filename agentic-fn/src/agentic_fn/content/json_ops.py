"""JSON document helpers for ``ContentKind.JSON`` payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentic_fn.content.errors import ContentKindError, ContentValidationError
from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits
from agentic_fn.content.model import Content, ContentMeta
from agentic_fn.content.validate import validate_content


def loads(body: str) -> Any:
    """Parse a JSON document string into Python objects."""

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ContentValidationError(f"invalid JSON: {exc}") from exc


def dumps(
    value: Any,
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    indent: int | None = None,
) -> str:
    """Serialize a Python value to a canonical JSON document string."""

    try:
        return json.dumps(
            value,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise ContentValidationError(f"value is not JSON-serializable: {exc}") from exc


def from_value(
    value: Any,
    *,
    meta: ContentMeta | None = None,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    sort_keys: bool = True,
    indent: int | None = None,
) -> Content:
    """Build JSON ``Content`` from an arbitrary JSON-serializable value."""

    body = dumps(value, sort_keys=sort_keys, indent=indent)
    return Content.json(body, meta=meta, limits=limits, validate=True)


def to_value(content: Content) -> Any:
    """Parse JSON content into Python objects."""

    content.ensure_kind(ContentKind.JSON)
    return loads(content.body)


def pointer_get(content: Content, path: str, *, default: Any = None) -> Any:
    """
    Resolve a minimal JSON Pointer-like path (``/a/b/0``).

    Empty path (``""`` or ``"/"``) returns the whole document. Missing paths
    return ``default`` rather than raising, which is friendlier inside pipelines.
    """

    content.ensure_kind(ContentKind.JSON)
    doc = loads(content.body)
    if path in ("", "/"):
        return doc
    if not path.startswith("/"):
        raise ContentValidationError(f"JSON pointer must start with '/', got {path!r}")
    current: Any = doc
    for raw in path.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                return default
            current = current[token]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                index = int(token)
            except ValueError:
                return default
            if index < 0 or index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def merge_objects(
    base: Content,
    patch: Mapping[str, Any],
    *,
    deep: bool = True,
) -> Content:
    """
    Merge ``patch`` into a JSON object content value.

    Only object roots are supported. Arrays / scalars raise ``ContentKindError``.
    """

    base.ensure_kind(ContentKind.JSON)
    root = loads(base.body)
    if not isinstance(root, dict):
        raise ContentKindError("merge_objects requires a JSON object root")
    if deep:
        merged = _deep_merge(dict(root), dict(patch))
    else:
        merged = dict(root)
        merged.update(patch)
    return from_value(
        merged,
        meta=base.meta.with_extras(merged=True),
    )


def _deep_merge(left: dict[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings (right wins on conflicts)."""

    for key, value in right.items():
        if (
            key in left
            and isinstance(left[key], dict)
            and isinstance(value, Mapping)
        ):
            left[key] = _deep_merge(dict(left[key]), value)
        else:
            left[key] = value
    return left


def canonicalize(content: Content) -> Content:
    """
    Re-serialize JSON with sorted keys and compact separators.

    Useful before digesting / comparing model outputs.
    """

    content.ensure_kind(ContentKind.JSON)
    value = loads(content.body)
    body = dumps(value, sort_keys=True, indent=None)
    return content.with_body(body)
