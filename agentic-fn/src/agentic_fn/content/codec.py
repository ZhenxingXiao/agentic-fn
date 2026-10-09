"""Serialize content values to/from State-friendly mappings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agentic_fn.content.errors import ContentCodecError
from agentic_fn.content.kinds import (
    CONTENT_REF_TYPE_MARKER,
    CONTENT_SCHEMA_VERSION,
    CONTENT_TYPE_MARKER,
    ContentKind,
)
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits
from agentic_fn.content.model import Content, ContentMeta
from agentic_fn.content.refs import ContentRef
from agentic_fn.core.state import State


def content_to_dict(content: Content) -> dict[str, Any]:
    """Encode ``Content`` as a JSON-serializable mapping."""

    return {
        "_type": CONTENT_TYPE_MARKER,
        "schema_version": CONTENT_SCHEMA_VERSION,
        "kind": content.kind.value,
        "body": content.body,
        "media_type": content.media_type,
        "meta": {
            "content_id": content.meta.content_id,
            "source": content.meta.source,
            "encoding": content.meta.encoding,
            "language": content.meta.language,
            "labels": dict(content.meta.labels),
            "derived_from": content.meta.derived_from,
            "extras": dict(content.meta.extras),
        },
        "digest": content.digest(),
        "char_length": content.char_length,
        "byte_length": content.byte_length,
    }


def content_from_dict(
    data: Mapping[str, Any],
    *,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    validate: bool = True,
) -> Content:
    """Decode a mapping produced by ``content_to_dict``."""

    if not isinstance(data, Mapping):
        raise ContentCodecError(f"content mapping must be a Mapping, got {type(data)!r}")
    if data.get("_type") != CONTENT_TYPE_MARKER:
        raise ContentCodecError(
            f"expected _type={CONTENT_TYPE_MARKER!r}, got {data.get('_type')!r}"
        )
    schema = str(data.get("schema_version") or "")
    if schema and schema != CONTENT_SCHEMA_VERSION:
        raise ContentCodecError(
            f"unsupported content schema_version={schema!r} "
            f"(expected {CONTENT_SCHEMA_VERSION!r})"
        )
    try:
        meta_raw = dict(data.get("meta") or {})
        meta = ContentMeta(
            content_id=str(meta_raw.get("content_id") or uuid4()),
            source=meta_raw.get("source"),
            encoding=str(meta_raw.get("encoding") or "utf-8"),
            language=meta_raw.get("language"),
            labels=dict(meta_raw.get("labels") or {}),
            derived_from=meta_raw.get("derived_from"),
            extras=dict(meta_raw.get("extras") or {}),
        )
        return Content.of(
            ContentKind.parse(str(data["kind"])),
            str(data["body"]),
            media_type=data.get("media_type"),
            meta=meta,
            limits=limits,
            validate=validate,
        )
    except ContentCodecError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize decode failures
        raise ContentCodecError(f"invalid content mapping: {exc}") from exc


def is_content_dict(value: object) -> bool:
    """Return whether ``value`` looks like a serialized ``Content``."""

    return isinstance(value, Mapping) and value.get("_type") == CONTENT_TYPE_MARKER


def is_content_ref_dict(value: object) -> bool:
    """Return whether ``value`` looks like a serialized ``ContentRef``."""

    return isinstance(value, Mapping) and value.get("_type") == CONTENT_REF_TYPE_MARKER


def coerce_content(
    value: Content | ContentRef | Mapping[str, Any] | str,
    *,
    default_kind: ContentKind = ContentKind.TEXT,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    validate: bool = True,
) -> Content | ContentRef:
    """
    Normalize common input shapes into ``Content`` or ``ContentRef``.

    - ``Content`` / ``ContentRef`` are returned as-is (Content optionally revalidated)
    - mappings are decoded by ``_type``
    - bare ``str`` becomes ``Content`` of ``default_kind``
    """

    if isinstance(value, Content):
        if validate:
            from agentic_fn.content.validate import validate_content

            validate_content(value, limits=limits)
        return value
    if isinstance(value, ContentRef):
        return value
    if isinstance(value, Mapping):
        if is_content_dict(value):
            return content_from_dict(value, limits=limits, validate=validate)
        if is_content_ref_dict(value):
            return ContentRef.from_dict(value)
        raise ContentCodecError(
            f"mapping is neither Content nor ContentRef (got _type={value.get('_type')!r})"
        )
    if isinstance(value, str):
        return Content.of(default_kind, value, limits=limits, validate=validate)
    raise ContentCodecError(f"cannot coerce {type(value)!r} to Content")


def put_content(state: State, key: str, content: Content | ContentRef) -> State:
    """
    Store content into ``State.data[key]`` as a serializable mapping.

    Prefer this helper over stuffing raw ``Content`` objects into State so
    digests and replay remain JSON-stable.
    """

    payload = content.to_dict()
    return state.update(**{key: payload})


def get_content(
    state: State,
    key: str,
    *,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    validate: bool = True,
    default: Content | ContentRef | None = None,
) -> Content | ContentRef:
    """
    Load content previously stored via ``put_content``.

    Raises ``KeyError`` when missing and no ``default`` is provided.
    Raises ``ContentCodecError`` when the stored value is malformed.
    """

    if key not in state.data:
        if default is not None:
            return default
        raise KeyError(key)
    value = state.data[key]
    if isinstance(value, (Content, ContentRef)):
        return value
    if not isinstance(value, Mapping):
        raise ContentCodecError(
            f"state[{key!r}] is not a content mapping (got {type(value)!r})"
        )
    return coerce_content(value, limits=limits, validate=validate)
