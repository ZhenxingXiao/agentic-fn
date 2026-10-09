"""
Typed textual content for production pipelines.

This package models ``text`` / ``json`` / ``markdown`` / ``html`` as kind-tagged
immutable values (not ``str`` subclasses). Bodies stay Unicode strings so they
serialize cleanly into ``State``; large payloads externalize via ``ContentRef``.

Heavy HTML sanitizers and Markdown engines are intentionally *not* bundled —
inject them through ``HtmlSanitizer`` / ``MarkdownRenderer`` protocols.
"""

from __future__ import annotations

from typing import Any

from agentic_fn.content.convert import (
    DEFAULT_CONVERSION_POLICY,
    ConversionPolicy,
    convert,
    ensure_convertible,
)
from agentic_fn.content.errors import (
    ContentCodecError,
    ContentConfigurationError,
    ContentConversionError,
    ContentError,
    ContentKindError,
    ContentLimitError,
    ContentStoreError,
    ContentValidationError,
)
from agentic_fn.content.kinds import (
    CONTENT_REF_TYPE_MARKER,
    CONTENT_SCHEMA_VERSION,
    CONTENT_TYPE_MARKER,
    ContentKind,
)
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits
from agentic_fn.content.model import Content, ContentMeta
from agentic_fn.content.protocols import ContentStore, HtmlSanitizer, MarkdownRenderer
from agentic_fn.content.refs import ContentRef, MemoryContentStore, externalize, resolve
from agentic_fn.content.validate import (
    DEFAULT_VALIDATION_POLICY,
    ValidationPolicy,
    validate_content,
)

__all__ = [
    "CONTENT_REF_TYPE_MARKER",
    "CONTENT_SCHEMA_VERSION",
    "CONTENT_TYPE_MARKER",
    "Content",
    "ContentCodecError",
    "ContentConfigurationError",
    "ContentConversionError",
    "ContentError",
    "ContentKind",
    "ContentKindError",
    "ContentLimitError",
    "ContentLimits",
    "ContentMeta",
    "ContentRef",
    "ContentStore",
    "ContentStoreError",
    "ContentValidationError",
    "ConversionPolicy",
    "DEFAULT_CONTENT_LIMITS",
    "DEFAULT_CONVERSION_POLICY",
    "DEFAULT_VALIDATION_POLICY",
    "HtmlSanitizer",
    "MarkdownRenderer",
    "MemoryContentStore",
    "ValidationPolicy",
    "canonicalize_json",
    "coerce_content",
    "convert",
    "ensure_convertible",
    "externalize",
    "from_json_value",
    "get_content",
    "html_escape",
    "html_extract_text",
    "html_unescape",
    "json_dumps",
    "json_loads",
    "json_merge_objects",
    "json_pointer_get",
    "markdown_strip_to_text",
    "put_content",
    "resolve",
    "text_join_blocks",
    "text_normalize",
    "text_truncate",
    "validate_content",
]


def __getattr__(name: str) -> Any:
    """Lazy ops exports to keep the package import surface light."""

    if name == "put_content":
        from agentic_fn.content.codec import put_content

        return put_content
    if name == "get_content":
        from agentic_fn.content.codec import get_content

        return get_content
    if name == "coerce_content":
        from agentic_fn.content.codec import coerce_content

        return coerce_content
    if name == "json_loads":
        from agentic_fn.content.json_ops import loads

        return loads
    if name == "json_dumps":
        from agentic_fn.content.json_ops import dumps

        return dumps
    if name == "from_json_value":
        from agentic_fn.content.json_ops import from_value

        return from_value
    if name == "json_pointer_get":
        from agentic_fn.content.json_ops import pointer_get

        return pointer_get
    if name == "json_merge_objects":
        from agentic_fn.content.json_ops import merge_objects

        return merge_objects
    if name == "canonicalize_json":
        from agentic_fn.content.json_ops import canonicalize

        return canonicalize
    if name == "text_normalize":
        from agentic_fn.content.text_ops import normalize

        return normalize
    if name == "text_truncate":
        from agentic_fn.content.text_ops import truncate

        return truncate
    if name == "text_join_blocks":
        from agentic_fn.content.text_ops import join_blocks

        return join_blocks
    if name == "html_extract_text":
        from agentic_fn.content.html_ops import extract_text

        return extract_text
    if name == "html_escape":
        from agentic_fn.content.html_ops import escape

        return escape
    if name == "html_unescape":
        from agentic_fn.content.html_ops import unescape

        return unescape
    if name == "markdown_strip_to_text":
        from agentic_fn.content.md_ops import strip_to_text

        return strip_to_text
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
