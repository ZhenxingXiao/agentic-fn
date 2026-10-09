"""Content kind enumeration and media-type defaults."""

from __future__ import annotations

from enum import Enum


class ContentKind(str, Enum):
    """
    Semantic format tag for textual payloads flowing through pipelines.

    The underlying storage is always a Unicode ``str`` (``Content.body``).
    The kind tells validators, converters, and LLM adapters how to interpret it.
    """

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"

    @property
    def default_media_type(self) -> str:
        """IANA-ish media type used when none is explicitly provided."""

        return _DEFAULT_MEDIA_TYPES[self]

    @classmethod
    def parse(cls, value: str | ContentKind) -> ContentKind:
        """Parse a kind from enum or string (case-insensitive)."""

        if isinstance(value, ContentKind):
            return value
        normalized = value.strip().lower()
        aliases = {
            "md": cls.MARKDOWN,
            "text/plain": cls.TEXT,
            "application/json": cls.JSON,
            "text/markdown": cls.MARKDOWN,
            "text/html": cls.HTML,
            "plain": cls.TEXT,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unknown content kind: {value!r}") from exc


_DEFAULT_MEDIA_TYPES: dict[ContentKind, str] = {
    ContentKind.TEXT: "text/plain; charset=utf-8",
    ContentKind.JSON: "application/json; charset=utf-8",
    ContentKind.MARKDOWN: "text/markdown; charset=utf-8",
    ContentKind.HTML: "text/html; charset=utf-8",
}

# Marker written into serialized mappings so State round-trips can rehydrate.
CONTENT_TYPE_MARKER = "agentic_fn.content"
CONTENT_SCHEMA_VERSION = "1"
CONTENT_REF_TYPE_MARKER = "agentic_fn.content_ref"
