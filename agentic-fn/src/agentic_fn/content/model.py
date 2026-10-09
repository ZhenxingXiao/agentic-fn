"""Immutable content value object used across pipeline boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from agentic_fn.content.errors import ContentKindError, ContentValidationError
from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits


@dataclass(frozen=True, slots=True)
class ContentMeta:
    """
    Non-payload metadata attached to a content value.

    Kept separate from ``body`` so digests can optionally ignore volatile labels
    while still carrying provenance for audit and routing.
    """

    content_id: str = ""
    source: str | None = None
    encoding: str = "utf-8"
    language: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    # Optional parent / derivation links (e.g. html -> markdown conversion).
    derived_from: str | None = None
    # Free-form extension bag for adapters (must stay JSON-serializable).
    extras: Mapping[str, Any] = field(default_factory=dict)

    def with_labels(self, **labels: str) -> ContentMeta:
        """Return a copy with merged labels."""

        merged = dict(self.labels)
        merged.update(labels)
        return replace(self, labels=merged)

    def with_extras(self, **extras: Any) -> ContentMeta:
        """Return a copy with merged extras."""

        merged = dict(self.extras)
        merged.update(extras)
        return replace(self, extras=merged)


@dataclass(frozen=True, slots=True)
class Content:
    """
    Kind-tagged textual payload.

    Design rules
    ------------
    - ``body`` is always a Unicode ``str`` (never bytes).
    - Prefer putting ``to_dict()`` into ``State.data`` rather than the object
      itself, so digests and replay stay JSON-stable.
    - Large bodies should be externalized as ``ContentRef`` (see ``refs``).
    - Construction validates kind + size by default (``validate=True``).
    """

    kind: ContentKind
    body: str
    media_type: str | None = None
    meta: ContentMeta = field(default_factory=ContentMeta)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @staticmethod
    def text(
        body: str,
        *,
        media_type: str | None = None,
        meta: ContentMeta | None = None,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        validate: bool = True,
    ) -> Content:
        """Create plain-text content."""

        return Content.of(
            ContentKind.TEXT,
            body,
            media_type=media_type,
            meta=meta,
            limits=limits,
            validate=validate,
        )

    @staticmethod
    def json(
        body: str,
        *,
        media_type: str | None = None,
        meta: ContentMeta | None = None,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        validate: bool = True,
    ) -> Content:
        """Create JSON-text content (body must be a JSON document string)."""

        return Content.of(
            ContentKind.JSON,
            body,
            media_type=media_type,
            meta=meta,
            limits=limits,
            validate=validate,
        )

    @staticmethod
    def markdown(
        body: str,
        *,
        media_type: str | None = None,
        meta: ContentMeta | None = None,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        validate: bool = True,
    ) -> Content:
        """Create Markdown content."""

        return Content.of(
            ContentKind.MARKDOWN,
            body,
            media_type=media_type,
            meta=meta,
            limits=limits,
            validate=validate,
        )

    @staticmethod
    def html(
        body: str,
        *,
        media_type: str | None = None,
        meta: ContentMeta | None = None,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        validate: bool = True,
    ) -> Content:
        """Create HTML content."""

        return Content.of(
            ContentKind.HTML,
            body,
            media_type=media_type,
            meta=meta,
            limits=limits,
            validate=validate,
        )

    @staticmethod
    def of(
        kind: ContentKind | str,
        body: str,
        *,
        media_type: str | None = None,
        meta: ContentMeta | None = None,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        validate: bool = True,
    ) -> Content:
        """
        Create content of any kind.

        When ``validate`` is True (default), kind-specific rules and size caps
        are enforced before the instance is returned.
        """

        if not isinstance(body, str):
            raise ContentValidationError(
                f"content body must be str, got {type(body)!r}"
            )
        resolved_kind = ContentKind.parse(kind)
        resolved_meta = meta or ContentMeta(content_id=str(uuid4()))
        if not resolved_meta.content_id:
            resolved_meta = replace(resolved_meta, content_id=str(uuid4()))
        content = Content(
            kind=resolved_kind,
            body=body,
            media_type=media_type or resolved_kind.default_media_type,
            meta=resolved_meta,
        )
        if validate:
            from agentic_fn.content.validate import validate_content

            validate_content(content, limits=limits)
        return content

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def char_length(self) -> int:
        """Number of Unicode code points in ``body``."""

        return len(self.body)

    @property
    def byte_length(self) -> int:
        """UTF-8 byte length of ``body``."""

        return len(self.body.encode("utf-8"))

    def is_empty(self) -> bool:
        """Return True when the body is empty."""

        return self.body == ""

    def digest(self, *, include_meta: bool = False) -> str:
        """
        Stable SHA-256 digest of kind + body (+ optional meta).

        Used for replay keys, deduplication, and ContentRef identity.
        """

        hasher = hashlib.sha256()
        hasher.update(self.kind.value.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(self.body.encode("utf-8"))
        if include_meta:
            import json

            meta_payload = {
                "source": self.meta.source,
                "encoding": self.meta.encoding,
                "language": self.meta.language,
                "labels": dict(self.meta.labels),
                "derived_from": self.meta.derived_from,
                "extras": dict(self.meta.extras),
            }
            hasher.update(b"\0")
            hasher.update(
                json.dumps(
                    meta_payload,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        return hasher.hexdigest()

    def preview(self, *, max_chars: int | None = None) -> str:
        """Return a truncated preview suitable for logs / traces."""

        limit = max_chars or DEFAULT_CONTENT_LIMITS.preview_chars
        if len(self.body) <= limit:
            return self.body
        return self.body[: max(0, limit - 1)] + "…"

    # ------------------------------------------------------------------
    # Kind guards / transforms
    # ------------------------------------------------------------------
    def ensure_kind(self, *kinds: ContentKind | str) -> Content:
        """
        Assert this content has one of the expected kinds.

        Returns ``self`` for fluent chaining; raises ``ContentKindError`` otherwise.
        """

        expected = {ContentKind.parse(k) for k in kinds}
        if self.kind not in expected:
            names = ", ".join(sorted(k.value for k in expected))
            raise ContentKindError(
                f"expected content kind in {{{names}}}, got {self.kind.value!r}"
            )
        return self

    def with_kind(
        self,
        kind: ContentKind | str,
        *,
        revalidate: bool = True,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    ) -> Content:
        """
        Return a copy re-tagged with a new kind (body unchanged).

        This is a *label* change, not a conversion. Prefer ``convert.convert``
        when the body must change (HTML→text, object→JSON string, …).
        """

        resolved = ContentKind.parse(kind)
        updated = replace(
            self,
            kind=resolved,
            media_type=resolved.default_media_type,
            meta=replace(self.meta, derived_from=self.meta.content_id or None),
        )
        if revalidate:
            from agentic_fn.content.validate import validate_content

            validate_content(updated, limits=limits)
        return updated

    def with_body(
        self,
        body: str,
        *,
        revalidate: bool = True,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    ) -> Content:
        """Return a copy with a replaced body."""

        updated = replace(self, body=body)
        if revalidate:
            from agentic_fn.content.validate import validate_content

            validate_content(updated, limits=limits)
        return updated

    def with_meta(self, meta: ContentMeta) -> Content:
        """Return a copy with replacement metadata."""

        return replace(self, meta=meta)

    def with_labels(self, **labels: str) -> Content:
        """Return a copy with merged meta labels."""

        return replace(self, meta=self.meta.with_labels(**labels))

    # ------------------------------------------------------------------
    # Serialization helpers (delegated for a smaller model surface)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-friendly mapping for ``State``."""

        from agentic_fn.content.codec import content_to_dict

        return content_to_dict(self)

    @staticmethod
    def from_dict(
        data: Mapping[str, Any],
        *,
        limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
        validate: bool = True,
    ) -> Content:
        """Deserialize from a mapping produced by ``to_dict``."""

        from agentic_fn.content.codec import content_from_dict

        return content_from_dict(data, limits=limits, validate=validate)

    # ------------------------------------------------------------------
    # Kind-specific convenience (thin wrappers over ops modules)
    # ------------------------------------------------------------------
    def parse_json(self) -> Any:
        """Parse JSON body into Python objects; requires ``kind == JSON``."""

        from agentic_fn.content import json_ops

        self.ensure_kind(ContentKind.JSON)
        return json_ops.loads(self.body)

    def as_text(
        self,
        *,
        markdown_renderer: Any | None = None,
    ) -> Content:
        """
        Convert this content to plain ``TEXT`` (lossy for HTML / Markdown).

        Equivalent to ``convert.convert(self, ContentKind.TEXT, ...)``.
        """

        from agentic_fn.content.convert import ConversionPolicy, convert

        return convert(
            self,
            ContentKind.TEXT,
            policy=ConversionPolicy(lossy=True),
            markdown_renderer=markdown_renderer,
        )

    def derive_meta(self, *, conversion: str) -> ContentMeta:
        """
        Build metadata for a derived content value (conversion / transform).

        Allocates a fresh ``content_id`` via the destination factory; callers
        typically pass the result into ``Content.of`` / kind factories.
        """

        labels = dict(self.meta.labels)
        labels["converted_from"] = self.kind.value
        extras = dict(self.meta.extras)
        extras["conversion"] = conversion
        return ContentMeta(
            content_id="",
            source=self.meta.source,
            encoding=self.meta.encoding,
            language=self.meta.language,
            labels=labels,
            derived_from=self.meta.content_id or None,
            extras=extras,
        )
