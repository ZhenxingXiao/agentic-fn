"""External content references for large payloads kept out of State."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from agentic_fn.content.errors import ContentCodecError, ContentStoreError
from agentic_fn.content.kinds import (
    CONTENT_REF_TYPE_MARKER,
    CONTENT_SCHEMA_VERSION,
    ContentKind,
)
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits
from agentic_fn.content.model import Content, ContentMeta
from agentic_fn.content.protocols import ContentStore


@dataclass(frozen=True, slots=True)
class ContentRef:
    """
    Handle to content stored outside ``State``.

    State carries only the reference (URI, digest, kind, sizes). Resolving the
    body requires a ``ContentStore`` implementation injected by the application.
    """

    uri: str
    kind: ContentKind
    digest: str
    char_length: int
    byte_length: int
    media_type: str | None = None
    content_id: str = field(default_factory=lambda: str(uuid4()))
    labels: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reference for State / wire transport."""

        return {
            "_type": CONTENT_REF_TYPE_MARKER,
            "schema_version": CONTENT_SCHEMA_VERSION,
            "uri": self.uri,
            "kind": self.kind.value,
            "digest": self.digest,
            "char_length": self.char_length,
            "byte_length": self.byte_length,
            "media_type": self.media_type,
            "content_id": self.content_id,
            "labels": dict(self.labels),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ContentRef:
        """Deserialize a reference mapping."""

        if data.get("_type") != CONTENT_REF_TYPE_MARKER:
            raise ContentCodecError(
                f"expected _type={CONTENT_REF_TYPE_MARKER!r}, got {data.get('_type')!r}"
            )
        try:
            return ContentRef(
                uri=str(data["uri"]),
                kind=ContentKind.parse(str(data["kind"])),
                digest=str(data["digest"]),
                char_length=int(data["char_length"]),
                byte_length=int(data["byte_length"]),
                media_type=data.get("media_type"),
                content_id=str(data.get("content_id") or uuid4()),
                labels=dict(data.get("labels") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContentCodecError(f"invalid ContentRef mapping: {exc}") from exc

    def with_labels(self, **labels: str) -> ContentRef:
        """Return a copy with merged labels."""

        merged = dict(self.labels)
        merged.update(labels)
        return replace(self, labels=merged)


def externalize(
    content: Content,
    store: ContentStore,
    *,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    force: bool = False,
) -> ContentRef | Content:
    """
    Persist large content into ``store`` and return a ``ContentRef``.

    When ``force`` is False and the body is under ``inline_max_chars``, the
    original ``Content`` is returned unchanged.
    """

    if not force and not limits.should_externalize(content.body):
        return content
    uri = store.put(
        content.body,
        kind=content.kind,
        media_type=content.media_type,
        content_id=content.meta.content_id or None,
    )
    return ContentRef(
        uri=uri,
        kind=content.kind,
        digest=content.digest(),
        char_length=content.char_length,
        byte_length=content.byte_length,
        media_type=content.media_type,
        content_id=content.meta.content_id or str(uuid4()),
        labels=dict(content.meta.labels),
    )


def resolve(
    ref: ContentRef,
    store: ContentStore,
    *,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    validate: bool = True,
    verify_digest: bool = True,
) -> Content:
    """
    Load a ``ContentRef`` back into an in-memory ``Content`` value.

    Raises ``ContentStoreError`` when the URI is missing or the digest mismatches.
    """

    try:
        body = store.get(ref.uri)
    except Exception as exc:  # noqa: BLE001 - normalize store failures
        raise ContentStoreError(
            f"failed to resolve content uri={ref.uri!r}: {exc}",
            cause=exc,
        ) from exc

    content = Content.of(
        ref.kind,
        body,
        media_type=ref.media_type,
        meta=ContentMeta(
            content_id=ref.content_id,
            labels=dict(ref.labels),
            source=f"content_ref:{ref.uri}",
        ),
        limits=limits,
        validate=validate,
    )
    if verify_digest and content.digest() != ref.digest:
        raise ContentStoreError(
            f"digest mismatch for uri={ref.uri!r}: "
            f"expected {ref.digest[:16]}…, got {content.digest()[:16]}…"
        )
    return content


@dataclass(slots=True)
class MemoryContentStore:
    """In-process content store for tests, harnesses, and local demos."""

    prefix: str = "mem://"
    _blobs: dict[str, str] = field(default_factory=dict, repr=False)

    def put(
        self,
        body: str,
        *,
        kind: ContentKind,
        media_type: str | None = None,
        content_id: str | None = None,
    ) -> str:
        """Store ``body`` and return a mem:// URI."""

        _ = media_type
        key = content_id or str(uuid4())
        uri = f"{self.prefix}{kind.value}/{key}"
        self._blobs[uri] = body
        return uri

    def get(self, uri: str) -> str:
        """Return a stored body."""

        try:
            return self._blobs[uri]
        except KeyError as exc:
            raise ContentStoreError(f"unknown content uri: {uri!r}") from exc

    def exists(self, uri: str) -> bool:
        """Return whether ``uri`` is present."""

        return uri in self._blobs

    def delete(self, uri: str) -> None:
        """Drop a stored body if present."""

        self._blobs.pop(uri, None)
