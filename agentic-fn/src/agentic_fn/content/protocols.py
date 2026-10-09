"""Pluggable protocols for heavy HTML / Markdown / storage backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentic_fn.content.kinds import ContentKind


@runtime_checkable
class HtmlSanitizer(Protocol):
    """
    Production HTML sanitizer (e.g. bleach / nh3 / custom allow-lists).

    The built-in converter only does structural text extraction. Enterprise
    deployments that re-emit HTML must inject a sanitizer before persistence
    or browser rendering.
    """

    def sanitize(self, html: str) -> str:
        """Return sanitized HTML safe for the configured trust boundary."""


@runtime_checkable
class MarkdownRenderer(Protocol):
    """
    Full Markdown renderer (e.g. markdown-it-py / mistune).

    Core ships a conservative markdown→text normalizer only. Use this protocol
    when you need HTML output or CommonMark fidelity.
    """

    def render_html(self, markdown: str) -> str:
        """Render Markdown to HTML."""

    def render_text(self, markdown: str) -> str:
        """Render Markdown to plain text (may strip formatting)."""


@runtime_checkable
class ContentStore(Protocol):
    """
    Object store used to externalize large content bodies.

    Implementations typically wrap S3 / GCS / local blob storage. The framework
    only persists ``ContentRef`` handles inside ``State``.
    """

    def put(
        self,
        body: str,
        *,
        kind: ContentKind,
        media_type: str | None = None,
        content_id: str | None = None,
    ) -> str:
        """
        Persist ``body`` and return an opaque object key / URI.

        The returned key is what ``ContentRef.uri`` stores.
        """

    def get(self, uri: str) -> str:
        """Load a previously stored body by URI / key."""

    def exists(self, uri: str) -> bool:
        """Return whether ``uri`` is resolvable."""

    def delete(self, uri: str) -> None:
        """Best-effort delete; missing keys should not raise."""
