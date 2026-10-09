"""Markdown helpers: conservative core normalizer + pluggable full renderer."""

from __future__ import annotations

import re

from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.model import Content
from agentic_fn.content.protocols import MarkdownRenderer

# Order matters: fenced code before inline code; links before emphasis.
_FENCED_CODE_RE = re.compile(r"```[\w+-]*\n.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_LIST_RE = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(.*?)\1", re.DOTALL)
_HR_RE = re.compile(r"^(\s*[-*_]){3,}\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def strip_to_text(markdown: str) -> str:
    """
    Conservatively strip common Markdown constructs to plain text.

    This is intentionally incomplete CommonMark — good enough for retrieval
    normalization and prompt flattening. For fidelity, inject a
    ``MarkdownRenderer``.
    """

    text = markdown.replace("\r\n", "\n").replace("\r", "\n")

    def _fenced(match: re.Match[str]) -> str:
        block = match.group(0)
        lines = block.strip("`").split("\n", 1)
        body = lines[1] if len(lines) > 1 else ""
        return body.rstrip("`").strip()

    text = _FENCED_CODE_RE.sub(_fenced, text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_RE.sub("", text)
    text = _HR_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def to_text_content(
    content: Content,
    *,
    renderer: MarkdownRenderer | None = None,
) -> Content:
    """Convert Markdown content into plain-text content."""

    content.ensure_kind(ContentKind.MARKDOWN)
    if renderer is not None:
        body = renderer.render_text(content.body)
        conversion = "markdown_renderer_to_text"
    else:
        body = strip_to_text(content.body)
        conversion = "markdown_strip_to_text"
    return Content.text(body, meta=content.derive_meta(conversion=conversion))


def to_html_content(
    content: Content,
    *,
    renderer: MarkdownRenderer,
) -> Content:
    """
    Render Markdown to HTML via an injected production renderer.

    Core does not ship a full Markdown engine — pass markdown-it-py / mistune /
    a hosted service adapter that implements ``MarkdownRenderer``.
    """

    content.ensure_kind(ContentKind.MARKDOWN)
    html = renderer.render_html(content.body)
    return Content.html(
        html,
        meta=content.derive_meta(conversion="markdown_to_html"),
    )
