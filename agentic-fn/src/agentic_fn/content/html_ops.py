"""HTML helpers built on the Python standard library (no BeautifulSoup)."""

from __future__ import annotations

import html as html_lib
import re
from html.parser import HTMLParser

from agentic_fn.content.errors import ContentValidationError
from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.model import Content
from agentic_fn.content.protocols import HtmlSanitizer

# Tags whose text content should be discarded for plain-text extraction.
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
# Block-ish tags that should introduce newlines in extracted text.
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "dl",
        "dt",
        "dd",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    }
)
_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    """HTMLParser that accumulates visible text and drops script/style bodies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        lowered = tag.lower()
        if lowered in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if lowered in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if lowered in _BLOCK_TAGS and lowered != "br":
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._chunks.append(data)

    def text(self) -> str:
        """Return extracted, lightly normalized plain text."""

        raw = "".join(self._chunks)
        raw = _WS_RE.sub(" ", raw)
        raw = _MULTI_NL_RE.sub("\n\n", raw)
        return "\n".join(line.strip() for line in raw.split("\n")).strip()


def unescape(text: str) -> str:
    """Decode HTML character references (``&amp;`` → ``&``)."""

    return html_lib.unescape(text)


def escape(text: str, *, quote: bool = True) -> str:
    """Escape plain text for safe embedding into HTML."""

    return html_lib.escape(text, quote=quote)


def extract_text(html: str) -> str:
    """
    Extract visible text from HTML, dropping ``script`` / ``style`` content.

    This is intentional lossy conversion for LLM / retrieval pipelines — not a
    fidelity-preserving DOM serializer.
    """

    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - HTMLParser can raise on broken markup
        raise ContentValidationError(f"failed to parse HTML: {exc}") from exc
    return unescape(parser.text())


def to_text_content(content: Content) -> Content:
    """Convert HTML content into plain-text content."""

    content.ensure_kind(ContentKind.HTML)
    return Content.text(
        extract_text(content.body),
        meta=content.derive_meta(conversion="html_to_text"),
    )


def sanitize(content: Content, sanitizer: HtmlSanitizer) -> Content:
    """
    Run a production sanitizer over HTML content and return a new value.

    The framework does not ship an allow-list sanitizer; inject bleach/nh3/etc.
    """

    content.ensure_kind(ContentKind.HTML)
    cleaned = sanitizer.sanitize(content.body)
    return content.with_body(cleaned)
