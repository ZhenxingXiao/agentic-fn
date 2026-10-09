"""Plain-text normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.model import Content

_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class TextNormalizeOptions:
    """Knobs for deterministic plain-text cleanup before embedding / prompting."""

    unicode_form: str | None = "NFC"
    strip: bool = True
    collapse_whitespace: bool = True
    normalize_newlines: bool = True
    collapse_blank_lines: bool = True
    replace_nul: bool = True


DEFAULT_TEXT_NORMALIZE = TextNormalizeOptions()


def normalize(text: str, *, options: TextNormalizeOptions = DEFAULT_TEXT_NORMALIZE) -> str:
    """Normalize a plain string according to ``options``."""

    out = text
    if options.replace_nul:
        out = out.replace("\x00", "")
    if options.unicode_form:
        out = unicodedata.normalize(options.unicode_form, out)
    if options.normalize_newlines:
        out = out.replace("\r\n", "\n").replace("\r", "\n")
    if options.collapse_whitespace:
        # Collapse runs of spaces/tabs and trim each line. Indentation is not
        # preserved — this targets embedding / prompt normalization, not code.
        out = "\n".join(_WS_RE.sub(" ", line).strip() for line in out.split("\n"))
    if options.collapse_blank_lines:
        out = _BLANK_LINES_RE.sub("\n\n", out)
    if options.strip:
        out = out.strip()
    return out


def normalize_content(
    content: Content,
    *,
    options: TextNormalizeOptions = DEFAULT_TEXT_NORMALIZE,
) -> Content:
    """
    Return content with a normalized body.

    Kind is preserved. Prefer converting to TEXT first when starting from HTML/MD.
    """

    return content.with_body(normalize(content.body, options=options))


def truncate(
    text: str,
    max_chars: int,
    *,
    ellipsis: str = "…",
    hard: bool = True,
) -> str:
    """
    Truncate ``text`` to ``max_chars`` including the ellipsis when truncated.

    When ``hard`` is False, truncation prefers the last whitespace before the cut.
    """

    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if len(ellipsis) >= max_chars:
        return ellipsis[:max_chars]
    budget = max_chars - len(ellipsis)
    if hard:
        return text[:budget] + ellipsis
    head = text[:budget]
    pivot = max(head.rfind(" "), head.rfind("\n"))
    if pivot <= 0:
        return head + ellipsis
    return head[:pivot].rstrip() + ellipsis


def truncate_content(
    content: Content,
    max_chars: int,
    *,
    ellipsis: str = "…",
    hard: bool = True,
) -> Content:
    """Return content with a truncated body (kind preserved)."""

    return content.with_body(
        truncate(content.body, max_chars, ellipsis=ellipsis, hard=hard)
    )


def join_blocks(
    blocks: list[Content | str],
    *,
    separator: str = "\n\n",
    kind: ContentKind = ContentKind.TEXT,
) -> Content:
    """
    Join multiple text-like blocks into a single content value.

    ``Content`` inputs contribute their bodies; bare strings are used directly.
    """

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, Content):
            parts.append(block.body)
        else:
            parts.append(str(block))
    return Content.of(kind, separator.join(parts))
