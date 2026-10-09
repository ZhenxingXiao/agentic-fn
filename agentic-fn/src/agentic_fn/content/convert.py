"""Explicit content-kind conversion with lossy / strict policies."""

from __future__ import annotations

from dataclasses import dataclass
from agentic_fn.content import html_ops, json_ops, md_ops, text_ops
from agentic_fn.content.errors import ContentConversionError, ContentKindError
from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits
from agentic_fn.content.model import Content
from agentic_fn.content.protocols import HtmlSanitizer, MarkdownRenderer


@dataclass(frozen=True, slots=True)
class ConversionPolicy:
    """
    Policy for ``convert``.

    ``lossy``
        Allow information-discarding transforms (HTML→text, MD→text).
    ``normalize_text``
        After landing on TEXT, apply ``text_ops.normalize``.
    ``canonicalize_json``
        When producing JSON, re-serialize with sorted keys.
    """

    lossy: bool = True
    normalize_text: bool = True
    canonicalize_json: bool = True


DEFAULT_CONVERSION_POLICY = ConversionPolicy()


def convert(
    content: Content,
    target: ContentKind | str,
    *,
    policy: ConversionPolicy = DEFAULT_CONVERSION_POLICY,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    markdown_renderer: MarkdownRenderer | None = None,
    html_sanitizer: HtmlSanitizer | None = None,
) -> Content:
    """
    Convert ``content`` to ``target`` kind.

    Supported matrix (production core, no optional deps required unless noted):

    ==============  ====  ====  ========  ====
    from \\ to       TEXT  JSON  MARKDOWN  HTML
    ==============  ====  ====  ========  ====
    TEXT            id    wrap  wrap      escape
    JSON            dump  id    wrap*     wrap*
    MARKDOWN        strip wrap* id        render†
    HTML            extract wrap* wrap*   sanitize‡ / id
    ==============  ====  ====  ========  ====

    ``*`` lossy / non-semantic wrap (body copied, kind retagged) — only when
    ``policy.lossy`` is True and explicitly useful for transport tagging.
    ``†`` requires ``markdown_renderer``.
    ``‡`` optional ``html_sanitizer`` applied when target is HTML.
    """

    target_kind = ContentKind.parse(target)
    if content.kind is target_kind:
        out = content
        if target_kind is ContentKind.HTML and html_sanitizer is not None:
            out = html_ops.sanitize(out, html_sanitizer)
        if target_kind is ContentKind.JSON and policy.canonicalize_json:
            out = json_ops.canonicalize(out)
        if target_kind is ContentKind.TEXT and policy.normalize_text:
            out = text_ops.normalize_content(out)
        return out

    result = _convert_dispatch(
        content,
        target_kind,
        policy=policy,
        markdown_renderer=markdown_renderer,
        html_sanitizer=html_sanitizer,
    )
    # Re-validate under caller limits after conversion.
    from agentic_fn.content.validate import validate_content

    return validate_content(result, limits=limits)


def _convert_dispatch(
    content: Content,
    target: ContentKind,
    *,
    policy: ConversionPolicy,
    markdown_renderer: MarkdownRenderer | None,
    html_sanitizer: HtmlSanitizer | None,
) -> Content:
    """Internal conversion matrix."""

    src = content.kind

    # --- to TEXT -------------------------------------------------------------
    if target is ContentKind.TEXT:
        if src is ContentKind.HTML:
            if not policy.lossy:
                raise ContentConversionError("HTML→TEXT is lossy; set policy.lossy=True")
            out = html_ops.to_text_content(content)
        elif src is ContentKind.MARKDOWN:
            if not policy.lossy:
                raise ContentConversionError(
                    "MARKDOWN→TEXT is lossy; set policy.lossy=True"
                )
            out = md_ops.to_text_content(content, renderer=markdown_renderer)
        elif src is ContentKind.JSON:
            # Body is already a JSON document string; retag as TEXT.
            out = Content.text(
                content.body,
                meta=content.derive_meta(conversion="json_to_text"),
            )
        else:
            raise ContentConversionError(f"unsupported conversion {src.value}->text")
        if policy.normalize_text:
            out = text_ops.normalize_content(out)
        return out

    # --- to JSON -------------------------------------------------------------
    if target is ContentKind.JSON:
        if src is ContentKind.TEXT:
            # Interpret body as a JSON document string.
            return Content.json(content.body)
        if src in {ContentKind.MARKDOWN, ContentKind.HTML}:
            if not policy.lossy:
                raise ContentConversionError(
                    f"{src.value}->JSON without structured parse is a wrap; "
                    "set policy.lossy=True to allow embedding as a JSON string value"
                )
            # Wrap original body as a JSON object (transport tagging).
            wrapped = json_ops.from_value(
                {"kind": src.value, "body": content.body},
                meta=content.derive_meta(conversion=f"{src.value}_wrap_json"),
            )
            return wrapped if not policy.canonicalize_json else json_ops.canonicalize(wrapped)
        raise ContentConversionError(f"unsupported conversion {src.value}->json")

    # --- to MARKDOWN ---------------------------------------------------------
    if target is ContentKind.MARKDOWN:
        if src is ContentKind.TEXT:
            # Plain text is valid Markdown.
            return Content.markdown(content.body)
        if src is ContentKind.HTML:
            if not policy.lossy:
                raise ContentConversionError(
                    "HTML→MARKDOWN requires an external converter; "
                    "core only allows lossy extract→text then wrap"
                )
            text = html_ops.extract_text(content.body)
            return Content.markdown(text)
        if src is ContentKind.JSON:
            if not policy.lossy:
                raise ContentConversionError("JSON→MARKDOWN is a non-semantic wrap")
            return Content.markdown(f"```json\n{content.body}\n```")
        raise ContentConversionError(f"unsupported conversion {src.value}→markdown")

    # --- to HTML -------------------------------------------------------------
    if target is ContentKind.HTML:
        if src is ContentKind.TEXT:
            escaped = html_ops.escape(content.body).replace("\n", "<br>\n")
            out = Content.html(f"<p>{escaped}</p>")
        elif src is ContentKind.MARKDOWN:
            if markdown_renderer is None:
                raise ContentConversionError(
                    "MARKDOWN→HTML requires markdown_renderer=..."
                )
            out = md_ops.to_html_content(content, renderer=markdown_renderer)
        elif src is ContentKind.JSON:
            if not policy.lossy:
                raise ContentConversionError("JSON→HTML is a non-semantic wrap")
            escaped = html_ops.escape(content.body)
            out = Content.html(f"<pre><code>{escaped}</code></pre>")
        else:
            raise ContentConversionError(f"unsupported conversion {src.value}→html")
        if html_sanitizer is not None:
            out = html_ops.sanitize(out, html_sanitizer)
        return out

    raise ContentKindError(f"unknown target kind: {target!r}")


def ensure_convertible(source: ContentKind, target: ContentKind) -> None:
    """
    Cheap static check used by builders / harnesses.

    Does not consider injected renderers; MARKDOWN→HTML is reported as
    ``conditional``.
    """

    if source is target:
        return
    allowed = {
        (ContentKind.HTML, ContentKind.TEXT),
        (ContentKind.MARKDOWN, ContentKind.TEXT),
        (ContentKind.JSON, ContentKind.TEXT),
        (ContentKind.TEXT, ContentKind.JSON),
        (ContentKind.TEXT, ContentKind.MARKDOWN),
        (ContentKind.TEXT, ContentKind.HTML),
        (ContentKind.HTML, ContentKind.MARKDOWN),
        (ContentKind.JSON, ContentKind.MARKDOWN),
        (ContentKind.JSON, ContentKind.HTML),
        (ContentKind.MARKDOWN, ContentKind.HTML),
        (ContentKind.MARKDOWN, ContentKind.JSON),
        (ContentKind.HTML, ContentKind.JSON),
    }
    if (source, target) not in allowed:
        raise ContentConversionError(
            f"conversion {source.value}→{target.value} is not supported"
        )
