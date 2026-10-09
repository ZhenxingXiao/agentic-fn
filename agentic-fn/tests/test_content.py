"""Production-oriented tests for the content subsystem."""

from __future__ import annotations

import unittest

from agentic_fn import (
    Content,
    ContentKind,
    ContentLimits,
    ContentRef,
    ConversionPolicy,
    MemoryContentStore,
    State,
    convert,
    externalize,
    get_content,
    put_content,
    resolve,
)
from agentic_fn.content.errors import (
    ContentConversionError,
    ContentKindError,
    ContentLimitError,
    ContentStoreError,
    ContentValidationError,
)
from agentic_fn.content import json_ops, md_ops, text_ops


class ContentModelTests(unittest.TestCase):
    """Construction, guards, and serialization."""

    def test_factories_and_ensure_kind(self) -> None:
        text = Content.text("hello")
        self.assertEqual(text.kind, ContentKind.TEXT)
        text.ensure_kind(ContentKind.TEXT)
        with self.assertRaises(ContentKindError):
            text.ensure_kind(ContentKind.JSON)

    def test_json_validation_rejects_invalid(self) -> None:
        with self.assertRaises(ContentValidationError):
            Content.json("{not-json")

    def test_json_round_trip_value(self) -> None:
        content = json_ops.from_value({"a": 1, "b": [True, None]})
        self.assertEqual(content.kind, ContentKind.JSON)
        self.assertEqual(json_ops.to_value(content), {"a": 1, "b": [True, None]})

    def test_dict_round_trip_stable_for_state(self) -> None:
        original = Content.markdown("# Title\n\nbody", meta=None)
        encoded = original.to_dict()
        restored = Content.from_dict(encoded)
        self.assertEqual(restored.kind, ContentKind.MARKDOWN)
        self.assertEqual(restored.body, original.body)
        self.assertEqual(restored.digest(), original.digest())

    def test_state_put_get_preserves_digest(self) -> None:
        content = Content.html("<p>hi</p>")
        state = put_content(State.empty(), "doc", content)
        # State digest must remain JSON-serializable.
        _ = state.digest()
        loaded = get_content(state, "doc")
        assert isinstance(loaded, Content)
        self.assertEqual(loaded.digest(), content.digest())

    def test_nul_rejected(self) -> None:
        with self.assertRaises(ContentValidationError):
            Content.text("a\x00b")

    def test_size_limit(self) -> None:
        limits = ContentLimits(
            max_chars=10,
            max_bytes=40,
            inline_max_chars=10,
            preview_chars=5,
        )
        with self.assertRaises(ContentLimitError):
            Content.text("abcdefghijk", limits=limits)


class JsonOpsTests(unittest.TestCase):
    """JSON pointer / merge / canonicalize."""

    def test_pointer_and_merge(self) -> None:
        content = json_ops.from_value({"user": {"name": "a"}, "tags": ["x", "y"]})
        self.assertEqual(json_ops.pointer_get(content, "/user/name"), "a")
        self.assertEqual(json_ops.pointer_get(content, "/tags/1"), "y")
        self.assertIsNone(json_ops.pointer_get(content, "/missing"))
        merged = json_ops.merge_objects(content, {"user": {"role": "admin"}, "ok": True})
        self.assertEqual(
            json_ops.to_value(merged),
            {"user": {"name": "a", "role": "admin"}, "tags": ["x", "y"], "ok": True},
        )

    def test_canonicalize_sorts_keys(self) -> None:
        raw = Content.json('{"b":1,"a":2}')
        canon = json_ops.canonicalize(raw)
        self.assertEqual(canon.body, '{"a":2,"b":1}')


class HtmlMarkdownTests(unittest.TestCase):
    """Builtin HTML / Markdown normalizers."""

    def test_html_extract_drops_script(self) -> None:
        html = Content.html(
            "<html><head><style>x{}</style></head>"
            "<body><script>evil()</script><p>Hello &amp; world</p></body></html>"
        )
        text = html.as_text()
        self.assertEqual(text.kind, ContentKind.TEXT)
        self.assertIn("Hello & world", text.body)
        self.assertNotIn("evil", text.body)
        self.assertNotIn("x{}", text.body)

    def test_markdown_strip(self) -> None:
        md = Content.markdown("# Hello\n\nThis is **bold** and a [link](http://x)")
        text = md_ops.to_text_content(md)
        self.assertIn("Hello", text.body)
        self.assertIn("bold", text.body)
        self.assertIn("link", text.body)
        self.assertNotIn("**", text.body)
        self.assertNotIn("http://x", text.body)

    def test_markdown_to_html_requires_renderer(self) -> None:
        md = Content.markdown("# Hi")
        with self.assertRaises(ContentConversionError):
            convert(md, ContentKind.HTML, policy=ConversionPolicy(lossy=True))

    def test_markdown_to_html_with_renderer(self) -> None:
        class DummyRenderer:
            def render_html(self, markdown: str) -> str:
                return f"<p>{markdown}</p>"

            def render_text(self, markdown: str) -> str:
                return markdown

        md = Content.markdown("hi")
        html = convert(
            md,
            ContentKind.HTML,
            markdown_renderer=DummyRenderer(),
        )
        self.assertEqual(html.kind, ContentKind.HTML)
        self.assertIn("<p>hi</p>", html.body)


class ConversionPolicyTests(unittest.TestCase):
    """Lossy conversion gates."""

    def test_lossy_gate(self) -> None:
        html = Content.html("<p>x</p>")
        with self.assertRaises(ContentConversionError):
            convert(html, ContentKind.TEXT, policy=ConversionPolicy(lossy=False))
        ok = convert(html, ContentKind.TEXT, policy=ConversionPolicy(lossy=True))
        self.assertEqual(ok.kind, ContentKind.TEXT)

    def test_text_to_json_requires_valid_document(self) -> None:
        with self.assertRaises(ContentValidationError):
            convert(Content.text("not-json"), ContentKind.JSON)


class ContentRefTests(unittest.TestCase):
    """Externalization and digest verification."""

    def test_externalize_and_resolve(self) -> None:
        store = MemoryContentStore()
        limits = ContentLimits(
            max_chars=10_000,
            max_bytes=40_000,
            inline_max_chars=10,
            preview_chars=20,
        )
        content = Content.text("this is longer than ten", limits=limits)
        ref = externalize(content, store, limits=limits)
        self.assertIsInstance(ref, ContentRef)
        assert isinstance(ref, ContentRef)
        restored = resolve(ref, store, limits=limits)
        self.assertEqual(restored.body, content.body)
        self.assertEqual(restored.digest(), content.digest())

    def test_digest_mismatch_detected(self) -> None:
        store = MemoryContentStore()
        content = Content.text("abc")
        ref = externalize(content, store, force=True)
        assert isinstance(ref, ContentRef)
        # Tamper with stored body.
        store._blobs[ref.uri] = "tampered"
        with self.assertRaises(ContentStoreError):
            resolve(ref, store)

    def test_under_threshold_keeps_inline(self) -> None:
        store = MemoryContentStore()
        content = Content.text("short")
        out = externalize(content, store)
        self.assertIsInstance(out, Content)


class TextOpsTests(unittest.TestCase):
    """Normalization and truncation."""

    def test_normalize_and_truncate(self) -> None:
        dirty = "  hello\r\n\r\n\r\n  world \t "
        clean = text_ops.normalize(dirty)
        self.assertEqual(clean, "hello\n\nworld")
        self.assertEqual(text_ops.truncate("abcdef", 4, ellipsis=".."), "ab..")


class PipelineIntegrationTests(unittest.TestCase):
    """Content flowing through a sync capability step."""

    def test_capability_converts_html_in_state(self) -> None:
        from agentic_fn import Context, SyncPipeline, capability

        @capability
        def html_to_text_step(state: State, ctx: Context) -> State:
            _ = ctx
            doc = get_content(state, "doc")
            assert isinstance(doc, Content)
            return put_content(state, "doc", doc.as_text())

        pipe = (
            SyncPipeline.builder("content_demo")
            .capability("html_to_text", html_to_text_step)
            .build()
        )
        initial = put_content(
            State.empty(),
            "doc",
            Content.html("<p>Hello <b>Agent</b></p>"),
        )
        result = pipe.run(initial)
        self.assertTrue(result.ok)
        out = get_content(result.state, "doc")
        assert isinstance(out, Content)
        self.assertEqual(out.kind, ContentKind.TEXT)
        self.assertIn("Hello", out.body)
        self.assertIn("Agent", out.body)


if __name__ == "__main__":
    unittest.main()
