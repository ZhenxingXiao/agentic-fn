"""Messages, content parts, media refs, and tool specs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from agentic_fn.content.codec import content_from_dict, content_to_dict, is_content_dict
from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.model import Content
from agentic_fn.content.refs import ContentRef
from agentic_fn.models.errors import ModelConfigurationError
from agentic_fn.models.kinds import (
    MEDIA_TYPE_MARKER,
    MESSAGE_TYPE_MARKER,
    MODEL_SCHEMA_VERSION,
    PartKind,
    Role,
)


@dataclass(frozen=True, slots=True)
class MediaRef:
    """
    Reference to binary media kept out of State by default.

    Prefer ``uri`` (https / s3 / signed URL) or ``content_ref`` over inlined bytes.
    """

    mime: str
    uri: str | None = None
    content_ref: ContentRef | None = None
    digest: str | None = None
    bytes_hint: int | None = None
    detail: str | None = None  # e.g. OpenAI vision detail: low|high|auto
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require at least one locator."""

        if not self.uri and self.content_ref is None:
            raise ModelConfigurationError("MediaRef requires uri or content_ref")

    def to_dict(self) -> dict[str, Any]:
        """Serialize for State / wire transport."""

        return {
            "_type": MEDIA_TYPE_MARKER,
            "schema_version": MODEL_SCHEMA_VERSION,
            "mime": self.mime,
            "uri": self.uri,
            "content_ref": self.content_ref.to_dict() if self.content_ref else None,
            "digest": self.digest,
            "bytes_hint": self.bytes_hint,
            "detail": self.detail,
            "labels": dict(self.labels),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> MediaRef:
        """Deserialize a media reference."""

        ref_raw = data.get("content_ref")
        return MediaRef(
            mime=str(data["mime"]),
            uri=data.get("uri"),
            content_ref=ContentRef.from_dict(ref_raw) if ref_raw else None,
            digest=data.get("digest"),
            bytes_hint=data.get("bytes_hint"),
            detail=data.get("detail"),
            labels=dict(data.get("labels") or {}),
        )


@dataclass(frozen=True, slots=True)
class ContentPart:
    """Tagged union member for multimodal message bodies."""

    kind: PartKind
    text: Content | None = None
    media: MediaRef | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: Content | None = None  # JSON content for tool call args
    tool_body: Content | None = None  # tool result body

    @staticmethod
    def text_part(body: str | Content, *, kind: ContentKind = ContentKind.TEXT) -> ContentPart:
        """Build a text part from a string or Content value."""

        content = body if isinstance(body, Content) else Content.of(kind, body)
        return ContentPart(kind=PartKind.TEXT, text=content)

    @staticmethod
    def image_part(media: MediaRef) -> ContentPart:
        """Build an image part."""

        return ContentPart(kind=PartKind.IMAGE, media=media)

    @staticmethod
    def tool_call_part(
        *,
        tool_call_id: str,
        name: str,
        args: Mapping[str, Any] | Content,
    ) -> ContentPart:
        """Build a tool-call part."""

        from agentic_fn.content import json_ops

        args_content = (
            args
            if isinstance(args, Content)
            else json_ops.from_value(dict(args))
        )
        return ContentPart(
            kind=PartKind.TOOL_CALL,
            tool_call_id=tool_call_id,
            tool_name=name,
            tool_args=args_content,
        )

    @staticmethod
    def tool_result_part(
        *,
        tool_call_id: str,
        name: str,
        body: str | Content | Mapping[str, Any],
    ) -> ContentPart:
        """Build a tool-result part."""

        from agentic_fn.content import json_ops

        if isinstance(body, Content):
            content = body
        elif isinstance(body, Mapping):
            content = json_ops.from_value(dict(body))
        else:
            content = Content.text(str(body))
        return ContentPart(
            kind=PartKind.TOOL_RESULT,
            tool_call_id=tool_call_id,
            tool_name=name,
            tool_body=content,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the part."""

        return {
            "kind": self.kind.value,
            "text": content_to_dict(self.text) if self.text else None,
            "media": self.media.to_dict() if self.media else None,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_args": content_to_dict(self.tool_args) if self.tool_args else None,
            "tool_body": content_to_dict(self.tool_body) if self.tool_body else None,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ContentPart:
        """Deserialize a part."""

        def _content(value: Any) -> Content | None:
            if value is None:
                return None
            if isinstance(value, Content):
                return value
            if is_content_dict(value):
                return content_from_dict(value)
            raise ModelConfigurationError(f"invalid content payload: {type(value)!r}")

        media_raw = data.get("media")
        return ContentPart(
            kind=PartKind(str(data["kind"])),
            text=_content(data.get("text")),
            media=MediaRef.from_dict(media_raw) if media_raw else None,
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            tool_args=_content(data.get("tool_args")),
            tool_body=_content(data.get("tool_body")),
        )


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a model conversation."""

    role: Role
    parts: tuple[ContentPart, ...]
    name: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid4()))

    @staticmethod
    def system(text: str) -> Message:
        """Create a system message."""

        return Message(role=Role.SYSTEM, parts=(ContentPart.text_part(text),))

    @staticmethod
    def user(text: str) -> Message:
        """Create a user text message."""

        return Message(role=Role.USER, parts=(ContentPart.text_part(text),))

    @staticmethod
    def assistant(text: str) -> Message:
        """Create an assistant text message."""

        return Message(role=Role.ASSISTANT, parts=(ContentPart.text_part(text),))

    @staticmethod
    def user_multimodal(text: str, *media: MediaRef) -> Message:
        """Create a user message with text and media parts."""

        parts: list[ContentPart] = [ContentPart.text_part(text)]
        for item in media:
            if item.mime.startswith("image/"):
                parts.append(ContentPart.image_part(item))
            elif item.mime.startswith("audio/"):
                parts.append(ContentPart(kind=PartKind.AUDIO, media=item))
            elif item.mime.startswith("video/"):
                parts.append(ContentPart(kind=PartKind.VIDEO, media=item))
            else:
                parts.append(ContentPart.image_part(item))
        return Message(role=Role.USER, parts=tuple(parts))

    def text(self) -> str:
        """Concatenate text parts (lossy view for logging / simple prompts)."""

        chunks: list[str] = []
        for part in self.parts:
            if part.kind is PartKind.TEXT and part.text is not None:
                chunks.append(part.text.body)
        return "\n".join(chunks)

    def has_vision(self) -> bool:
        """Return whether any image/video part is present."""

        return any(p.kind in {PartKind.IMAGE, PartKind.VIDEO} for p in self.parts)

    def has_audio(self) -> bool:
        """Return whether any audio part is present."""

        return any(p.kind is PartKind.AUDIO for p in self.parts)

    def with_parts(self, *parts: ContentPart) -> Message:
        """Return a copy with replaced parts."""

        return replace(self, parts=tuple(parts))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message."""

        return {
            "_type": MESSAGE_TYPE_MARKER,
            "schema_version": MODEL_SCHEMA_VERSION,
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
            "name": self.name,
            "message_id": self.message_id,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Message:
        """Deserialize a message."""

        return Message(
            role=Role(str(data["role"])),
            parts=tuple(ContentPart.from_dict(p) for p in data.get("parts") or ()),
            name=data.get("name"),
            message_id=str(data.get("message_id") or uuid4()),
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """JSON-schema style tool declaration for function calling."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool spec."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ToolSpec:
        """Deserialize a tool spec."""

        return ToolSpec(
            name=str(data["name"]),
            description=str(data.get("description") or ""),
            parameters=dict(data.get("parameters") or {}),
        )


def messages_require_capabilities(messages: Sequence[Message]) -> frozenset[str]:
    """Infer required capability tokens from message parts."""

    from agentic_fn.models.kinds import CAP_AUDIO, CAP_TOOLS, CAP_VISION

    needed: set[str] = set()
    for message in messages:
        if message.has_vision():
            needed.add(CAP_VISION)
        if message.has_audio():
            needed.add(CAP_AUDIO)
        for part in message.parts:
            if part.kind in {PartKind.TOOL_CALL, PartKind.TOOL_RESULT}:
                needed.add(CAP_TOOLS)
    return frozenset(needed)
