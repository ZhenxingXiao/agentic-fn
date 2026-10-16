"""Enumerations for the model IO subsystem."""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Modality(str, Enum):
    """Logical capability class for routing and validation."""

    CHAT = "chat"
    DECIDE = "decide"
    MULTIMODAL = "multimodal"
    EMBED = "embed"


class FinishReason(str, Enum):
    """Normalized completion finish reason."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class PartKind(str, Enum):
    """Message content part discriminator."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class RouteStrategy(str, Enum):
    """Candidate selection strategy used by ``ModelRouter``."""

    PRIORITY = "priority"
    WEIGHTED = "weighted"
    STICKY = "sticky"
    LATENCY_AWARE = "latency_aware"


class ResponseFormatKind(str, Enum):
    """Requested response shape."""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


# Capability tokens used in route candidate declarations and client ads.
CAP_CHAT = "chat"
CAP_TOOLS = "tools"
CAP_VISION = "vision"
CAP_AUDIO = "audio"
CAP_DECIDE = "decide"
CAP_JSON_SCHEMA = "json_schema"
CAP_STREAM = "stream"
CAP_EMBED = "embed"

MODEL_TYPE_MARKER = "agentic_fn.model_response"
DECIDE_TYPE_MARKER = "agentic_fn.decide_response"
MESSAGE_TYPE_MARKER = "agentic_fn.message"
REQUEST_TYPE_MARKER = "agentic_fn.model_request"
MEDIA_TYPE_MARKER = "agentic_fn.media_ref"
MEDIA_REF_TYPE_MARKER = MEDIA_TYPE_MARKER
MODEL_SCHEMA_VERSION = "1"
