"""
Model IO control plane: contracts, router, runtime, and optional adapters.

Vendor SDKs are extras. All production traffic must go through ``ModelRuntime``
so routing, budget, circuit breaking, safety, and replay stay centralized.
"""

from __future__ import annotations

from typing import Any

from agentic_fn.models.codec import (
    get_decide_response,
    get_messages,
    get_model_response,
    put_decide_response,
    put_messages,
    put_model_response,
)
from agentic_fn.models.effect import make_decide_effect, make_model_effect
from agentic_fn.models.errors import (
    ModelCapabilityError,
    ModelConfigurationError,
    ModelError,
    ModelExhaustedError,
    ModelPermanentError,
    ModelReplayError,
    ModelRouteError,
    ModelSafetyError,
    ModelTransientError,
)
from agentic_fn.models.factory import build_model_runtime, demo_runtime
from agentic_fn.models.fake import FakeClient
from agentic_fn.models.kinds import (
    CAP_AUDIO,
    CAP_CHAT,
    CAP_DECIDE,
    CAP_EMBED,
    CAP_JSON_SCHEMA,
    CAP_STREAM,
    CAP_TOOLS,
    CAP_VISION,
    FinishReason,
    Modality,
    PartKind,
    ResponseFormatKind,
    Role,
    RouteStrategy,
)
from agentic_fn.models.message import ContentPart, MediaRef, Message, ToolSpec
from agentic_fn.models.policy import DEFAULT_MODEL_CALL_POLICY, ModelCallPolicy
from agentic_fn.models.protocols import EnvSecretProvider
from agentic_fn.models.registry import ClientRegistry
from agentic_fn.models.request import (
    DecideRequest,
    ModelParams,
    ModelRequest,
    ResponseFormat,
    RoutingHints,
)
from agentic_fn.models.response import DecideResponse, ModelResponse, RouteTrace, Usage
from agentic_fn.models.router import (
    Candidate,
    DefaultModelRouter,
    FailoverPolicy,
    ProviderHealthView,
    RouteEntry,
    RouteTable,
    validate_route_table,
)
from agentic_fn.models.runtime import ModelRuntime
from agentic_fn.models.safety import (
    CompositeOutputFilter,
    DenySubstringFilter,
    NoOpOutputFilter,
)

__all__ = [
    "CAP_AUDIO",
    "CAP_CHAT",
    "CAP_DECIDE",
    "CAP_EMBED",
    "CAP_JSON_SCHEMA",
    "CAP_STREAM",
    "CAP_TOOLS",
    "CAP_VISION",
    "Candidate",
    "ClientRegistry",
    "CompositeOutputFilter",
    "ContentPart",
    "DEFAULT_MODEL_CALL_POLICY",
    "DecideRequest",
    "DecideResponse",
    "DefaultModelRouter",
    "DenySubstringFilter",
    "EnvSecretProvider",
    "FakeClient",
    "FailoverPolicy",
    "FinishReason",
    "MediaRef",
    "Message",
    "Modality",
    "ModelCallPolicy",
    "ModelCapabilityError",
    "ModelConfigurationError",
    "ModelError",
    "ModelExhaustedError",
    "ModelParams",
    "ModelPermanentError",
    "ModelReplayError",
    "ModelRequest",
    "ModelResponse",
    "ModelRouteError",
    "ModelRuntime",
    "ModelSafetyError",
    "ModelTransientError",
    "NoOpOutputFilter",
    "PartKind",
    "ProviderHealthView",
    "ResponseFormat",
    "ResponseFormatKind",
    "Role",
    "RouteEntry",
    "RouteStrategy",
    "RouteTable",
    "RouteTrace",
    "RoutingHints",
    "ToolSpec",
    "Usage",
    "build_model_runtime",
    "demo_runtime",
    "get_decide_response",
    "get_messages",
    "get_model_response",
    "make_decide_effect",
    "make_model_effect",
    "put_decide_response",
    "put_messages",
    "put_model_response",
    "validate_route_table",
]


def __getattr__(name: str) -> Any:
    """Lazy adapter exports."""

    if name in {
        "OpenAIClient",
        "OpenAICompatibleClient",
        "LayaClient",
        "HttpJsonDecideClient",
    }:
        from agentic_fn.models import adapters as adapters_pkg

        return getattr(adapters_pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
