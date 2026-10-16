"""Helpers to assemble a production ``ModelRuntime`` from parts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agentic_fn.models.fake import FakeClient
from agentic_fn.models.kinds import (
    CAP_CHAT,
    CAP_DECIDE,
    CAP_JSON_SCHEMA,
    CAP_STREAM,
    CAP_TOOLS,
    CAP_VISION,
    Modality,
    RouteStrategy,
)
from agentic_fn.models.policy import ModelCallPolicy
from agentic_fn.models.protocols import ModelClient, OutputFilter, SecretProvider
from agentic_fn.models.registry import ClientRegistry
from agentic_fn.models.router.config import Candidate, RouteEntry, RouteTable
from agentic_fn.models.router.health import ProviderHealthView
from agentic_fn.models.router.router import DefaultModelRouter
from agentic_fn.models.runtime import ModelRuntime
from agentic_fn.models.safety import NoOpOutputFilter


def build_model_runtime(
    *,
    table: RouteTable,
    clients: Mapping[str, ModelClient] | ClientRegistry,
    secrets: SecretProvider | None = None,
    policy: ModelCallPolicy | None = None,
    output_filter: OutputFilter | None = None,
    validate: bool = True,
) -> ModelRuntime:
    """
    Wire router + registry + runtime.

    ``clients`` keys must match ``Candidate.provider`` values in ``table``.
    """

    registry = clients if isinstance(clients, ClientRegistry) else ClientRegistry()
    if not isinstance(clients, ClientRegistry):
        for provider, client in clients.items():
            registry.register(provider, client, replace=True)

    health = ProviderHealthView()
    router = DefaultModelRouter(
        table=table,
        health=health,
        known_providers=registry.providers(),
        validate_on_init=validate,
    )
    return ModelRuntime(
        router=router,
        clients=registry,
        policy=policy or ModelCallPolicy(),
        secrets=secrets,
        health=health,
        output_filter=output_filter or NoOpOutputFilter(),
    )


def demo_runtime(
    *,
    fake_reply: str = "ok",
    include_decide: bool = True,
) -> ModelRuntime:
    """
    Batteries-included offline runtime for tests and local demos.

    Logical models:
    - ``chat.default`` -> fake chat
    - ``decide.policy`` -> fake decide (when include_decide)
    - ``vision.ocr`` -> fake chat with vision capability
    """

    chat = FakeClient(provider="fake", default_reply=fake_reply)
    decide = FakeClient(provider="fake_decide", default_action="allow")
    clients: dict[str, ModelClient] = {"fake": chat}
    entries: dict[str, RouteEntry] = {
        "chat.default": RouteEntry(
            modality=Modality.CHAT,
            strategy=RouteStrategy.PRIORITY,
            candidates=(
                Candidate(
                    provider="fake",
                    physical_model="fake-model",
                    priority=10,
                    capabilities=frozenset(
                        {
                            CAP_CHAT,
                            CAP_TOOLS,
                            CAP_VISION,
                            CAP_JSON_SCHEMA,
                            CAP_STREAM,
                        }
                    ),
                ),
            ),
        ),
        "vision.ocr": RouteEntry(
            modality=Modality.MULTIMODAL,
            strategy=RouteStrategy.PRIORITY,
            candidates=(
                Candidate(
                    provider="fake",
                    physical_model="fake-vision",
                    priority=10,
                    capabilities=frozenset({CAP_CHAT, CAP_VISION, CAP_STREAM}),
                ),
            ),
        ),
    }
    if include_decide:
        clients["fake_decide"] = decide
        entries["decide.policy"] = RouteEntry(
            modality=Modality.DECIDE,
            strategy=RouteStrategy.PRIORITY,
            candidates=(
                Candidate(
                    provider="fake_decide",
                    physical_model="fake-decide",
                    priority=10,
                    capabilities=frozenset({CAP_DECIDE}),
                ),
            ),
        )
    return build_model_runtime(table=RouteTable(entries=entries), clients=clients)
