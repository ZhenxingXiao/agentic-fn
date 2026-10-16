"""OpenAI-compatible Chat Completions adapter (OpenAI, DashScope compat, vLLM...)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_fn.core.context import Context
from agentic_fn.models.adapters._http import post_json
from agentic_fn.models.errors import ModelCapabilityError, ModelPermanentError
from agentic_fn.models.kinds import (
    CAP_CHAT,
    CAP_JSON_SCHEMA,
    CAP_TOOLS,
    CAP_VISION,
    FinishReason,
    PartKind,
    ResponseFormatKind,
    Role,
)
from agentic_fn.models.message import ContentPart, Message, ToolSpec
from agentic_fn.models.protocols import SecretProvider
from agentic_fn.models.request import DecideRequest, ModelRequest
from agentic_fn.models.response import DecideResponse, ModelResponse, Usage


@dataclass(slots=True)
class OpenAICompatibleClient:
    """
    HTTP Chat Completions client against an OpenAI-compatible base URL.

    Works for official OpenAI, Alibaba DashScope compatible mode, OneAPI, vLLM.
    """

    base_url: str = "https://api.openai.com/v1"
    api_key_secret: str = "openai.api_key"
    provider: str = "openai_compatible"
    secrets: SecretProvider | None = None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {CAP_CHAT, CAP_TOOLS, CAP_VISION, CAP_JSON_SCHEMA}
        )
    )

    def complete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Execute chat.completions."""

        _ = ctx
        if CAP_CHAT not in self.capabilities:
            raise ModelCapabilityError(f"{self.provider} cannot complete")
        physical = request.physical_model or "gpt-4o-mini"
        payload = self._build_payload(request, physical)
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            **dict(self.default_headers),
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        data = post_json(url, payload, headers=headers, timeout_s=request.timeout_s or 60.0)
        return self._parse_response(data, request=request, physical=physical)

    async def acomplete(self, request: ModelRequest, ctx: Context) -> ModelResponse:
        """Async wrapper (stdlib HTTP is sync; offload not required for v1)."""

        return self.complete(request, ctx)

    def decide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Not supported on generic chat endpoints."""

        _ = request, ctx
        raise ModelCapabilityError(f"{self.provider} does not implement decide()")

    async def adecide(self, request: DecideRequest, ctx: Context) -> DecideResponse:
        """Async decide stub."""

        return self.decide(request, ctx)

    def stream(self, request: ModelRequest, ctx: Context) -> Iterator[ModelResponse]:
        """Streaming not implemented in stdlib adapter; use complete."""

        _ = request, ctx
        raise ModelCapabilityError(
            f"{self.provider} stream() requires a streaming-capable extra"
        )

    def _api_key(self) -> str:
        """Resolve API key."""

        if self.secrets is None:
            raise ModelPermanentError(
                f"{self.provider} requires secrets=SecretProvider(...)"
            )
        return self.secrets.get(self.api_key_secret)

    def _build_payload(self, request: ModelRequest, physical: str) -> dict[str, Any]:
        """Map ModelRequest to OpenAI chat.completions body."""

        messages = [self._map_message(m) for m in request.messages]
        payload: dict[str, Any] = {
            "model": physical,
            "messages": messages,
        }
        params = request.params
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.top_p is not None:
            payload["top_p"] = params.top_p
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        if params.stop:
            payload["stop"] = list(params.stop)
        if params.seed is not None:
            payload["seed"] = params.seed
        if request.tools:
            payload["tools"] = [self._map_tool(t) for t in request.tools]
        fmt = request.response_format
        if fmt.kind is ResponseFormatKind.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        elif fmt.kind is ResponseFormatKind.JSON_SCHEMA and fmt.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.schema_name or "response",
                    "schema": dict(fmt.json_schema),
                },
            }
        payload.update(dict(params.extra))
        return payload

    def _map_message(self, message: Message) -> dict[str, Any]:
        """Map one message to OpenAI format."""

        role = message.role.value
        # Simple text-only path
        if all(p.kind is PartKind.TEXT for p in message.parts):
            text = message.text()
            out: dict[str, Any] = {"role": role, "content": text}
            if message.name:
                out["name"] = message.name
            return out

        content: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for part in message.parts:
            if part.kind is PartKind.TEXT and part.text is not None:
                content.append({"type": "text", "text": part.text.body})
            elif part.kind is PartKind.IMAGE and part.media is not None:
                url = part.media.uri or ""
                image: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
                if part.media.detail:
                    image["image_url"]["detail"] = part.media.detail
                content.append(image)
            elif part.kind is PartKind.TOOL_CALL:
                import json

                args = part.tool_args.body if part.tool_args else "{}"
                tool_calls.append(
                    {
                        "id": part.tool_call_id or "",
                        "type": "function",
                        "function": {
                            "name": part.tool_name or "",
                            "arguments": args if isinstance(args, str) else json.dumps(args),
                        },
                    }
                )
            elif part.kind is PartKind.TOOL_RESULT:
                return {
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": part.tool_body.body if part.tool_body else "",
                    "name": part.tool_name,
                }
        out = {"role": role, "content": content or ""}
        if tool_calls:
            out["tool_calls"] = tool_calls
        if message.name:
            out["name"] = message.name
        return out

    @staticmethod
    def _map_tool(tool: ToolSpec) -> dict[str, Any]:
        """Map tool spec."""

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        request: ModelRequest,
        physical: str,
    ) -> ModelResponse:
        """Parse OpenAI-style response JSON."""

        import json

        choices = data.get("choices") or []
        if not choices:
            raise ModelPermanentError("provider returned no choices")
        message = choices[0].get("message") or {}
        finish = str(choices[0].get("finish_reason") or "stop")
        finish_reason = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
        }.get(finish, FinishReason.UNKNOWN)

        parts: list[ContentPart] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(ContentPart.text_part(content))
        for call in message.get("tool_calls") or ():
            fn = call.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args_obj = {"raw": args_raw}
            parts.append(
                ContentPart.tool_call_part(
                    tool_call_id=str(call.get("id") or ""),
                    name=str(fn.get("name") or ""),
                    args=args_obj if isinstance(args_obj, dict) else {"value": args_obj},
                )
            )
        if not parts:
            parts.append(ContentPart.text_part(""))

        usage_raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=usage_raw.get("prompt_tokens"),
            output_tokens=usage_raw.get("completion_tokens"),
            total_tokens=usage_raw.get("total_tokens"),
        )
        return ModelResponse(
            logical_model=request.model,
            provider=self.provider,
            physical_model=physical,
            message=Message(role=Role.ASSISTANT, parts=tuple(parts)),
            finish_reason=finish_reason,
            usage=usage,
            request_digest=request.digest(),
            labels={"raw_json": json.dumps(data, ensure_ascii=False)},
        )
