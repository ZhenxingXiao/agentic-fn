"""Kind-aware validation for content values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentic_fn.content.errors import ContentValidationError
from agentic_fn.content.kinds import ContentKind
from agentic_fn.content.limits import DEFAULT_CONTENT_LIMITS, ContentLimits
from agentic_fn.content.model import Content


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """
    Controls how strictly content is checked at construction / ingress.

    ``strict_json``
        Require JSON bodies to parse and respect nesting/key budgets.
    ``reject_nul``
        Reject embedded NUL bytes (common sign of binary mishandling).
    ``require_media_type``
        Require a non-empty media_type field.
    """

    strict_json: bool = True
    reject_nul: bool = True
    require_media_type: bool = False
    enforce_inline: bool = False


DEFAULT_VALIDATION_POLICY = ValidationPolicy()


def validate_content(
    content: Content,
    *,
    limits: ContentLimits = DEFAULT_CONTENT_LIMITS,
    policy: ValidationPolicy = DEFAULT_VALIDATION_POLICY,
) -> Content:
    """
    Validate ``content`` against size limits and kind-specific rules.

    Returns the same instance on success for fluent use.
    """

    if not isinstance(content, Content):
        raise ContentValidationError(f"expected Content, got {type(content)!r}")
    if not isinstance(content.body, str):
        raise ContentValidationError("content.body must be str")
    if not isinstance(content.kind, ContentKind):
        raise ContentValidationError("content.kind must be ContentKind")

    limits.check_body(content.body, enforce_inline=policy.enforce_inline)

    if policy.require_media_type and not content.media_type:
        raise ContentValidationError("media_type is required by validation policy")

    if policy.reject_nul and "\x00" in content.body:
        raise ContentValidationError("content body contains NUL bytes")

    if content.kind is ContentKind.JSON and policy.strict_json:
        _validate_json_body(content.body, limits=limits)

    return content


def _validate_json_body(body: str, *, limits: ContentLimits) -> None:
    """Parse JSON and enforce depth / key budgets."""

    if not body.strip():
        raise ContentValidationError("JSON content body is empty")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ContentValidationError(f"invalid JSON content: {exc}") from exc
    _walk_json_limits(value, depth=0, limits=limits, key_counter=[0])


def _walk_json_limits(
    value: Any,
    *,
    depth: int,
    limits: ContentLimits,
    key_counter: list[int],
) -> None:
    """Recursively enforce JSON complexity limits."""

    if depth > limits.json_max_depth:
        raise ContentValidationError(
            f"JSON nesting exceeds json_max_depth={limits.json_max_depth}"
        )
    if isinstance(value, dict):
        key_counter[0] += len(value)
        if key_counter[0] > limits.json_max_keys:
            raise ContentValidationError(
                f"JSON exceeds json_max_keys={limits.json_max_keys}"
            )
        for nested in value.values():
            _walk_json_limits(
                nested,
                depth=depth + 1,
                limits=limits,
                key_counter=key_counter,
            )
    elif isinstance(value, list):
        for nested in value:
            _walk_json_limits(
                nested,
                depth=depth + 1,
                limits=limits,
                key_counter=key_counter,
            )
