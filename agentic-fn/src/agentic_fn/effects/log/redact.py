"""Secret/PII redaction helpers for structured log fields."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "private_key",
        "client_secret",
        "session",
        "cookie",
        "credentials",
    }
)


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """
    Policy controlling how sensitive values are masked before emit.

    Matching is case-insensitive on mapping keys. Nested mappings/lists are
    walked recursively up to ``max_depth``.
    """

    sensitive_keys: frozenset[str] = field(default_factory=lambda: _DEFAULT_SENSITIVE_KEYS)
    mask: str = "***REDACTED***"
    max_depth: int = 6
    extra_key_patterns: tuple[re.Pattern[str], ...] = ()

    def is_sensitive_key(self, key: str) -> bool:
        """Return whether ``key`` should be redacted."""

        lowered = key.lower()
        if lowered in self.sensitive_keys:
            return True
        normalized = lowered.replace("-", "_")
        if normalized in self.sensitive_keys:
            return True
        return any(pattern.search(key) is not None for pattern in self.extra_key_patterns)

    def redact_mapping(self, data: Mapping[str, Any], *, depth: int = 0) -> dict[str, Any]:
        """Return a deep-copied mapping with sensitive values masked."""

        if depth >= self.max_depth:
            return {"_truncated": True}
        result: dict[str, Any] = {}
        for key, value in data.items():
            if self.is_sensitive_key(str(key)):
                result[key] = self.mask
            else:
                result[key] = self.redact_value(value, depth=depth + 1)
        return result

    def redact_value(self, value: Any, *, depth: int = 0) -> Any:
        """Redact a single value according to this policy."""

        if depth >= self.max_depth:
            return self.mask if isinstance(value, (str, bytes)) else {"_truncated": True}
        if isinstance(value, Mapping):
            return self.redact_mapping(value, depth=depth)
        if isinstance(value, list):
            return [self.redact_value(item, depth=depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item, depth=depth + 1) for item in value)
        return value


def default_redaction_policy(
    *,
    extra_keys: Iterable[str] = (),
    extra_patterns: Iterable[str] = (),
) -> RedactionPolicy:
    """Build a redaction policy with optional enterprise extensions."""

    keys = set(_DEFAULT_SENSITIVE_KEYS)
    keys.update(k.lower() for k in extra_keys)
    patterns = tuple(re.compile(p, re.IGNORECASE) for p in extra_patterns)
    return RedactionPolicy(sensitive_keys=frozenset(keys), extra_key_patterns=patterns)


def apply_redaction_to_record_fields(
    fields: Mapping[str, Any],
    policy: RedactionPolicy,
) -> dict[str, Any]:
    """Convenience wrapper used by redacting sinks."""

    return policy.redact_mapping(fields)
