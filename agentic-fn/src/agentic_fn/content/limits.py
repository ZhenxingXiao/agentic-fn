"""Size and complexity limits for production content handling."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_fn.content.errors import ContentConfigurationError, ContentLimitError


@dataclass(frozen=True, slots=True)
class ContentLimits:
    """
    Hard caps applied at content construction / validation boundaries.

    Large bodies should be externalized via ``ContentRef`` once they exceed
    ``inline_max_chars`` (State stays small and serializable).
    """

    # Absolute ceiling for any inlined body (defense in depth).
    max_chars: int = 2_000_000
    max_bytes: int = 4_000_000
    # Soft threshold recommending / requiring externalization.
    inline_max_chars: int = 256_000
    # JSON-specific nesting / container limits.
    json_max_depth: int = 64
    json_max_keys: int = 50_000
    # Preview / log truncation defaults.
    preview_chars: int = 240
    digest_prefix_chars: int = 16

    def __post_init__(self) -> None:
        """Validate limit configuration."""

        if self.max_chars <= 0 or self.max_bytes <= 0:
            raise ContentConfigurationError("max_chars and max_bytes must be positive")
        if self.inline_max_chars <= 0 or self.inline_max_chars > self.max_chars:
            raise ContentConfigurationError(
                "inline_max_chars must be in (0, max_chars]"
            )
        if self.json_max_depth <= 0 or self.json_max_keys <= 0:
            raise ContentConfigurationError("json limits must be positive")
        if self.preview_chars <= 0:
            raise ContentConfigurationError("preview_chars must be positive")

    def check_body(self, body: str, *, enforce_inline: bool = False) -> None:
        """
        Raise ``ContentLimitError`` when ``body`` exceeds configured caps.

        Parameters
        ----------
        enforce_inline:
            When True, also reject bodies larger than ``inline_max_chars``
            (useful when the caller insists on keeping content inside State).
        """

        char_len = len(body)
        if char_len > self.max_chars:
            raise ContentLimitError(
                f"content exceeds max_chars={self.max_chars} (got {char_len})"
            )
        byte_len = len(body.encode("utf-8"))
        if byte_len > self.max_bytes:
            raise ContentLimitError(
                f"content exceeds max_bytes={self.max_bytes} (got {byte_len})"
            )
        if enforce_inline and char_len > self.inline_max_chars:
            raise ContentLimitError(
                f"content exceeds inline_max_chars={self.inline_max_chars} "
                f"(got {char_len}); store as ContentRef instead"
            )

    def should_externalize(self, body: str) -> bool:
        """Return whether production code should prefer a ``ContentRef``."""

        return len(body) > self.inline_max_chars


DEFAULT_CONTENT_LIMITS = ContentLimits()
