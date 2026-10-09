"""Content-domain exception hierarchy."""

from __future__ import annotations

from agentic_fn.core.errors import ConfigurationError, PermanentError


class ContentError(PermanentError):
    """Base class for content model / conversion failures."""


class ContentValidationError(ContentError):
    """Raised when content fails structural or kind-specific validation."""


class ContentKindError(ContentValidationError):
    """Raised when an operation is incompatible with the content kind."""


class ContentLimitError(ContentValidationError):
    """Raised when content exceeds configured size or nesting limits."""


class ContentCodecError(ContentValidationError):
    """Raised when serialization / deserialization of content fails."""


class ContentConversionError(ContentError):
    """Raised when a kind conversion is impossible or refused by policy."""


class ContentStoreError(ContentError):
    """Raised when a content reference cannot be resolved or persisted."""


class ContentConfigurationError(ConfigurationError):
    """Raised when content subsystem configuration is invalid."""
