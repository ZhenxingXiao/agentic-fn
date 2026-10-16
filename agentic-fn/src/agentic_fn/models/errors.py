"""Model IO exception hierarchy."""

from __future__ import annotations

from agentic_fn.core.errors import ConfigurationError, PermanentError, TransientError


class ModelError(PermanentError):
    """Base class for model-layer failures."""


class ModelTransientError(TransientError, ModelError):
    """Retryable / failover-eligible model failure (rate limit, 5xx, network)."""


class ModelPermanentError(ModelError):
    """Non-retryable model failure (bad request, auth, unsupported)."""


class ModelCapabilityError(ModelPermanentError):
    """Provider/client does not support the requested capability."""


class ModelSafetyError(ModelPermanentError):
    """Blocked by safety / content policy filter."""


class ModelRouteError(ModelPermanentError):
    """Router could not produce a valid route decision."""


class ModelExhaustedError(ModelPermanentError):
    """All route attempts failed."""


class ModelConfigurationError(ConfigurationError):
    """Invalid route table, registry, or runtime configuration."""


class ModelReplayError(ModelPermanentError):
    """Replay mode could not satisfy a model call from the recorder."""
