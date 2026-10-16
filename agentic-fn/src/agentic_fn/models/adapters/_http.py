"""Minimal stdlib HTTP helper for adapters (no httpx required)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from agentic_fn.models.errors import ModelPermanentError, ModelTransientError


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """POST JSON and return parsed object; map HTTP errors to model errors."""

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - URL from config
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {408, 409, 425, 429} or exc.code >= 500:
            raise ModelTransientError(
                f"HTTP {exc.code} from {url}: {detail[:500]}",
                cause=exc,
            ) from exc
        raise ModelPermanentError(
            f"HTTP {exc.code} from {url}: {detail[:500]}",
            cause=exc,
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelTransientError(f"network error calling {url}: {exc}", cause=exc) from exc
    except TimeoutError as exc:
        raise ModelTransientError(f"timeout calling {url}", cause=exc) from exc
