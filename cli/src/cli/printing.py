"""Shared pretty-print helpers for demo output."""

from __future__ import annotations

from typing import Any

from agentic_fn import PipelineResult


def banner(title: str) -> None:
    """Print a section header so demos are easy to scan in the terminal."""

    line = "=" * 60
    print(f"\n{line}")
    print(f" {title}")
    print(f"{line}")


def note(message: str) -> None:
    """Print a short explanatory note (what this demo is teaching)."""

    print(f"  note: {message}")


def kv(key: str, value: Any) -> None:
    """Print a key/value pair with stable indentation."""

    print(f"  {key}: {value}")


def print_result(result: PipelineResult) -> None:
    """
    Dump the outcome of a pipeline run.

    Shows status, final business state, soft-error bucket (if any),
    metrics, and a compact step trace - enough for manual inspection.
    """

    kv("status", result.status.name)
    kv("ok", result.ok)
    kv("state.data", dict(result.state.data))
    if result.error is not None:
        kv("error", f"{type(result.error).__name__}: {result.error}")
    kv(
        "metrics",
        (
            f"duration_ms={result.metrics.duration_ms:.2f}, "
            f"steps={result.metrics.step_count}, "
            f"retries={result.metrics.retry_count}, "
            f"effects={result.metrics.effect_count}"
        ),
    )
    if result.trace:
        print("  trace:")
        for record in result.trace:
            mark = "ok" if record.success else "FAIL"
            print(
                f"    - [{mark}] {record.name} "
                f"(attempt={record.attempt}, {record.duration_ms:.2f}ms"
                + (f", err={record.error_type}" if record.error_type else "")
                + ")"
            )
