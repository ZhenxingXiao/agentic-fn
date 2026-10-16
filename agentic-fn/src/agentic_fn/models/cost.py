"""Cost estimation helpers for budget accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Price:
    """Per-million-token prices (account currency units)."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


@dataclass(slots=True)
class TableCostModel:
    """
    Simple lookup cost model: (provider, physical_model) -> Price.

    Unknown models return None (Usage.cost stays unset unless filled by adapter).
    """

    prices: Mapping[tuple[str, str], Price] = field(default_factory=dict)
    default: Price | None = None

    def estimate(
        self,
        *,
        provider: str,
        physical_model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        """Estimate cost from token counts."""

        price = self.prices.get((provider, physical_model), self.default)
        if price is None:
            return None
        inp = float(input_tokens or 0)
        out = float(output_tokens or 0)
        return (inp * price.input_per_mtok + out * price.output_per_mtok) / 1_000_000.0


# Illustrative defaults (not contractual billing figures).
DEFAULT_COST_MODEL = TableCostModel(
    prices={
        ("openai", "gpt-4.1"): Price(input_per_mtok=2.0, output_per_mtok=8.0),
        ("openai", "gpt-4o-mini"): Price(input_per_mtok=0.15, output_per_mtok=0.6),
        ("dashscope_compat", "qwen-max"): Price(input_per_mtok=1.0, output_per_mtok=4.0),
        ("dashscope_compat", "qwen-plus"): Price(input_per_mtok=0.4, output_per_mtok=1.2),
        ("fake", "fake-model"): Price(input_per_mtok=0.0, output_per_mtok=0.0),
    }
)
