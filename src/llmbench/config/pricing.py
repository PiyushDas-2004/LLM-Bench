"""User-editable pricing table used for cost accounting and estimates.

Prices change frequently, so LLMBench never hardcodes them. The table is a YAML
file with an ``as_of`` date; every report that shows money also shows that date.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelPrice(BaseModel):
    """Price for one model in USD per one million tokens."""

    model_config = ConfigDict(extra="forbid")

    input_per_1m: float = Field(ge=0)
    output_per_1m: float = Field(ge=0)


class PricingTable(BaseModel):
    """Mapping of ``provider/model`` keys to prices.

    Lookups try the exact ``provider/model`` key, then ``provider/*`` as a
    provider-wide default. Unknown models cost ``None`` (reported as unknown,
    never silently as zero).
    """

    model_config = ConfigDict(extra="forbid")

    as_of: dt.date
    currency: str = "USD"
    note: str = "Prices must be verified by the user against the provider's pricing page."
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PricingTable:
        """Load a pricing table from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def empty(cls) -> PricingTable:
        """A table with no prices; every non-mock model reports unknown cost."""
        return cls(as_of=dt.date(1970, 1, 1), models={})

    def price_for(self, provider: str, model: str) -> ModelPrice | None:
        """Return the price entry for a model, or ``None`` if not listed."""
        return self.models.get(f"{provider}/{model}") or self.models.get(f"{provider}/*")

    def cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float | None:
        """Cost in USD of one call, or ``None`` if the model has no price entry."""
        price = self.price_for(provider, model)
        if price is None:
            return None
        return (input_tokens * price.input_per_1m + output_tokens * price.output_per_1m) / 1e6
