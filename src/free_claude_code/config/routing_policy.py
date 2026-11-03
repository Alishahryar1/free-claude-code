"""Operator-declared policy candidates; configuration performs no provider I/O."""

import re
from datetime import date
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from .provider_catalog import PROVIDER_CATALOG

MODEL_ID = re.compile(r"[A-Za-z0-9_./:@+-]{1,200}")
LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio", "llamacpp"})


class RoutingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: str
    context_tokens: int = Field(gt=0)
    supports_tools: bool | None = None
    supports_thinking: bool | None = None
    supports_images: bool | None = None
    supports_documents: bool | None = None
    input_cost_per_million: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    output_cost_per_million: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    pricing_source: str | None = Field(default=None, max_length=200)
    pricing_date: date | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        provider, separator, model = value.partition("/")
        if (
            not MODEL_ID.fullmatch(value)
            or not separator
            or not model
            or provider not in PROVIDER_CATALOG
        ):
            raise ValueError("Candidate must name a supported built-in provider/model")
        return value

    @model_validator(mode="after")
    def validate_prices(self) -> Self:
        priced = (
            self.input_cost_per_million is not None
            or self.output_cost_per_million is not None
        )
        if priced and (
            self.input_cost_per_million is None
            or self.output_cost_per_million is None
            or not self.pricing_source
            or not self.pricing_date
        ):
            raise ValueError(
                "Prices require both rates, pricing_source and pricing_date"
            )
        return self


def parse_candidates(raw: str) -> tuple[RoutingCandidate, ...]:
    candidates = TypeAdapter(list[RoutingCandidate]).validate_json(raw)
    if len(candidates) > 32:
        raise ValueError("At most 32 routing candidates are supported")
    if len({candidate.model for candidate in candidates}) != len(candidates):
        raise ValueError("Routing candidate model identifiers must be unique")
    return tuple(candidates)
