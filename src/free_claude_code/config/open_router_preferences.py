"""Validated OpenRouter provider-routing preferences for request bodies."""

import json
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

DataCollectionPolicy = Literal["allow", "deny"]
SortStrategy = Literal["price", "throughput", "latency"]
SortPartition = Literal["model", "none"]
Quantization = Literal["fp4", "fp6", "fp8", "fp16", "bf16", "int4", "int8", "unknown"]
ProviderSlug = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ProviderSort(BaseModel):
    """Sort a model's endpoints by an attribute, optionally across models."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    by: SortStrategy
    partition: SortPartition = "model"


class PercentileThreshold(BaseModel):
    """A performance threshold evaluated at selected percentile cutoffs."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    p50: float | None = Field(default=None, gt=0)
    p75: float | None = Field(default=None, gt=0)
    p90: float | None = Field(default=None, gt=0)
    p99: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def at_least_one_cutoff(self) -> PercentileThreshold:
        if all(value is None for value in (self.p50, self.p75, self.p90, self.p99)):
            raise ValueError("Set at least one percentile cutoff")
        return self


class MaxPrice(BaseModel):
    """Maximum pricing before OpenRouter refuses to route the request."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    prompt: float | None = Field(default=None, ge=0)
    completion: float | None = Field(default=None, ge=0)
    request: float | None = Field(default=None, ge=0)
    image: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def at_least_one_limit(self) -> MaxPrice:
        if all(
            value is None
            for value in (self.prompt, self.completion, self.request, self.image)
        ):
            raise ValueError("Set at least one price limit")
        return self


class OpenRouterProviderPreferences(BaseModel):
    """OpenRouter's documented ``provider`` request object.

    One JSON object applied as the ``provider`` field on every OpenRouter
    chat request, overriding account defaults and model-slug routing
    variants such as ``:floor``.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    order: tuple[ProviderSlug, ...] | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: DataCollectionPolicy | None = None
    zdr: bool | None = None
    enforce_distillable_text: bool | None = None
    only: tuple[ProviderSlug, ...] | None = None
    ignore: tuple[ProviderSlug, ...] | None = None
    quantizations: tuple[Quantization, ...] | None = None
    sort: SortStrategy | ProviderSort | None = None
    preferred_min_throughput: float | PercentileThreshold | None = None
    preferred_max_latency: float | PercentileThreshold | None = None
    max_price: MaxPrice | None = None

    @field_validator("order", "only", "ignore")
    @classmethod
    def at_least_one_provider(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is not None and not value:
            raise ValueError("List at least one provider slug")
        return value

    @model_validator(mode="after")
    def at_least_one_preference(self) -> OpenRouterProviderPreferences:
        if not self.to_request_field():
            raise ValueError("Set at least one provider preference")
        return self

    def to_request_field(self) -> dict[str, Any]:
        """Return the JSON object sent as the request's ``provider`` field."""

        return self.model_dump(mode="json", exclude_none=True)


def decode_open_router_provider_preferences(value: object) -> object:
    """Decode the env JSON string, treating blank values as unset."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            raise ValueError(
                "OPENROUTER_PROVIDER_PREFERENCES must be a JSON object."
            ) from None
    return value


OpenRouterProviderPreferencesField = Annotated[
    OpenRouterProviderPreferences | None,
    BeforeValidator(decode_open_router_provider_preferences),
]


def decode_quantizations(value: object) -> object:
    """Decode a comma-separated quantization list; blank or empty means any."""

    if isinstance(value, str):
        items = (part.strip() for part in value.split(","))
        return tuple(item for item in items if item) or None
    if isinstance(value, list | tuple):
        return tuple(value) or None
    return value


OpenRouterQuantizationsField = Annotated[
    tuple[Quantization, ...] | None, BeforeValidator(decode_quantizations)
]

QUANTIZATION_VALUES: tuple[str, ...] = get_args(Quantization)
