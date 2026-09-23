"""Provider model-list response parsing helpers."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeIs

from free_claude_code.application.model_metadata import (
    ProviderModelInfo as _ProviderModelInfo,
)
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability

type ModelListScalar = str | bool
type RequiredPathValues = tuple[
    tuple[tuple[str, ...], tuple[ModelListScalar, ...]], ...
]
type InputModalityBooleanPaths = tuple[tuple[ModelInputModality, tuple[str, ...]], ...]
type ModelTokenLimitResolver = Callable[[object], int | None]


class ModelListResponseError(ValueError):
    """A provider model-list response cannot be parsed safely."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def model_infos_from_ids(
    model_ids: Iterable[str], *, supports_thinking: bool | None = None
) -> frozenset[_ProviderModelInfo]:
    """Build unknown-capability model metadata from plain provider model ids."""
    return frozenset(
        _ProviderModelInfo(model_id=model_id, supports_thinking=supports_thinking)
        for model_id in model_ids
        if model_id.strip()
    )


def extract_openai_model_infos(
    payload: Any,
    *,
    provider_name: str,
    collection_field: str | None = "data",
    id_field: str = "id",
    aliases_field: str | None = None,
    required_path_values: RequiredPathValues = (),
    required_null_field: str | None = None,
    required_sequence_items: tuple[tuple[str, str], ...] = (),
    exclude_missing_sequence_fields: bool = False,
    tags_field: str | None = None,
    thinking_tag: str = "reasoning",
    non_thinking_tag: str | None = None,
    thinking_boolean_path: tuple[str, ...] | None = None,
    input_modalities_path: tuple[str, ...] | None = None,
    thinking_sequence_path: tuple[str, ...] | None = None,
    fixed_input_modalities: frozenset[ModelInputModality] | None = None,
    input_modality_boolean_paths: InputModalityBooleanPaths = (),
    context_window_tokens_path: tuple[str, ...] | None = None,
    max_output_tokens_path: tuple[str, ...] | None = None,
    context_window_tokens_resolver: ModelTokenLimitResolver | None = None,
) -> frozenset[_ProviderModelInfo]:
    """Extract routable IDs from an OpenAI-compatible model-list response."""
    model_infos: dict[str, _ProviderModelInfo] = {}
    item_location = collection_field or "root-array"
    for item in model_list_items(
        payload,
        provider_name=provider_name,
        collection_field=collection_field,
    ):
        model_id = _field(item, id_field)
        if not isinstance(model_id, str) or not model_id.strip():
            raise _malformed(
                provider_name,
                f"expected every {item_location} item to include {id_field}",
            )
        included = True
        for path, allowed_values in required_path_values:
            path_value = _path(item, path)
            matching_types = tuple(
                allowed
                for allowed in allowed_values
                if type(path_value) is type(allowed)
            )
            if path_value is _MISSING or not matching_types:
                expected_types = "/".join(
                    dict.fromkeys(_scalar_type_name(value) for value in allowed_values)
                )
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{'.'.join(path)} as {expected_types}",
                )
            if path_value not in matching_types:
                included = False

        if required_null_field is not None:
            if not _has_field(item, required_null_field):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{required_null_field}",
                )
            if _field(item, required_null_field) is not None:
                included = False

        missing_sequence_field = False
        for field_name, required_item in required_sequence_items:
            values = _field(item, field_name)
            if values is None and exclude_missing_sequence_fields:
                missing_sequence_field = True
                continue
            if not _is_sequence(values) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{field_name} string array",
                )
            if required_item not in values:
                included = False

        if missing_sequence_field:
            continue

        supports_thinking: bool | None = None
        intrinsic_facts: set[bool] = set()
        if tags_field is not None:
            tags_value = _field(item, tags_field)
            tags = _optional_string_sequence(tags_value)
            if tags is not None:
                tag_set = frozenset(tags)
                if thinking_tag in tag_set:
                    intrinsic_facts.add(True)
                if non_thinking_tag is not None and non_thinking_tag in tag_set:
                    intrinsic_facts.add(False)
                if thinking_tag in tag_set:
                    supports_thinking = True
                elif non_thinking_tag is not None and non_thinking_tag in tag_set:
                    supports_thinking = False

        if thinking_boolean_path is not None:
            capability = _path(item, thinking_boolean_path)
            if isinstance(capability, bool):
                supports_thinking = capability
                intrinsic_facts.add(capability)

        if thinking_sequence_path is not None:
            values = _optional_string_sequence(_path(item, thinking_sequence_path))
            if values is not None:
                supports_thinking = thinking_tag in values

        input_modalities = _input_modalities(
            item,
            sequence_path=input_modalities_path,
            fixed=fixed_input_modalities,
            boolean_paths=input_modality_boolean_paths,
        )
        context_window_tokens = (
            context_window_tokens_resolver(item)
            if context_window_tokens_resolver is not None
            else _optional_positive_int_at_path(item, context_window_tokens_path)
        )
        max_output_tokens = _optional_positive_int_at_path(item, max_output_tokens_path)

        if not included:
            continue

        model_info = _ProviderModelInfo(
            model_id=model_id,
            reasoning_capability=ReasoningCapability.NONE
            if intrinsic_facts == {False}
            else ReasoningCapability.UNKNOWN,
            supports_thinking=supports_thinking,
            input_modalities=input_modalities,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
        )
        model_infos.setdefault(model_id, model_info)
        if aliases_field is not None:
            aliases = _field(item, aliases_field)
            if not _is_sequence(aliases):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{aliases_field} array",
                )
            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    raise _malformed(
                        provider_name,
                        f"expected every {aliases_field} item to be a model id",
                    )
                model_infos.setdefault(
                    alias,
                    replace(model_info, model_id=alias),
                )

    if not model_infos:
        raise _malformed(provider_name, "response did not include any model ids")
    return frozenset(model_infos.values())


def extract_tool_capable_model_infos(
    payload: Any, *, provider_name: str
) -> frozenset[_ProviderModelInfo]:
    """Extract tool-capable models with ``supported_parameters`` metadata."""
    data = model_list_items(payload, provider_name=provider_name)

    model_infos: set[_ProviderModelInfo] = set()
    for item in data:
        model_id = _field(item, "id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise _malformed(provider_name, "expected every data item to include id")

        supported_parameters = _field(item, "supported_parameters")
        if not _is_sequence(supported_parameters):
            continue
        supported_parameter_names = {
            param for param in supported_parameters if isinstance(param, str)
        }
        if supported_parameter_names.isdisjoint({"tools", "tool_choice"}):
            continue
        capability_parameters = _optional_string_sequence(supported_parameters)
        model_infos.add(
            _ProviderModelInfo(
                model_id=model_id,
                reasoning_capability=reasoning_capability_from_options(
                    _field(item, "reasoning")
                ),
                supports_thinking=(
                    "reasoning" in capability_parameters
                    if capability_parameters is not None
                    else None
                ),
                input_modalities=optional_input_modalities(
                    _path(item, ("architecture", "input_modalities"))
                ),
                context_window_tokens=optional_positive_int(
                    _path(item, ("context_length",))
                ),
                max_output_tokens=optional_positive_int(
                    _path(item, ("top_provider", "max_completion_tokens"))
                ),
            )
        )

    return frozenset(model_infos)


def reasoning_capability_from_options(options: object) -> ReasoningCapability:
    """Retain explicit gateway computation facts; missing controls prove nothing."""
    if not isinstance(options, Mapping):
        return ReasoningCapability.UNKNOWN
    mandatory = options.get("mandatory")
    efforts = options.get("supported_efforts")
    if mandatory is not None and not isinstance(mandatory, bool):
        return ReasoningCapability.UNKNOWN
    if efforts is not None and _optional_string_sequence(efforts) is None:
        return ReasoningCapability.UNKNOWN
    can_disable = isinstance(efforts, (list, tuple)) and "none" in efforts
    if mandatory is True:
        return (
            ReasoningCapability.UNKNOWN if can_disable else ReasoningCapability.REQUIRED
        )
    return ReasoningCapability.OPTIONAL if can_disable else ReasoningCapability.UNKNOWN


def model_list_items(
    payload: Any,
    *,
    provider_name: str,
    collection_field: str | None = "data",
) -> tuple[Any, ...]:
    """Return a validated OpenAI-shaped model-list data array."""
    data = payload if collection_field is None else _field(payload, collection_field)
    if not _is_sequence(data):
        location = (
            "root array"
            if collection_field is None
            else (f"top-level {collection_field} array")
        )
        raise _malformed(
            provider_name,
            f"expected {location}",
        )
    return tuple(data)


def validate_model_list_page(
    payload: Any,
    *,
    provider_name: str,
    expected_page: int,
    current_page_path: tuple[str, ...],
    total_pages_path: tuple[str, ...],
    max_pages: int,
    expected_total_pages: int | None = None,
) -> int:
    """Validate numbered pagination metadata and return the total page count."""
    current_page = _path(payload, current_page_path)
    if type(current_page) is not int:
        raise _malformed(
            provider_name,
            f"expected {'.'.join(current_page_path)} to be an integer",
        )
    if current_page != expected_page:
        raise _malformed(
            provider_name,
            f"expected {'.'.join(current_page_path)} to be {expected_page}",
        )

    total_pages = _path(payload, total_pages_path)
    if type(total_pages) is not int:
        raise _malformed(
            provider_name,
            f"expected {'.'.join(total_pages_path)} to be an integer",
        )
    if total_pages < 1 or total_pages > max_pages:
        raise _malformed(
            provider_name,
            f"expected {'.'.join(total_pages_path)} between 1 and {max_pages}",
        )
    if expected_total_pages is not None and total_pages != expected_total_pages:
        raise _malformed(
            provider_name,
            f"expected {'.'.join(total_pages_path)} to remain {expected_total_pages}",
        )
    return total_pages


def merge_model_list_pages(
    payloads: Iterable[Any],
    *,
    provider_name: str,
    collection_field: str | None,
) -> tuple[Any, ...] | dict[str, tuple[Any, ...]]:
    """Combine complete model-list pages before strict record parsing."""
    merged: list[Any] = []
    for payload in payloads:
        merged.extend(
            model_list_items(
                payload,
                provider_name=provider_name,
                collection_field=collection_field,
            )
        )

    items = tuple(merged)
    if collection_field is None:
        return items
    return {collection_field: items}


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _has_field(item: Any, name: str) -> bool:
    if isinstance(item, Mapping):
        return name in item
    return hasattr(item, name)


_MISSING = object()


def _path(item: Any, path: tuple[str, ...]) -> Any:
    current = item
    for name in path:
        if not _has_field(current, name):
            return _MISSING
        current = _field(current, name)
    return current


def _is_sequence(value: object) -> TypeIs[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _optional_string_sequence(value: object) -> tuple[str, ...] | None:
    if value is _MISSING or value is None or not _is_sequence(value):
        return None
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        strings.append(item)
    return tuple(strings)


_INPUT_MODALITY_BY_VALUE = {modality.value: modality for modality in ModelInputModality}


def optional_input_modalities(
    value: object,
) -> frozenset[ModelInputModality] | None:
    """Normalize a provider's optional input-modality string sequence."""
    values = _optional_string_sequence(value)
    if values is None:
        return None
    modalities = frozenset(
        _INPUT_MODALITY_BY_VALUE[item]
        for item in values
        if item in _INPUT_MODALITY_BY_VALUE
    )
    if ModelInputModality.TEXT not in modalities:
        return None
    return modalities


def optional_positive_int(value: object) -> int | None:
    """Return an exact positive integer, otherwise unknown."""
    return value if type(value) is int and value > 0 else None


def live_provider_context_window_consensus(item: object) -> int | None:
    """Return the common context limit across every advertised live route."""
    routes = _path(item, ("providers",))
    if not _is_sequence(routes):
        return None
    live_limits: list[int] = []
    for route in routes:
        if _path(route, ("status",)) != "live":
            continue
        limit = optional_positive_int(_path(route, ("context_length",)))
        if limit is None:
            return None
        live_limits.append(limit)
    if not live_limits or len(set(live_limits)) != 1:
        return None
    return live_limits[0]


def _optional_positive_int_at_path(
    item: object, path: tuple[str, ...] | None
) -> int | None:
    if path is None:
        return None
    return optional_positive_int(_path(item, path))


def _input_modalities(
    item: object,
    *,
    sequence_path: tuple[str, ...] | None,
    fixed: frozenset[ModelInputModality] | None,
    boolean_paths: InputModalityBooleanPaths,
) -> frozenset[ModelInputModality] | None:
    if sequence_path is not None:
        return optional_input_modalities(_path(item, sequence_path))
    if not boolean_paths:
        return fixed

    modalities = set(fixed or ())
    for modality, path in boolean_paths:
        enabled = _path(item, path)
        if not isinstance(enabled, bool):
            return None
        if enabled:
            modalities.add(modality)
        else:
            modalities.discard(modality)
    return frozenset(modalities)


def _scalar_type_name(value: ModelListScalar) -> str:
    return type(value).__name__


def _malformed(provider_name: str, reason: str) -> ModelListResponseError:
    return ModelListResponseError(
        f"{provider_name} model-list response is malformed: {reason}"
    )
