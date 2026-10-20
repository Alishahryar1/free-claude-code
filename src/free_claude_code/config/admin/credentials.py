"""Version-bound masked credential edits shared by both provider editors."""

import hashlib
import hmac
import json
from dataclasses import dataclass, field

from pydantic import SecretStr

from free_claude_code.config.api_keys import ApiKeyList, api_key_values
from free_claude_code.config.errors import ConfigConflictError
from free_claude_code.core.json_types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class CredentialSubmission:
    key: str
    credentials: ApiKeyList = field(repr=False)
    provider_id: str | None = None


@dataclass(frozen=True, slots=True)
class CredentialRowError:
    key: str
    index: int
    message: str
    provider_id: str | None = None

    def public(self) -> JsonObject:
        result: JsonObject = {
            "key": self.key,
            "index": self.index,
            "message": self.message,
        }
        if self.provider_id is not None:
            result["provider_id"] = self.provider_id
        return result


class CredentialEditError(ValueError):
    def __init__(self, errors: list[CredentialRowError]) -> None:
        super().__init__("Check the highlighted API key rows.")
        self.row_errors = tuple(errors)


def credential_revision(
    owner: str, keys: ApiKeyList, source: str, signing_key: bytes
) -> str:
    payload = json.dumps(
        ["fcc-api-keys-v1", owner, source, api_key_values(keys)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(signing_key, payload, hashlib.sha256).hexdigest()


def credential_projection(
    owner: str, keys: ApiKeyList, source: str, signing_key: bytes
) -> JsonObject:
    return {
        "revision": credential_revision(owner, keys, source, signing_key),
        "entries": [{"reference": index} for index in range(len(keys))],
    }


def resolve_credential_edit(
    owner: str,
    keys: ApiKeyList,
    source: str,
    submitted: JsonValue,
    signing_key: bytes,
    *,
    key: str,
    provider_id: str | None = None,
    new_owner: bool = False,
) -> ApiKeyList:
    if not isinstance(submitted, dict) or set(submitted) != {"revision", "entries"}:
        raise ValueError(f"{key}: submit an ordered key list with its revision")
    entries, revision = submitted["entries"], submitted["revision"]
    if not isinstance(entries, list):
        raise ValueError(f"{key}: entries must be an array")
    if new_owner:
        if revision is not None:
            raise ValueError(f"{key}: a new provider has no saved credential revision")
    elif not isinstance(revision, str) or not hmac.compare_digest(
        revision.encode("utf-8"),
        credential_revision(owner, keys, source, signing_key).encode("ascii"),
    ):
        raise ConfigConflictError(
            "The API key list changed. Reopen the modal before saving again."
        )
    if source == "process" and entries != [
        {"retain": index} for index in range(len(keys))
    ]:
        raise ValueError(f"{key}: API keys are managed by the process environment")

    errors: dict[int, CredentialRowError] = {}
    values: list[str] = []
    seen: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        message = None
        text = ""
        if isinstance(entry, dict) and set(entry) == {"retain"}:
            reference = entry["retain"]
            if (
                new_owner
                or type(reference) is not int
                or not 0 <= reference < len(keys)
            ):
                message = "This saved key reference is invalid."
            else:
                text = keys[reference].get_secret_value()
        elif isinstance(entry, dict) and set(entry) == {"value"}:
            value = entry["value"]
            if not isinstance(value, str):
                message = "Enter an API key as text."
            elif not (text := value.strip()) or text == "********":
                message = "Enter a key or remove this row."
        else:
            message = "A row must retain a saved key or supply a new value."
        if message:
            errors[index] = CredentialRowError(key, index, message, provider_id)
        else:
            seen.setdefault(text, []).append(index)
        values.append(text)
    for positions in seen.values():
        if len(positions) > 1:
            for index in positions:
                errors[index] = CredentialRowError(
                    key,
                    index,
                    "Duplicate API key. Remove one of these rows.",
                    provider_id,
                )
    if errors:
        raise CredentialEditError([errors[index] for index in sorted(errors)])
    return tuple(SecretStr(value) for value in values)
