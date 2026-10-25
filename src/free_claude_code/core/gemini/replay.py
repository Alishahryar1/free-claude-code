"""Stateless source-native state carried in Gemini's opaque signature field."""

import base64
import binascii
import json
import re
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import cast

from ..history_replay import (
    HistoryProtocol,
    ReplayOrigin,
    ReplayRecord,
    decode_replay,
    encode_replay,
)
from ..json_types import JsonObject
from .errors import GeminiConversionError

_PREFIX = b"fcc:gemini-replay:v1:"


@dataclass(frozen=True, slots=True)
class CallReplay:
    origin: ReplayOrigin
    call_id: str
    extra_content: JsonObject


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    response_id: str
    records: tuple[ReplayRecord, ...] = ()
    calls: tuple[CallReplay, ...] = ()
    thought_echo_sha256: str | None = None


def encode_bundle(bundle: ReplayBundle) -> str:
    payload = {
        "response_id": bundle.response_id,
        "records": [encode_replay(record) for record in bundle.records],
        "calls": [asdict(call) for call in bundle.calls],
    }
    if bundle.thought_echo_sha256 is not None:
        payload["thought_echo_sha256"] = bundle.thought_echo_sha256
    raw = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return base64.b64encode(_PREFIX + raw.encode()).decode("ascii")


def decode_bundle(signature: str) -> ReplayBundle | None:
    try:
        raw = base64.b64decode(
            signature + "=" * (-len(signature) % 4), altchars=b"-_", validate=True
        )
        if not raw.startswith(_PREFIX):
            return None
        value = json.loads(raw[len(_PREFIX) :])
        if not isinstance(value, dict) or set(value) - {"thought_echo_sha256"} != {
            "response_id",
            "records",
            "calls",
        }:
            raise ValueError("Invalid bundle fields")
        digest = value.get("thought_echo_sha256")
        if "thought_echo_sha256" in value and (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        ):
            raise ValueError("Invalid thought echo digest")
        if not isinstance(value["response_id"], str) or not value["response_id"]:
            raise ValueError("Missing response identity")
        if not isinstance(value["records"], list) or not isinstance(
            value["calls"], list
        ):
            raise ValueError("Invalid replay collections")
        if any(not isinstance(record, str) for record in value["records"]):
            raise ValueError("Invalid reasoning record")
        records = tuple(decode_replay(record) for record in value["records"])
        calls = tuple(_call_record(item) for item in value["calls"])
        if len({call.call_id for call in calls}) != len(calls):
            raise ValueError("Duplicate call metadata")
        return ReplayBundle(value["response_id"], records, calls, digest)
    except (
        ValueError,
        TypeError,
        KeyError,
        binascii.Error,
        UnicodeError,
        RecursionError,
    ) as error:
        raise GeminiConversionError(
            "Malformed Gemini thoughtSignature replay"
        ) from error


def _call_record(value: object) -> CallReplay:
    if not isinstance(value, dict) or set(value) != {
        "origin",
        "call_id",
        "extra_content",
    }:
        raise ValueError("Invalid call replay")
    origin = value["origin"]
    if (
        not isinstance(origin, dict)
        or set(origin) != {"provider", "protocol", "endpoint", "connection", "model"}
        or not all(isinstance(item, str) for item in origin.values())
        or origin["protocol"] != "chat"
        or not origin["provider"]
        or not origin["model"]
        or not isinstance(value["call_id"], str)
        or not value["call_id"]
        or not isinstance(value["extra_content"], dict)
    ):
        raise ValueError("Invalid call replay provenance")
    return CallReplay(
        ReplayOrigin(
            origin["provider"],
            cast(HistoryProtocol, origin["protocol"]),
            origin["endpoint"],
            origin["connection"],
            origin["model"],
        ),
        value["call_id"],
        deepcopy(value["extra_content"]),
    )


def apply_call_replay(
    body: JsonObject, origin: ReplayOrigin, records: Iterable[CallReplay]
) -> JsonObject:
    """Restore metadata only after the actual Chat endpoint/account is known."""
    applicable = {
        record.call_id: record for record in records if origin.accepts(record.origin)
    }
    if not applicable:
        return body
    result = deepcopy(body)
    messages = result.get("messages")
    for message in messages if isinstance(messages, list) else ():
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        for call in calls if isinstance(calls, list) else ():
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                record = applicable.get(call["id"])
                if record is not None:
                    call["extra_content"] = deepcopy(record.extra_content)
    return result
