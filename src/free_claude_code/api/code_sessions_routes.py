"""Local Admin commands, snapshots and SSE for coding conversations."""

import base64
import json
from collections.abc import AsyncIterator, Mapping
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from free_claude_code.application.code_sessions import (
    CodeApplicationPort,
    CodeDetail,
    CodeItem,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
    CodeValidationError,
)
from free_claude_code.application.code_sessions.models import CodeMode
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.session_events import EventOverflowError
from free_claude_code.config.model_refs import split_provider_model_ref
from free_claude_code.core.json_types import JsonObject, JsonValue

from .admin_routes import admin_page_response
from .admin_security import require_loopback_admin
from .dependencies import get_services
from .markdown import render_markdown
from .ports import ApiServices

router = APIRouter(dependencies=[Depends(require_loopback_admin)])


class CommandPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreatePayload(CommandPayload):
    session_id: str
    cwd: str = Field(min_length=1, max_length=4096)
    harness: Literal["codex"] = "codex"


class FolderPickerPayload(CommandPayload):
    initial_path: str | None = Field(default=None, max_length=4096)


class SettingsPayload(CommandPayload):
    expected_revision: int = Field(gt=0)
    title: str | None = Field(default=None, min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1)
    reasoning_effort: str | None = None
    mode: CodeMode = "config"


class SendPayload(CommandPayload):
    operation_id: str
    expected_revision: int = Field(gt=0)
    expected_epoch: str
    text: str = Field(min_length=1, max_length=1_000_000)


class StopPayload(CommandPayload):
    operation_id: str


class AnswerPayload(CommandPayload):
    response_id: str
    answer: JsonObject


def _code(services: ApiServices) -> CodeApplicationPort:
    if services.code is None:
        raise CodeUnavailableError("Code sessions is unavailable in this FCC runtime.")
    return services.code


@router.get("/admin/code", include_in_schema=False)
@router.get("/admin/code/{session_id}", include_in_schema=False)
def code_page(request: Request, session_id: str | None = None):
    return admin_page_response()


@router.get("/admin/api/code/bootstrap")
def bootstrap(services: ApiServices = Depends(get_services)) -> JsonObject:
    code = _code(services)
    available, message = code.availability()
    catalog = code.catalog()
    return {
        "available": available,
        "message": message,
        "harnesses": [{"id": "codex", "name": "Codex"}],
        "epoch": code.epoch,
        "models": [model.model_dump(mode="json") for model in catalog.models],
        "default_model": catalog.default_model,
    }


@router.post("/admin/api/code/folder-picker")
async def pick_folder(
    payload: FolderPickerPayload, services: ApiServices = Depends(get_services)
) -> JsonObject:
    try:
        return {"path": await services.admin.pick_folder(payload.initial_path)}
    except ApplicationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc


@router.get("/admin/api/code/events", response_class=EventSourceResponse)
async def events(
    services: ApiServices = Depends(get_services),
) -> AsyncIterator[ServerSentEvent]:
    subscription, ready = await _code(services).subscribe()
    try:
        summaries = ready.get("sessions")
        if isinstance(summaries, list):
            ready = {
                **ready,
                "sessions": [
                    _event_payload(value)
                    for value in summaries
                    if isinstance(value, Mapping)
                ],
            }
        yield ServerSentEvent(
            event="feed.ready", id=str(subscription.cursor), retry=1000, data=ready
        )
        try:
            async for event in subscription:
                yield ServerSentEvent(
                    event=event.event,
                    id=str(event.id),
                    data={**_event_payload(event.data), "cursor": event.id},
                )
        except EventOverflowError as exc:
            yield ServerSentEvent(
                event="feed.resync_required",
                id=str(exc.cursor),
                data={"cursor": exc.cursor},
            )
    finally:
        await subscription.aclose()


@router.get("/admin/api/code/sessions")
async def list_sessions(
    cursor: str | None = None,
    query: str = Query(default="", max_length=4096),
    limit: int = Query(default=25, ge=1, le=25),
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    code = _code(services)
    snapshot_cursor = code.cursor
    page = await code.list_sessions(_decode_cursor(cursor), limit, query)
    return {
        "sessions": [_session_payload(session) for session in page.sessions],
        "next_cursor": _encode_cursor(page.next_cursor),
        "epoch": code.epoch,
        "cursor": snapshot_cursor,
    }


@router.post("/admin/api/code/sessions", status_code=201)
async def create(
    payload: CreatePayload, services: ApiServices = Depends(get_services)
) -> JsonObject:
    return _session_payload(
        await _code(services).create_session(payload.session_id, payload.cwd)
    )


@router.get("/admin/api/code/sessions/{session_id}")
async def detail(
    session_id: str,
    include_item_ids: tuple[str, ...] = Query(default=()),
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    return _detail_payload(
        await _code(services).get_detail(session_id, include_item_ids=include_item_ids)
    )


@router.get("/admin/api/code/sessions/{session_id}/items")
async def older_items(
    session_id: str,
    before: str,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    return _detail_payload(
        await _code(services).get_detail(session_id, before=_decode_item_cursor(before))
    )


@router.patch("/admin/api/code/sessions/{session_id}")
async def update_settings(
    session_id: str,
    payload: SettingsPayload,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    return _session_payload(
        await _code(services).update_settings(
            session_id,
            payload.expected_revision,
            payload.model_dump(exclude={"expected_revision"}, exclude_unset=True),
        )
    )


@router.delete("/admin/api/code/sessions/{session_id}", status_code=202)
async def delete(
    session_id: str,
    expected_revision: int = Query(gt=0),
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    session = await _code(services).delete_session(session_id, expected_revision)
    return {
        "session_id": session_id,
        "deleted": session is None,
        "session": _session_payload(session) if session else None,
    }


@router.post("/admin/api/code/sessions/{session_id}/turns", status_code=202)
async def send(
    session_id: str, payload: SendPayload, services: ApiServices = Depends(get_services)
) -> JsonObject:
    return _run_payload(
        await _code(services).send(
            session_id,
            payload.operation_id,
            payload.expected_revision,
            payload.text,
            expected_epoch=payload.expected_epoch,
        )
    )


@router.post("/admin/api/code/sessions/{session_id}/stop", status_code=202)
async def stop(
    session_id: str, payload: StopPayload, services: ApiServices = Depends(get_services)
) -> JsonObject:
    return _run_payload(await _code(services).stop(session_id, payload.operation_id))


@router.post(
    "/admin/api/code/sessions/{session_id}/prompts/{prompt_id}/responses",
    status_code=202,
)
async def answer(
    session_id: str,
    prompt_id: str,
    payload: AnswerPayload,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    return _prompt_payload(
        await _code(services).answer(
            session_id, prompt_id, payload.response_id, payload.answer
        )
    )


def _session_payload(session: CodeSession) -> JsonObject:
    provider_id, model_name = split_provider_model_ref(session.model)
    return {
        **session.model_dump(
            mode="json",
            exclude={
                "native_thread_id",
                "native_may_have_input",
                "native_permission_defaults",
                "auto_title",
            },
        ),
        "provider_id": provider_id,
        "model_name": model_name,
    }


def _run_payload(run: CodeRun) -> JsonObject:
    return run.model_dump(mode="json", exclude={"native_turn_id", "submission_started"})


def _item_payload(item: CodeItem) -> JsonObject:
    return {
        **item.model_dump(
            mode="json", exclude={"raw", "native_turn_id", "native_item_id"}
        ),
        "html": render_markdown(item.text)
        if item.kind in {"text", "reasoning"}
        else None,
    }


def _prompt_payload(prompt: CodePrompt) -> JsonObject:
    return prompt.model_dump(
        mode="json",
        exclude={"raw", "generation", "request_id", "native_turn_id", "native_item_id"},
    )


def _detail_payload(detail: CodeDetail) -> JsonObject:
    return {
        "session": _session_payload(detail.session),
        "run": _run_payload(detail.run) if detail.run else None,
        "runs": [_run_payload(run) for run in detail.runs],
        "active_prompt_ids": list(detail.active_prompt_ids),
        "active_review_ids": list(detail.active_review_ids),
        "items": [_item_payload(item) for item in detail.items],
        "prompts": [_prompt_payload(prompt) for prompt in detail.prompts],
        "epoch": detail.epoch,
        "version": detail.version,
        "cursor": detail.cursor,
        "next_before": _encode_cursor(detail.next_before),
    }


def _event_payload(data: Mapping[str, JsonValue]) -> JsonObject:
    result = dict(data)
    if isinstance(data.get("session"), Mapping):
        result["session"] = _session_payload(
            CodeSession.model_validate(data["session"])
        )
    if isinstance(data.get("run"), Mapping):
        result["run"] = _run_payload(CodeRun.model_validate(data["run"]))
    if isinstance(data.get("item"), Mapping):
        result["item"] = _item_payload(CodeItem.model_validate(data["item"]))
    if isinstance(data.get("prompt"), Mapping):
        result["prompt"] = _prompt_payload(CodePrompt.model_validate(data["prompt"]))
    items = data.get("items")
    if isinstance(items, list):
        result["items"] = [
            _item_payload(CodeItem.model_validate(value)) for value in items
        ]
    prompts = data.get("prompts")
    runs = data.get("runs")
    if isinstance(runs, list):
        result["runs"] = [_run_payload(CodeRun.model_validate(value)) for value in runs]
    if isinstance(prompts, list):
        result["prompts"] = [
            _prompt_payload(CodePrompt.model_validate(value)) for value in prompts
        ]
    return result


def _encode_cursor(cursor: tuple[int, str] | tuple[int, int] | None) -> str | None:
    return (
        base64.urlsafe_b64encode(json.dumps(cursor).encode()).decode()
        if cursor
        else None
    )


def _decode_cursor(cursor: str | None) -> tuple[int, str] | None:
    if cursor is None:
        return None
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor).decode())
        if (
            not isinstance(value, list)
            or len(value) != 2
            or type(value[0]) is not int
            or not isinstance(value[1], str)
        ):
            raise ValueError
        return value[0], value[1]
    except ValueError, UnicodeError:
        raise CodeValidationError("Invalid session page cursor.") from None


def _decode_item_cursor(cursor: str) -> tuple[int, int]:
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor).decode())
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(part) is not int or part < 1 for part in value)
        ):
            raise ValueError
        return value[0], value[1]
    except ValueError, UnicodeError:
        raise CodeValidationError("Invalid transcript page cursor.") from None
