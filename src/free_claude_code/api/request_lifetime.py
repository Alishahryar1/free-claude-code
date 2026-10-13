"""Client-owned lifetime boundary for long-running HTTP requests."""

import asyncio

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CLIENT_OWNED_PATHS = frozenset(
    {
        "/v1/messages",
        "/v1/responses",
        "/v1/messages/count_tokens",
        "/admin/api/code/folder-picker",
    }
)


class ClientRequestLifetimeMiddleware:
    """Cancel client-owned operations when their HTTP client disconnects."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not _owns_client_lifetime(scope):
            await self._app(scope, receive, send)
            return

        messages: asyncio.Queue[Message] = asyncio.Queue(maxsize=1)

        async def receive_until_disconnect() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                await messages.put(message)

        async def receive_for_application() -> Message:
            return await messages.get()

        async def run_application() -> None:
            await self._app(scope, receive_for_application, send)

        application_task = asyncio.create_task(
            run_application(),
            name="fcc-client-application",
        )
        receive_task = asyncio.create_task(
            receive_until_disconnect(),
            name="fcc-client-disconnect-receiver",
        )
        try:
            done, _pending = await asyncio.wait(
                (application_task, receive_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if application_task in done:
                try:
                    application_task.result()
                finally:
                    await _cancel_and_wait(receive_task)
                return

            receive_outcome = _task_outcome(receive_task)
            (application_outcome,) = await _cancel_and_wait(application_task)
            if receive_outcome is not None:
                raise receive_outcome
            if application_outcome is not None and not isinstance(
                application_outcome,
                asyncio.CancelledError,
            ):
                raise application_outcome
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                _cancel_and_wait(application_task, receive_task),
                name="fcc-client-lifetime-cleanup",
            )
            await _wait_for_cleanup(cleanup_task)
            raise


def _owns_client_lifetime(scope: Scope) -> bool:
    path = scope.get("path")
    return scope["type"] == "http" and (
        (scope.get("method") == "POST" and path in _CLIENT_OWNED_PATHS)
        or (
            scope.get("method") == "GET" and path in {"/v1/models", "/muse-code/models"}
        )
    )


def _task_outcome(task: asyncio.Task[None]) -> BaseException | None:
    try:
        task.result()
    except BaseException as exc:
        return exc
    return None


async def _cancel_and_wait(
    *tasks: asyncio.Task[None],
) -> tuple[BaseException | None, ...]:
    for task in tasks:
        if not task.done():
            task.cancel()

    outcomes: list[BaseException | None] = []
    for task in tasks:
        try:
            await task
        except BaseException as exc:
            outcomes.append(exc)
        else:
            outcomes.append(None)
    return tuple(outcomes)


async def _wait_for_cleanup(
    task: asyncio.Task[tuple[BaseException | None, ...]],
) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()
