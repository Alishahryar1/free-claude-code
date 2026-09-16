"""Bounded request waits on initialization owned by a longer-lived service."""

import asyncio

from .errors import ApplicationUnavailableError

STARTUP_WAIT_SECONDS = 30.0


class InitializationWait:
    """One request's remaining time spent waiting, excluding normal execution."""

    def __init__(self, seconds: float = STARTUP_WAIT_SECONDS) -> None:
        self.remaining = seconds

    async def wait[T](self, task: asyncio.Task[T]) -> T:
        if task.done():
            if task.cancelled():
                raise ApplicationUnavailableError(
                    "FCC startup was interrupted. Retry shortly."
                )
            return task.result()
        started = asyncio.get_running_loop().time()
        caller = asyncio.current_task()
        cancellations = caller.cancelling() if caller else 0
        try:
            async with asyncio.timeout(self.remaining):
                return await asyncio.shield(task)
        except TimeoutError:
            raise ApplicationUnavailableError(
                "FCC is still starting. Try again shortly."
            ) from None
        except asyncio.CancelledError:
            if task.cancelled() and caller and caller.cancelling() == cancellations:
                raise ApplicationUnavailableError(
                    "FCC startup was interrupted. Retry shortly."
                ) from None
            raise
        finally:
            self.remaining = max(
                0.0, self.remaining - (asyncio.get_running_loop().time() - started)
            )
