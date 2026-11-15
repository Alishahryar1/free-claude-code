"""Request-scoped endpoint resolution borrowed from provider account owners."""

from free_claude_code.providers.endpoint_types import EndpointContext, HttpEndpoint


class RequestEndpoint:
    """Resolve each attempt's snapshot, retaining a pending forced refresh."""

    def __init__(self, context: EndpointContext) -> None:
        self._context = context
        self._refresh_pending = False
        self.snapshot: HttpEndpoint | None = None

    @property
    def can_refresh(self) -> bool:
        """Only an account owner, never a fixed snapshot, can renew credentials."""
        return not isinstance(self._context, HttpEndpoint)

    def request_refresh(self) -> None:
        self._refresh_pending = True

    async def resolve(self) -> HttpEndpoint:
        endpoint = await self._context.endpoint(force_refresh=self._refresh_pending)
        self.snapshot = endpoint
        self._refresh_pending = False
        return endpoint
