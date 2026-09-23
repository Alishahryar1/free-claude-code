"""OpenAI browser and device authorization flows."""

import asyncio
import base64
import hashlib
import html
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
from aiohttp import web

OPENAI_AUTH_ISSUER = "https://auth.openai.com"
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_ORIGINATOR = "codex_cli_rs"
OPENAI_OAUTH_SCOPE = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
OPENAI_TOKEN_URL = f"{OPENAI_AUTH_ISSUER}/oauth/token"
OPENAI_REVOKE_URL = f"{OPENAI_AUTH_ISSUER}/oauth/revoke"
OPENAI_DEVICE_USER_CODE_URL = f"{OPENAI_AUTH_ISSUER}/api/accounts/deviceauth/usercode"
OPENAI_DEVICE_TOKEN_URL = f"{OPENAI_AUTH_ISSUER}/api/accounts/deviceauth/token"
OPENAI_DEVICE_VERIFICATION_URL = f"{OPENAI_AUTH_ISSUER}/codex/device"
OPENAI_DEVICE_REDIRECT_URI = f"{OPENAI_AUTH_ISSUER}/deviceauth/callback"
OPENAI_CALLBACK_HOST = "localhost"
LOGIN_LIFETIME_SECONDS = 15 * 60


class OpenAILoginError(RuntimeError):
    """An interactive authorization flow could not complete."""


@dataclass(frozen=True, slots=True, repr=False)
class AuthorizationGrant:
    """OAuth authorization code and matching PKCE material."""

    code: str
    redirect_uri: str
    code_verifier: str


@dataclass(frozen=True, slots=True, repr=False)
class _DeviceAuthorization:
    verification_url: str
    user_code: str
    expires_at: int
    interval: float
    device_auth_id: str


class BrowserAuthorization:
    """Short-lived loopback callback server for one PKCE login."""

    def __init__(
        self,
        *,
        auth_url: str,
        redirect_uri: str,
        code_verifier: str,
        runner: web.AppRunner,
        result: asyncio.Future[AuthorizationGrant],
    ) -> None:
        self.auth_url = auth_url
        self.redirect_uri = redirect_uri
        self.code_verifier = code_verifier
        self._runner = runner
        self._result = result
        self._closed = False

    @classmethod
    async def start(cls) -> BrowserAuthorization:
        """Bind an allow-listed loopback port and construct the authorization URL."""

        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
        loop = asyncio.get_running_loop()
        result: asyncio.Future[AuthorizationGrant] = loop.create_future()
        runner: web.AppRunner | None = None
        redirect_uri = ""
        for candidate in (1455, 1457):
            candidate_redirect_uri = (
                f"http://{OPENAI_CALLBACK_HOST}:{candidate}/auth/callback"
            )
            app = web.Application()

            async def callback(
                request: web.Request,
                callback_redirect_uri: str = candidate_redirect_uri,
            ) -> web.Response:
                if request.query.get("state") != state:
                    return web.Response(status=400, text="State mismatch")
                if error := request.query.get("error"):
                    description = request.query.get("error_description", error)
                    if not result.done():
                        result.set_exception(OpenAILoginError(description))
                    return _callback_page("Sign-in failed", description, status=400)
                code = request.query.get("code")
                if not code:
                    return web.Response(status=400, text="Missing authorization code")
                if not result.done():
                    result.set_result(
                        AuthorizationGrant(
                            code=code,
                            redirect_uri=callback_redirect_uri,
                            code_verifier=verifier,
                        )
                    )
                return _callback_page(
                    "Connected",
                    "OpenAI is connected to Free Claude Code. You can close this tab.",
                    close_window=True,
                )

            app.router.add_get("/auth/callback", callback)
            candidate_runner = web.AppRunner(app, access_log=None)
            await candidate_runner.setup()
            try:
                candidate_site = web.TCPSite(
                    candidate_runner,
                    OPENAI_CALLBACK_HOST,
                    candidate,
                )
                await candidate_site.start()
            except OSError:
                await candidate_runner.cleanup()
                continue
            runner = candidate_runner
            redirect_uri = candidate_redirect_uri
            break
        if runner is None:
            raise OpenAILoginError(
                "OpenAI sign-in could not bind localhost ports 1455 or 1457."
            )

        auth_url = f"{OPENAI_AUTH_ISSUER}/oauth/authorize?" + urlencode(
            {
                "response_type": "code",
                "client_id": OPENAI_OAUTH_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": OPENAI_OAUTH_SCOPE,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
                "originator": OPENAI_CODEX_ORIGINATOR,
            }
        )
        return cls(
            auth_url=auth_url,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
            runner=runner,
            result=result,
        )

    async def wait(self) -> AuthorizationGrant:
        """Wait for one valid callback, bounded by the login lifetime."""

        try:
            return await asyncio.wait_for(self._result, timeout=LOGIN_LIFETIME_SECONDS)
        except TimeoutError as exc:
            raise OpenAILoginError("OpenAI sign-in timed out.") from exc

    async def close(self) -> None:
        """Stop the callback listener."""

        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.cancel()
        await self._runner.cleanup()


async def request_device_authorization(
    client: httpx.AsyncClient,
) -> _DeviceAuthorization:
    """Request a device code from OpenAI's documented Codex flow."""

    response = await client.post(
        OPENAI_DEVICE_USER_CODE_URL,
        json={"client_id": OPENAI_OAUTH_CLIENT_ID},
    )
    response.raise_for_status()
    payload = response.json()
    device_auth_id = payload.get("device_auth_id")
    user_code = payload.get("user_code", payload.get("usercode"))
    if not isinstance(device_auth_id, str) or not isinstance(user_code, str):
        raise OpenAILoginError("OpenAI returned an invalid device-code response.")
    try:
        interval = max(1.0, float(payload.get("interval", 5)))
    except TypeError, ValueError:
        interval = 5.0
    return _DeviceAuthorization(
        verification_url=OPENAI_DEVICE_VERIFICATION_URL,
        user_code=user_code,
        expires_at=int(time.time()) + LOGIN_LIFETIME_SECONDS,
        interval=interval,
        device_auth_id=device_auth_id,
    )


async def wait_for_device_grant(
    client: httpx.AsyncClient,
    authorization: _DeviceAuthorization,
) -> AuthorizationGrant:
    """Poll until OpenAI issues the PKCE-backed authorization code."""

    while time.time() < authorization.expires_at:
        response = await client.post(
            OPENAI_DEVICE_TOKEN_URL,
            json={
                "device_auth_id": authorization.device_auth_id,
                "user_code": authorization.user_code,
            },
        )
        if response.is_success:
            payload = response.json()
            code = payload.get("authorization_code")
            verifier = payload.get("code_verifier")
            if not isinstance(code, str) or not isinstance(verifier, str):
                raise OpenAILoginError(
                    "OpenAI returned an invalid device authorization response."
                )
            return AuthorizationGrant(
                code=code,
                redirect_uri=OPENAI_DEVICE_REDIRECT_URI,
                code_verifier=verifier,
            )
        if response.status_code not in {403, 404}:
            response.raise_for_status()
        await asyncio.sleep(authorization.interval)
    raise OpenAILoginError("OpenAI device sign-in timed out.")


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _callback_page(
    title: str,
    message: str,
    *,
    status: int = 200,
    close_window: bool = False,
) -> web.Response:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    close_script = "<script>window.close()</script>" if close_window else ""
    return web.Response(
        status=status,
        content_type="text/html",
        text=(
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{safe_title}</title>"
            "<style>body{font-family:system-ui;max-width:36rem;margin:12vh auto;"
            "padding:2rem;color:#202124}h1{font-size:1.6rem}</style>"
            f"<h1>{safe_title}</h1><p>{safe_message}</p>"
            f"{close_script}"
        ),
    )
