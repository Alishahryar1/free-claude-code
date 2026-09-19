"""CodeBuddy/WorkBuddy upstream site presets.

The intl-cli site (codebuddy.ai) is the only enabled surface today; cn-cli
(copilot.tencent.com) uses the same protocol with different identity headers
and is kept here so a future change can enable it without reshaping config.
"""

from dataclasses import dataclass

CODEBUDDY_USER_AGENT = "CLI/2.63.2 CodeBuddy/2.63.2"
CODEBUDDY_PRODUCT = "SaaS"
# The chat API is OpenAI-compatible under the site's /v2 prefix.
CODEBUDDY_CHAT_API_PREFIX = "/v2"
CODEBUDDY_DEFAULT_SITE_ID = "intl-cli"


@dataclass(frozen=True, slots=True)
class CodeBuddySite:
    """Immutable per-site upstream identity."""

    site_id: str
    api_base: str
    origin: str
    user_agent: str
    product: str
    enabled: bool = True

    @property
    def chat_base_url(self) -> str:
        """Return the OpenAI-compatible root the SDK client posts against."""

        return f"{self.api_base}{CODEBUDDY_CHAT_API_PREFIX}"


SITE_PRESETS: dict[str, CodeBuddySite] = {
    "intl-cli": CodeBuddySite(
        site_id="intl-cli",
        api_base="https://www.codebuddy.ai",
        origin="https://www.codebuddy.ai",
        user_agent=CODEBUDDY_USER_AGENT,
        product=CODEBUDDY_PRODUCT,
    ),
    # Same protocol as intl-cli; reserved for a future opt-in.
    "cn-cli": CodeBuddySite(
        site_id="cn-cli",
        api_base="https://copilot.tencent.com",
        origin="https://www.codebuddy.cn",
        user_agent=CODEBUDDY_USER_AGENT,
        product=CODEBUDDY_PRODUCT,
        enabled=False,
    ),
}


def default_site() -> CodeBuddySite:
    """Return the site preset this provider is scoped to."""

    return SITE_PRESETS[CODEBUDDY_DEFAULT_SITE_ID]
