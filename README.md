<div align="center">

# Free Claude Code

### Use Claude Code CLI & VSCode — for free. No Anthropic API key required.

[![GitHub Stars](https://img.shields.io/github/stars/Alishahryar1/free-claude-code?style=for-the-badge&logo=github)](https://github.com/Alishahryar1/free-claude-code/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Alishahryar1/free-claude-code?style=for-the-badge&logo=github)](https://github.com/Alishahryar1/free-claude-code/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Tested with Pytest](https://img.shields.io/badge/testing-Pytest-00c0ff.svg?style=for-the-badge)](https://github.com/Alishahryar1/free-claude-code/actions/workflows/tests.yml)
[![Type checking: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00.svg?style=for-the-badge)](https://pypi.org/project/ty/)
[![Code style: Ruff](https://img.shields.io/badge/code%20formatting-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Logging: Loguru](https://img.shields.io/badge/logging-loguru-4ecdc4.svg?style=for-the-badge)](https://github.com/Delgan/loguru)

A lightweight proxy server that translates Claude Code's Anthropic API calls into **NVIDIA NIM**, **OpenRouter**, or **LM Studio** format.
Get **40 free requests/min** on NVIDIA NIM, access **hundreds of models** on OpenRouter, or run **fully local** with LM Studio.

[Features](#features) · [Quick Start](#quick-start) · [How It Works](#how-it-works) · [Discord Bot](#discord-bot) · [Configuration](#configuration)

---

</div>

<div align="center">
  <img src="pic.png" alt="Free Claude Code in action" width="700">
  <p><em>Claude Code running via NVIDIA NIM — completely free</em></p>
</div>

## Features

| Feature | Description |
|---------|-------------|
| **Zero Cost** | 40 req/min free on NVIDIA NIM. Free models on OpenRouter. Fully local with LM Studio |
| **Drop-in Replacement** | Set 2 env vars — no modifications to Claude Code CLI or VSCode extension needed |
| **3 Providers** | NVIDIA NIM, OpenRouter (hundreds of models), LM Studio (local & offline) |
| **Thinking Token Support** | Parses `<think>` tags and `reasoning_content` into native Claude thinking blocks |
| **Heuristic Tool Parser** | Models outputting tool calls as text are auto-parsed into structured tool use |
| **Request Optimization** | 5 categories of trivial API calls intercepted locally — saves quota and latency |
| **Discord Bot** | Remote autonomous coding with tree-based threading, session persistence, and live progress (Telegram also supported) |
| **Smart Rate Limiting** | Proactive rolling-window throttle + reactive 429 exponential backoff across all providers |
| **Subagent Control** | Task tool interception forces `run_in_background=False` — no runaway subagents |
| **Extensible** | Clean `BaseProvider` and `MessagingPlatform` ABCs — add new providers or platforms easily |

## Quick Start

### Prerequisites

1. Get an API key (or use LM Studio locally):
   - **NVIDIA NIM**: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
   - **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys)
   - **LM Studio**: No API key needed — run locally with [LM Studio](https://lmstudio.ai)
2. Install [Claude Code](https://github.com/anthropics/claude-code)
3. Install [uv](https://github.com/astral-sh/uv)

### Clone & Configure

```bash
git clone https://github.com/Alishahryar1/free-claude-code.git
cd free-claude-code
cp .env.example .env
```

Choose your provider and edit `.env`:

<details>
<summary><b>NVIDIA NIM</b> (recommended — 40 req/min free)</summary>

```dotenv
PROVIDER_TYPE=nvidia_nim
NVIDIA_NIM_API_KEY=nvapi-your-key-here
MODEL=stepfun-ai/step-3.5-flash
```

</details>

<details>
<summary><b>OpenRouter</b> (hundreds of models)</summary>

```dotenv
PROVIDER_TYPE=open_router
OPENROUTER_API_KEY=sk-or-your-key-here
MODEL=stepfun/step-3.5-flash:free
```

</details>

<details>
<summary><b>LM Studio</b> (fully local, no API key)</summary>

```dotenv
PROVIDER_TYPE=lmstudio
MODEL=lmstudio-community/qwen2.5-7b-instruct
```

</details>

### Run It

**Terminal 1** — Start the proxy server:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2** — Run Claude Code:

```bash
ANTHROPIC_AUTH_TOKEN=freecc ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses your configured provider for free.

<details>
<summary><b>VSCode Extension Setup</b></summary>

1. Start the proxy server (same as above).
2. Open Settings (`Ctrl + ,`) and search for `claude-code.environmentVariables`.
3. Click **Edit in settings.json** and add:

```json
"claude-code.environmentVariables": [
  { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
  { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" }
]
```

4. Reload extensions.
5. **If you see the login screen** ("How do you want to log in?"): Click **Anthropic Console**, then authorize. The extension will start working. You may be redirected to buy credits in the browser — ignore that; the extension already works.

To switch back to Anthropic models, comment out the added block and reload extensions.

</details>

---

## How It Works

```
┌─────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
│  Claude Code    │───────>│  Free Claude Code    │───────>│  LLM Provider    │
│  CLI / VSCode   │<───────│  Proxy (:8082)       │<───────│  NIM / OR / LMS  │
└─────────────────┘        └──────────────────────┘        └──────────────────┘
   Anthropic API                     │                       OpenAI-compatible
   format (SSE)              ┌───────┴────────┐                format (SSE)
                             │ Optimizations  │
                             ├────────────────┤
                             │ Quota probes   │
                             │ Title gen skip │
                             │ Prefix detect  │
                             │ Suggestion skip│
                             │ Filepath mock  │
                             └────────────────┘
```

- **Transparent proxy** — Claude Code sends standard Anthropic API requests to the proxy server
- **Request optimization** — 5 categories of trivial requests (quota probes, title generation, prefix detection, suggestions, filepath extraction) are intercepted and responded to instantly without using API quota
- **Format translation** — Real requests are translated from Anthropic format to the provider's OpenAI-compatible format and streamed back
- **Thinking tokens** — `<think>` tags and `reasoning_content` fields are converted into native Claude thinking blocks so Claude Code renders them correctly

---

## Providers

| Provider | Cost | Rate Limit | Models | Best For |
|----------|------|------------|--------|----------|
| **NVIDIA NIM** | Free | 40 req/min | Kimi K2, GLM5, Devstral, MiniMax | Daily driver — generous free tier |
| **OpenRouter** | Free / Pay | Varies | 200+ (GPT-4o, Claude, Step, etc.) | Model variety, fallback options |
| **LM Studio** | Free (local) | Unlimited | Any GGUF model | Privacy, offline use, no rate limits |

Switch providers by changing `PROVIDER_TYPE` in `.env`:

| Provider | `PROVIDER_TYPE` | API Key Variable | Base URL |
|----------|-----------------|------------------|----------|
| NVIDIA NIM | `nvidia_nim` | `NVIDIA_NIM_API_KEY` | `integrate.api.nvidia.com/v1` |
| OpenRouter | `open_router` | `OPENROUTER_API_KEY` | `openrouter.ai/api/v1` |
| LM Studio | `lmstudio` | (none) | `localhost:1234/v1` |

OpenRouter gives access to hundreds of models (StepFun, OpenAI, Anthropic, etc.) through a single API. Set `MODEL` to any OpenRouter model ID.

LM Studio runs locally — start the server in LM Studio's Developer tab or via `lms server start`, load a model, and set `MODEL` to the model identifier.

---

## Discord Bot

Control Claude Code remotely from Discord. Send tasks, watch live progress, and manage multiple concurrent sessions. Discord is the default messaging platform; Telegram is also supported.

**Capabilities:**
- Tree-based message threading — reply to messages to fork conversations
- Session persistence across server restarts
- Live streaming of thinking tokens, tool calls, and results
- Up to 10 concurrent Claude CLI sessions
- Commands: `/stop` (cancel tasks), `/clear` (reset all sessions), `/stats`

### Setup

1. **Create a Discord Bot** — Go to [Discord Developer Portal](https://discord.com/developers/applications), create an application, add a bot, and copy the token. Enable **Message Content Intent** under Bot settings.

2. **Edit `.env`:**

```dotenv
MESSAGING_PLATFORM=discord
DISCORD_BOT_TOKEN=your_discord_bot_token
ALLOWED_DISCORD_CHANNELS=123456789,987654321
```

> Enable Developer Mode in Discord (Settings → Advanced), then right-click a channel and "Copy ID" to get channel IDs. Comma-separate multiple channels. If empty, no channels are allowed.

3. **Configure the workspace** (where Claude will operate):

```dotenv
CLAUDE_WORKSPACE=./agent_workspace
ALLOWED_DIR=C:/Users/yourname/projects
```

4. **Start the server:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

5. **Invite the bot** to your server (OAuth2 → URL Generator, scopes: `bot`, permissions: Read Messages, Send Messages, Manage Messages, Read Message History). Send a message in an allowed channel with a task. Claude responds with thinking tokens, tool calls as they execute, and the final result. Reply `/stop` to a running task to cancel it.

### Telegram (Alternative)

To use Telegram instead, set `MESSAGING_PLATFORM=telegram` and configure:

```dotenv
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
ALLOWED_TELEGRAM_USER_ID=your_telegram_user_id
```

Get a token from [@BotFather](https://t.me/BotFather); find your user ID via [@userinfobot](https://t.me/userinfobot).

---

## Models

<details>
<summary><b>NVIDIA NIM</b></summary>

Full list in [`nvidia_nim_models.json`](nvidia_nim_models.json).

Popular models:
- `z-ai/glm5`
- `stepfun-ai/step-3.5-flash`
- `moonshotai/kimi-k2.5`
- `minimaxai/minimax-m2.1`
- `mistralai/devstral-2-123b-instruct-2512`

Browse: [build.nvidia.com](https://build.nvidia.com/explore/discover)

Update model list:
```bash
curl "https://integrate.api.nvidia.com/v1/models" > nvidia_nim_models.json
```

</details>

<details>
<summary><b>OpenRouter</b></summary>

Hundreds of models from StepFun, OpenAI, Anthropic, Google, and more.

Popular models:
- `stepfun/step-3.5-flash:free`
- `deepseek/deepseek-r1-0528:free`
- `openai/gpt-oss-120b:free`

Browse: [openrouter.ai/models](https://openrouter.ai/models)

Browse free models: [https://openrouter.ai/collections/free-models](https://openrouter.ai/collections/free-models)

</details>

<details>
<summary><b>LM Studio</b></summary>

Run models locally with [LM Studio](https://lmstudio.ai). Load a model in the Chat or Developer tab, then set `MODEL` to its identifier.

Examples (native tool-use support):
- `lmstudio-community/qwen2.5-7b-instruct`
- `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF`
- `bartowski/Ministral-8B-Instruct-2410-GGUF`

Browse: [model.lmstudio.ai](https://model.lmstudio.ai)

</details>

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVIDER_TYPE` | Provider: `nvidia_nim`, `open_router`, or `lmstudio` | `nvidia_nim` |
| `MODEL` | Model to use for all requests | `stepfun-ai/step-3.5-flash` |
| `NVIDIA_NIM_API_KEY` | NVIDIA API key (NIM provider) | required |
| `OPENROUTER_API_KEY` | OpenRouter API key (OpenRouter provider) | required |
| `LM_STUDIO_BASE_URL` | LM Studio server URL | `http://localhost:1234/v1` |
| `PROVIDER_RATE_LIMIT` | LLM API requests per window | `40` |
| `PROVIDER_RATE_WINDOW` | Rate limit window (seconds) | `60` |
| `HTTP_READ_TIMEOUT` | Read timeout for provider API requests (seconds) | `300` |
| `HTTP_WRITE_TIMEOUT` | Write timeout for provider API requests (seconds) | `10` |
| `HTTP_CONNECT_TIMEOUT` | Connect timeout for provider API requests (seconds) | `2` |
| `FAST_PREFIX_DETECTION` | Enable fast prefix detection | `true` |
| `ENABLE_NETWORK_PROBE_MOCK` | Enable network probe mock | `true` |
| `ENABLE_TITLE_GENERATION_SKIP` | Skip title generation | `true` |
| `ENABLE_SUGGESTION_MODE_SKIP` | Skip suggestion mode | `true` |
| `ENABLE_FILEPATH_EXTRACTION_MOCK` | Enable filepath extraction mock | `true` |
| `MESSAGING_PLATFORM` | Messaging platform: `discord` or `telegram` | `discord` |
| `DISCORD_BOT_TOKEN` | Discord Bot Token | `""` |
| `ALLOWED_DISCORD_CHANNELS` | Comma-separated channel IDs (empty = none allowed) | `""` |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token | `""` |
| `ALLOWED_TELEGRAM_USER_ID` | Allowed Telegram User ID | `""` |
| `MESSAGING_RATE_LIMIT` | Messaging messages per window | `1` |
| `MESSAGING_RATE_WINDOW` | Messaging window (seconds) | `1` |
| `CLAUDE_WORKSPACE` | Directory for agent workspace | `./agent_workspace` |
| `ALLOWED_DIR` | Allowed directories for agent | `""` |
| `MAX_CLI_SESSIONS` | Max concurrent CLI sessions | `10` |

See [`.env.example`](.env.example) for all supported parameters.

---

## Development

### Project Structure

```
free-claude-code/
├── server.py              # Entry point
├── api/                   # FastAPI routes, request detection, optimization handlers
├── providers/             # BaseProvider ABC + NVIDIA NIM, OpenRouter, LM Studio
├── messaging/             # MessagingPlatform ABC + Discord/Telegram bots, session management
├── config/                # Settings, NIM config, logging
├── cli/                   # CLI session and process management
├── utils/                 # Text utilities
└── tests/                 # Pytest test suite
```

### Commands

```bash
uv run pytest          # Run tests
uv run ty check        # Type checking
uv run ruff check      # Code style checking
uv run ruff format     # Code formatting
```

---

## Extending

### Adding a Provider

Extend `BaseProvider` in `providers/` to add support for other APIs:

```python
from providers.base import BaseProvider, ProviderConfig

class MyProvider(BaseProvider):
    async def stream_response(self, request, input_tokens=0, *, request_id=None):
        # Yield Anthropic SSE format events
        ...
```

### Adding a Messaging Platform

Extend `MessagingPlatform` in `messaging/` to add Slack or other platforms:

```python
from messaging.base import MessagingPlatform

class MyPlatform(MessagingPlatform):
    async def start(self):
        # Initialize connection
        ...

    async def stop(self):
        # Cleanup
        ...

    async def send_message(self, chat_id, text, reply_to=None, parse_mode=None):
        # Send a message
        ...

    async def edit_message(self, chat_id, message_id, text, parse_mode=None):
        # Edit an existing message
        ...

    def on_message(self, handler):
        # Register callback for incoming messages
        ...
```

---

## Contributing

Contributions are welcome! Here are some ways to help:

- Report bugs or suggest features via [Issues](https://github.com/Alishahryar1/free-claude-code/issues)
- Add new LLM providers (Groq, Together AI, etc.)
- Add new messaging platforms (Slack, etc.)
- Improve test coverage

```bash
# Fork the repo, then:
git checkout -b my-feature
# Make your changes
uv run pytest && uv run ty check && uv run ruff check && uv run ruff format --check
# Open a pull request
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Built with [FastAPI](https://fastapi.tiangolo.com/), [OpenAI Python SDK](https://github.com/openai/openai-python), [discord.py](https://github.com/Rapptz/discord.py), and [python-telegram-bot](https://python-telegram-bot.org/).
