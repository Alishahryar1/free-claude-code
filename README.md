<div align="center">

# 🚀 Free Claude Code

### Use Claude Code for free with NVIDIA NIM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Tested with Pytest](https://img.shields.io/badge/testing-Pytest-00c0ff.svg?style=for-the-badge)](https://github.com/Alishahryar1/free-claude-code/actions/workflows/tests.yml)
[![Type checking: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00.svg?style=for-the-badge)](https://pypi.org/project/ty/)
[![Code style: Ruff](https://img.shields.io/badge/code%20formatting-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Logging: Loguru](https://img.shields.io/badge/logging-loguru-4ecdc4.svg?style=for-the-badge)](https://github.com/Delgan/loguru)

A lightweight proxy that converts Claude Code's Anthropic API requests to NVIDIA NIM format.  
**40 reqs/min free** · **Telegram bot** · **VSCode & CLI**

[Quick Start](#quick-start) · [Telegram Bot](#telegram-bot-integration) · [Models](#available-models) · [Configuration](#configuration)

---

</div>

![Claude Code exploring cc-nim](pic.png)

## Quick Start

### 1. Prerequisites

1. Get a new API key from [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
2. Install [claude-code](https://github.com/anthropics/claude-code)
3. Install [uv](https://github.com/astral-sh/uv)

### 2. Clone & Configure

```bash
git clone https://github.com/Alishahryar1/free-claude-code.git
cd free-claude-code

cp .env.example .env
```

Edit `.env`:

```dotenv
NVIDIA_NIM_API_KEY=nvapi-your-key-here
MODEL=moonshotai/kimi-k2-thinking
```

---

### Claude Code CLI

**Terminal 1 - Start the server:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2 - Run Claude Code:**

```bash
ANTHROPIC_AUTH_TOKEN=freecc ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses NVIDIA NIM for free.

---

### Claude Code VSCode Extension

1. Start the server in the terminal:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

2. Open Settings (`Ctrl + ,`).
3. Search for `claude-code.environmentVariables`.
4. Click **Edit in settings.json** and add the following block:

```json
"claude-code.environmentVariables": [
  { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
  { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" },
]
```

5. Reload extensions.

6. **If you see the login screen** ("How do you want to log in?"): Click **Anthropic Console**, then authorize. The extension will start working. You may be redirected to buy credits in the browser—ignore that; the extension already works.

That's it! The Claude Code VSCode extension now uses NVIDIA NIM for free. To go back to Anthropic models just comment out the added block and reload extensions.

---

### Telegram Bot Integration

Control Claude Code remotely via Telegram! Set an allowed directory, send tasks from your phone, and watch Claude-Code autonomously work on multiple tasks.

#### Setup

1. **Get a Bot Token**:
   - Open Telegram and message [@BotFather](https://t.me/BotFather)
   - Send `/newbot` and follow the prompts
   - Copy the **HTTP API Token**

2. **Edit `.env`:**

```dotenv
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
ALLOWED_TELEGRAM_USER_ID=your_telegram_user_id
```

> 💡 To find your Telegram user ID, message [@userinfobot](https://t.me/userinfobot) on Telegram.

3. **Configure the workspace** (where Claude will operate):

```dotenv
CLAUDE_WORKSPACE=./agent_workspace
ALLOWED_DIR=C:/Users/yourname/projects
```

4. **Start the server:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

5. **Usage**:
   - **Send a message** to the bot on Telegram with a task
   - Claude will respond with:
     - 💭 **Thinking tokens** (reasoning steps)
     - 🔧 **Tool calls** as they execute
     - ✅ **Final result** when complete
   - Send `/stop` to cancel all running tasks
   - Reply `/stop` to a running task to cancel it
   - Send `/clear` to clear the chat and delete all sessions from memory

## Available Models

See [`nvidia_nim_models.json`](nvidia_nim_models.json) for the full list of supported models.

Popular choices:

- `z-ai/glm5`
- `stepfun-ai/step-3.5-flash`
- `moonshotai/kimi-k2.5`
- `minimaxai/minimax-m2.1`
- `mistralai/devstral-2-123b-instruct-2512`

Browse all models at [build.nvidia.com](https://build.nvidia.com/explore/discover)

### Updating the Model List

To update `nvidia_nim_models.json` with the latest models from NVIDIA NIM, run the following command:

```bash
curl "https://integrate.api.nvidia.com/v1/models" > nvidia_nim_models.json
```

## Configuration

| Variable                          | Description                     | Default                       |
| --------------------------------- | ------------------------------- | ----------------------------- |
| `NVIDIA_NIM_API_KEY`              | Your NVIDIA API key             | required                      |
| `MODEL`                           | Model to use for all requests   | `stepfun-ai/step-3.5-flash` |
| `CLAUDE_WORKSPACE`                | Directory for agent workspace   | `./agent_workspace`           |
| `ALLOWED_DIR`                     | Allowed directories for agent   | `""`                          |
| `MAX_CLI_SESSIONS`                | Max concurrent CLI sessions     | `10`                          |
| `FAST_PREFIX_DETECTION`           | Enable fast prefix detection    | `true`                        |
| `ENABLE_NETWORK_PROBE_MOCK`       | Enable network probe mock       | `true`                        |
| `ENABLE_TITLE_GENERATION_SKIP`    | Skip title generation           | `true`                        |
| `ENABLE_SUGGESTION_MODE_SKIP`     | Skip suggestion mode            | `true`                        |
| `ENABLE_FILEPATH_EXTRACTION_MOCK` | Enable filepath extraction mock | `true`                        |
| `TELEGRAM_BOT_TOKEN`              | Telegram Bot Token              | `""`                          |
| `ALLOWED_TELEGRAM_USER_ID`        | Allowed Telegram User ID        | `""`                          |
| `MESSAGING_RATE_LIMIT`            | Telegram messages per window    | `1`                           |
| `MESSAGING_RATE_WINDOW`           | Messaging window (seconds)      | `1`                           |
| `NVIDIA_NIM_RATE_LIMIT`           | API requests per window         | `40`                          |
| `NVIDIA_NIM_RATE_WINDOW`          | Rate limit window (seconds)     | `60`                          |

The NVIDIA NIM base URL is fixed to `https://integrate.api.nvidia.com/v1`.

**NIM Settings (prefix `NVIDIA_NIM_`)**

| Variable                                | Description                   | Default |
| --------------------------------------- | ----------------------------- | ------- |
| `NVIDIA_NIM_TEMPERATURE`                | Sampling temperature          | `1.0`   |
| `NVIDIA_NIM_TOP_P`                      | Top-p nucleus sampling        | `1.0`   |
| `NVIDIA_NIM_TOP_K`                      | Top-k sampling                | `-1`    |
| `NVIDIA_NIM_MAX_TOKENS`                 | Max tokens for generation     | `81920` |
| `NVIDIA_NIM_PRESENCE_PENALTY`           | Presence penalty              | `0.0`   |
| `NVIDIA_NIM_FREQUENCY_PENALTY`          | Frequency penalty             | `0.0`   |
| `NVIDIA_NIM_MIN_P`                      | Min-p sampling                | `0.0`   |
| `NVIDIA_NIM_REPETITION_PENALTY`         | Repetition penalty            | `1.0`   |
| `NVIDIA_NIM_SEED`                       | RNG seed (blank = unset)      | unset   |
| `NVIDIA_NIM_STOP`                       | Stop string (blank = unset)   | unset   |
| `NVIDIA_NIM_PARALLEL_TOOL_CALLS`        | Parallel tool calls           | `true`  |
| `NVIDIA_NIM_RETURN_TOKENS_AS_TOKEN_IDS` | Return token ids              | `false` |
| `NVIDIA_NIM_INCLUDE_STOP_STR_IN_OUTPUT` | Include stop string in output | `false` |
| `NVIDIA_NIM_IGNORE_EOS`                 | Ignore EOS token              | `false` |
| `NVIDIA_NIM_MIN_TOKENS`                 | Minimum generated tokens      | `0`     |
| `NVIDIA_NIM_CHAT_TEMPLATE`              | Chat template override        | unset   |
| `NVIDIA_NIM_REQUEST_ID`                 | Request id override           | unset   |
| `NVIDIA_NIM_REASONING_EFFORT`           | Reasoning effort              | `high`  |
| `NVIDIA_NIM_INCLUDE_REASONING`          | Include reasoning in response | `true`  |

All `NVIDIA_NIM_*` settings are strictly validated; unknown keys with this prefix will cause startup errors.

See [`.env.example`](.env.example) for all supported parameters.

## Development

### Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

To run type checking:

```bash
uv run ty check
```

To run formatting:

```bash
uv run ruff format
```

### Adding Your Own Provider

Extend `BaseProvider` in `providers/` to add support for other APIs:

```python
from providers.base import BaseProvider, ProviderConfig

class MyProvider(BaseProvider):
    async def stream_response(self, request, input_tokens=0):
        # Yield Anthropic SSE format events
        pass
```

### Adding Your Own Messaging App

Extend `MessagingPlatform` in `messaging/` to add support for other platforms (Discord, Slack, etc.):

```python
from messaging.base import MessagingPlatform
from messaging.models import IncomingMessage

class MyPlatform(MessagingPlatform):
    async def start(self):
        # Initialize connection
        pass

    async def stop(self):
        # Cleanup
        pass

    async def queue_send_message(self, chat_id, text, **kwargs):
        # Send message to platform
        pass

    async def queue_edit_message(self, chat_id, message_id, text, **kwargs):
        # Edit existing message
        pass

    def on_message(self, handler):
        # Register callback for incoming messages
        # Handler expects an IncomingMessage object
        pass
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
