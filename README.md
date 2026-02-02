# cc-nim

Use **Claude Code CLI for free** with NVIDIA NIM's free unlimited 40 reqs/min API. This lightweight proxy converts Claude Code's Anthropic API requests to NVIDIA NIM format. **Includes Telegram bot integration** for remote control from your phone!

## Quick Start

### 1. Get Your Free NVIDIA API Key

1. Get a new API key from [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
2. Install [claude-code](https://github.com/anthropics/claude-code)
3. Install [uv](https://github.com/astral-sh/uv)

### 2. Clone & Configure

```bash
git clone https://github.com/Alishahryar1/cc-nim.git
cd cc-nim

cp .env.example .env
```

Edit `.env`:

```dotenv
NVIDIA_NIM_API_KEY=nvapi-your-key-here
MODEL=moonshotai/kimi-k2-thinking
```

---

### Claude Code

**Terminal 1 - Start the proxy:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2 - Run Claude Code:**

```bash
ANTHROPIC_AUTH_TOKEN=ccnim ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses NVIDIA NIM for free.

---

### Telegram Bot Integration

Control Claude Code remotely via Telegram! Set an allowed directory, send tasks from your phone, and watch Claude-Code autonomously work on multiple tasks.

#### Setup

1. **Get a Bot Token**:
   - Open Telegram and message [@BotFather](https://t.me/BotFather)
   - Send `/newbot` and follow the prompts
   - Copy the **HTTP API Token**

2. **Add to `.env`:**

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
   - Send `/start` to your bot
   - Send any text prompt to start a task

- **Send a message** to yourself on Telegram with a task
- Claude will respond with:
  - 💭 **Thinking tokens** (reasoning steps)
  - 🔧 **Tool calls** as they execute
  - ✅ **Final result** when complete
- Send `/stop` to cancel a running task

## Available Models

See [`nvidia_nim_models.json`](nvidia_nim_models.json) for the full list of supported models.

Popular choices:

- `moonshotai/kimi-k2.5`
- `z-ai/glm4.7`
- `minimaxai/minimax-m2.1`
- `mistralai/devstral-2-123b-instruct-2512`

Browse all models at [build.nvidia.com](https://build.nvidia.com/explore/discover)

### Updating the Model List

To update `nvidia_nim_models.json` with the latest models from NVIDIA NIM, run the following command:

```bash
curl "https://integrate.api.nvidia.com/v1/models" > nvidia_nim_models.json
```

## Configuration

| Variable                   | Description                   | Default                               |
| -------------------------- | ----------------------------- | ------------------------------------- |
| `NVIDIA_NIM_API_KEY`       | Your NVIDIA API key           | required                              |
| `MODEL`                    | Model to use for all requests | `moonshotai/kimi-k2-thinking`         |
| `NVIDIA_NIM_BASE_URL`      | NIM endpoint                  | `https://integrate.api.nvidia.com/v1` |
| `CLAUDE_WORKSPACE`         | Directory for agent workspace | `./agent_workspace`                   |
| `ALLOWED_DIR`              | Allowed directories for agent | `""`                                  |
| `MAX_CLI_SESSIONS`         | Max concurrent CLI sessions   | `10`                                  |
| `TELEGRAM_BOT_TOKEN`       | Telegram Bot Token            | `""`                                  |
| `ALLOWED_TELEGRAM_USER_ID` | Allowed Telegram User ID      | `""`                                  |
| `MESSAGING_RATE_LIMIT`     | Telegram messages per window  | `1`                                  |
| `MESSAGING_RATE_WINDOW`    | Messaging window (seconds)    | `1`                                   |
| `NVIDIA_NIM_RATE_LIMIT`    | API requests per window       | `40`                                  |
| `NVIDIA_NIM_RATE_WINDOW`   | Rate limit window (seconds)   | `60`                                  |
| `NVIDIA_NIM_TEMPERATURE`   | Model temperature             | `1.0`                                 |
| `NVIDIA_NIM_MAX_TOKENS`    | Max tokens for generation     | `81920`                               |

See [`.env.example`](.env.example) for all supported parameters.

## Development

### Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

### Adding Your Own Provider

Extend `BaseProvider` in `providers/` to add support for other APIs:

```python
from providers.base import BaseProvider, ProviderConfig

class MyProvider(BaseProvider):
    async def complete(self, request):
        # Make API call, return raw JSON
        pass

    async def stream_response(self, request, input_tokens=0):
        # Yield Anthropic SSE format events
        pass

    def convert_response(self, response_json, original_request):
        # Convert to Anthropic response format
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
