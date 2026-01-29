# cc-nim

Use **Claude Code CLI for free** with NVIDIA NIM's free unlimited 40 reqs/min API. This lightweight proxy converts Claude Code's Anthropic API requests to NVIDIA NIM format. **Includes Telegram bot integration** for remote control from your phone!

## Quick Start

### 1. Get Your Free NVIDIA API Key

1. Visit [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
2. Sign in or create a free NVIDIA account
3. Generate a new API key (`nvapi-...`)
4. Install [claude-code](https://github.com/anthropics/claude-code)
5. Install [uv](https://github.com/astral-sh/uv)

### 2. Install & Configure

```bash
git clone https://github.com/Alishahryar1/cc-nim.git
cd cc-nim

cp .env.example .env
```

Edit `.env`:

```dotenv
NVIDIA_NIM_API_KEY=nvapi-your-key-here
BIG_MODEL=moonshotai/kimi-k2-thinking
SMALL_MODEL=moonshotai/kimi-k2-thinking
```
Set Claude-Code to Sonnet for `BIG_MODEL` and Haiku for `SMALL_MODEL` 

---

### Claude Code

**Terminal 1 - Start the proxy:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2 - Run Claude Code:**

```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

That's it! Claude Code now uses NVIDIA NIM for free.

---

### Telegram Bot Integration

Control Claude Code remotely via Telegram! Send tasks from your phone and watch Claude work.

#### Setup

1. **Get Telegram API credentials** from [my.telegram.org](https://my.telegram.org):
   - Log in with your phone number
   - Go to "API Development Tools"
   - Create an app and copy your `api_id` and `api_hash`

2. **Add to `.env`:**

```dotenv
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash_here
TELEGRAM_USER_ID=your_telegram_user_id
```

> 💡 To find your Telegram user ID, message [@userinfobot](https://t.me/userinfobot) on Telegram.

3. **Configure the workspace** (where Claude will operate):

```dotenv
CLAUDE_WORKSPACE=./agent_workspace
ALLOWED_DIRS=C:/Users/yourname/projects
```

4. **Start the server:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

5. **Authenticate Telegram** (first run only):
   - The server will prompt for your phone number and code
   - This creates a `claude_bot.session` file for future runs

#### Usage

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

| Variable                 | Description                    | Default                               |
| ------------------------ | ------------------------------ | ------------------------------------- |
| `NVIDIA_NIM_API_KEY`     | Your NVIDIA API key            | required                              |
| `BIG_MODEL`              | Model for Sonnet/Opus requests | `moonshotai/kimi-k2-thinking`         |
| `SMALL_MODEL`            | Model for Haiku requests       | `moonshotai/kimi-k2-thinking`         |
| `NVIDIA_NIM_BASE_URL`    | NIM endpoint                   | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_NIM_RATE_LIMIT`  | Requests per window            | `40`                                  |
| `NVIDIA_NIM_RATE_WINDOW` | Rate limit window (seconds)    | `60`                                  |

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
