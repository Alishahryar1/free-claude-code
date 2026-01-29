"""
Claude Code Proxy - NVIDIA NIM Implementation

This server acts as a robust proxy between Anthropic API requests and NVIDIA NIM,
enabling Claude Code CLI to utilize NIM models with full support for:
- Streaming with SSE (Server-Sent Events)
- Thinking/Reasoning blocks and Reasoning-Split mode
- Native and heuristic tool use parsing
- Automatic model mapping (Haiku/Sonnet/Opus to NIM equivalents)
- Fast prefix detection for CLI policy specifications
"""

import time
import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, field_validator, model_validator
from providers.nvidia_nim import NvidiaNimProvider, ProviderConfig
from providers.exceptions import ProviderError
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import tiktoken
from providers.claude_cli import CLIParser
from providers.cli_session_manager import CLISessionManager
from providers.logging_utils import log_request_compact

# Optional: telethon for the bot
try:
    from telethon import TelegramClient, events, errors
except ImportError:
    TelegramClient = None
    events = None
    errors = None

# Initialize tokenizer
ENCODER = tiktoken.get_encoding("cl100k_base")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log", encoding="utf-8", mode="w")],
)
logger = logging.getLogger(__name__)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# =============================================================================
# Models
# =============================================================================

BIG_MODEL = os.getenv("BIG_MODEL", "moonshotai/kimi-k2-instruct")
SMALL_MODEL = os.getenv("SMALL_MODEL", "moonshotai/kimi-k2-instruct")


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
                ContentBlockThinking,
            ]
        ],
    ]
    reasoning_content: Optional[str] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    extra_body: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @model_validator(mode="after")
    def map_model(self) -> "MessagesRequest":
        if self.original_model is None:
            self.original_model = self.model

        clean_v = self.model
        for prefix in ["anthropic/", "openai/", "gemini/"]:
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix) :]
                break

        if "haiku" in clean_v.lower():
            self.model = SMALL_MODEL
        elif "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
            self.model = BIG_MODEL

        if self.model != self.original_model:
            logger.debug(f"MODEL MAPPING: '{self.original_model}' -> '{self.model}'")

        return self


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None

    @field_validator("model")
    @classmethod
    def validate_model_field(cls, v, info):
        clean_v = v
        for prefix in ["anthropic/", "openai/", "gemini/"]:
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix) :]
                break

        if "haiku" in clean_v.lower():
            return SMALL_MODEL
        elif "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
            return BIG_MODEL
        return v


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[
        Union[
            ContentBlockText, ContentBlockToolUse, ContentBlockThinking, Dict[str, Any]
        ]
    ]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# =============================================================================
# Provider
# =============================================================================

provider_config = ProviderConfig(
    api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
    base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    rate_limit=int(os.getenv("NVIDIA_NIM_RATE_LIMIT", "40")),
    rate_window=int(os.getenv("NVIDIA_NIM_RATE_WINDOW", "60")),
)

# Global provider instance for DI
_provider: Optional[NvidiaNimProvider] = None


def get_provider() -> NvidiaNimProvider:
    global _provider
    if _provider is None:
        _provider = NvidiaNimProvider(provider_config)
    return _provider


# =============================================================================
# FastAPI App
# =============================================================================

# Internal storage path for bot data (sessions, etc.) - defined early for lifespan
INTERNAL_DATA_PATH = os.path.abspath(os.getenv("CLAUDE_WORKSPACE", "agent_workspace"))
os.makedirs(INTERNAL_DATA_PATH, exist_ok=True)

tele_client: Optional["TelegramClient"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tele_client
    try:
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        if TelegramClient and api_id and api_hash:
            logger.info("Starting Telegram Bot...")
            session_path = os.path.join(INTERNAL_DATA_PATH, "claude_bot.session")
            tele_client = TelegramClient(session_path, int(api_id), api_hash)

            # Register handlers BEFORE starting
            register_bot_handlers(tele_client)

            await tele_client.start()
            asyncio.create_task(tele_client.run_until_disconnected())

            # Notify user
            try:
                await tele_client.send_message(
                    "me", f"🚀 **Claude unified server is online!** (v{app.version})"
                )
            except:
                pass
            logger.info("Bot started and online message sent.")
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        tele_client = None

    yield
    if tele_client:
        await tele_client.disconnect()
    logger.info("Server shutting down...")
    global _provider
    if _provider and hasattr(_provider, "_client"):
        await _provider._client.aclose()


# =============================================================================
# Telegram Bot & CLI Configuration
# =============================================================================

# The working directory where Claude CLI runs (user's project)
ALLOWED_DIR = os.getenv("ALLOWED_DIR", "")
if ALLOWED_DIR:
    # Handle Windows backslash corrosion (\a, \b etc) by replacing them
    ALLOWED_DIR = (
        ALLOWED_DIR.replace("\a", "\\a")
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\v", "\\v")
    )
    CLI_WORKSPACE = os.path.abspath(os.path.normpath(ALLOWED_DIR))
else:
    # Fallback to internal data path if no ALLOWED_DIR specified
    CLI_WORKSPACE = INTERNAL_DATA_PATH

os.makedirs(CLI_WORKSPACE, exist_ok=True)

# Internal URL for the CLI to use (points to this server)
INTERNAL_API_URL = "http://localhost:8082/v1"

# Initialize Global Instances
# CLI Session Manager - each conversation gets its own CLI instance
cli_session_manager = CLISessionManager(
    workspace_path=CLI_WORKSPACE,
    api_url=INTERNAL_API_URL,
    allowed_dirs=[CLI_WORKSPACE],
    max_sessions=int(os.getenv("MAX_CLI_SESSIONS", "10")),
)

# Session storage and message queue (stored in internal data path, not user's project)
from providers.session_store import SessionStore
from providers.message_queue import MessageQueueManager, QueuedMessage

session_store = SessionStore(os.path.join(INTERNAL_DATA_PATH, "sessions.json"))
message_queue = MessageQueueManager()


def register_bot_handlers(client: "TelegramClient"):
    ALLOWED_USER_ID = os.getenv("ALLOWED_TELEGRAM_USER_ID")
    logger.info(f"DEBUG: Registering bot handlers. Allowed user ID: {ALLOWED_USER_ID}")

    # Global flood wait state (shared across all bot tasks)
    global_flood_wait_until = 0

    async def send_error_to_user(chat_id: int, error_msg: str, context: str = ""):
        """Send a formatted error message to the user."""
        try:
            formatted = f"❌ **Error**"
            if context:
                formatted += f" ({context})"
            formatted += f"\n\n```\n{str(error_msg)[:500]}\n```"
            await client.send_message(chat_id, formatted, parse_mode="markdown")
        except Exception as e:
            logger.error(f"Failed to send error to user: {e}")

    async def process_claude_task(
        session_id_to_resume: Optional[str], queued_msg: QueuedMessage
    ):
        """
        Core task processor - handles a single Claude CLI interaction.
        Now uses CLISessionManager for multi-instance support.
        """
        prompt = queued_msg.prompt
        status_msg_id = queued_msg.reply_msg_id
        chat_id = queued_msg.chat_id
        original_msg_id = queued_msg.msg_id

        # Get the status message object
        try:
            status_msg = await client.get_messages(chat_id, ids=status_msg_id)
        except Exception as e:
            logger.error(f"Failed to get status message: {e}")
            await send_error_to_user(chat_id, str(e), "getting status message")
            return

        # Unified message accumulator
        message_parts = []
        last_ui_update = 0
        captured_session_id = session_id_to_resume
        temp_session_id = None  # Track temp ID for new sessions
        cli_session = None  # The CLISession instance for this task
        flood_alert_sent = False  # Track if we briefed the user about flood

        def safe_markdown_truncate(text, limit=3800):
            """Truncate text carefully to avoid breaking markdown entities or blocks."""
            if len(text) <= limit:
                return text

            # Show the end of the content as it's usually the most relevant
            truncated = "..." + text[-(limit - 5) :]

            # Simple check for unclosed code blocks
            # This is a heuristic but covers common CLI output issues
            if truncated.count("```") % 2 != 0:
                truncated += "\n```"

            return truncated

        def build_unified_message(status=None):
            lines = []
            if status:
                lines.append(status)
                lines.append("")

            for part_type, content in message_parts:
                if part_type == "thinking":
                    display_thinking = content[:1200] + (
                        "..." if len(content) > 1200 else ""
                    )
                    lines.append(f"💭 **Thinking:**\n```\n{display_thinking}\n```")
                elif part_type == "tool":
                    lines.append(f"🔧 **Tools:** `{content}`")
                elif part_type == "subagent":
                    lines.append(f"🤖 **Subagent:** {content}")
                elif part_type == "content":
                    lines.append(content)
                elif part_type == "error":
                    lines.append(f"⚠️ {content}")

            result = "\n".join(lines)
            return safe_markdown_truncate(result)

        async def update_bot_ui(status=None, force=False):
            nonlocal last_ui_update, global_flood_wait_until, flood_alert_sent
            now = time.time()

            # 1. Check if we're in a global flood wait
            if now < global_flood_wait_until:
                # If forced, we check if it's been a while since we logged the skip
                if force and now - last_ui_update > 5:
                    logger.warning(
                        f"BOT: Skipping UI update due to active FloodWait ({int(global_flood_wait_until - now)}s remaining)"
                    )
                return

            # Reset alert flag when we are out of wait
            flood_alert_sent = False

            if not force and now - last_ui_update < 1.0:
                return

            try:
                display = build_unified_message(status)
                if display:
                    await status_msg.edit(display, parse_mode="markdown")
                    last_ui_update = now
            except errors.FloodWaitError as e:
                # Telegram specific flood wait
                wait_time = e.seconds
                global_flood_wait_until = now + wait_time
                logger.error(
                    f"BOT: UI update hit rate limit. Waiting for {wait_time}s. Exception: {e}"
                )

                # Try to alert the user once if the wait is long
                if wait_time > 10 and not flood_alert_sent:
                    try:
                        await client.send_message(
                            chat_id,
                            f"⏳ **Telegram rate limit hit.**\nUI updates will pause for {wait_time}s. Claude is still working in the background.",
                            reply_to=original_msg_id,
                        )
                        flood_alert_sent = True
                    except:
                        pass
            except Exception as e:
                logger.error(f"BOT: UI update failed: {e}")

        try:
            # Get or create CLI session from the manager
            is_resume = session_id_to_resume is not None
            log_prefix = (
                f"Resuming session {session_id_to_resume}"
                if is_resume
                else "Starting new session"
            )
            logger.info(f"BOT: {log_prefix} for prompt: {prompt[:50]}...")

            try:
                (
                    cli_session,
                    session_or_temp_id,
                    is_new,
                ) = await cli_session_manager.get_or_create_session(
                    session_id=session_id_to_resume
                )
                if is_new:
                    temp_session_id = session_or_temp_id
                    logger.info(
                        f"BOT: Created new CLI session with temp_id: {temp_session_id}"
                    )
                else:
                    captured_session_id = session_or_temp_id
                    logger.info(f"BOT: Reusing CLI session: {captured_session_id}")
            except RuntimeError as e:
                # Max sessions reached
                logger.warning(f"BOT: Session limit reached: {e}")
                message_parts.append(("error", str(e)))
                await update_bot_ui("⏳ **Session limit reached**", force=True)
                return
            except Exception as e:
                logger.error(f"BOT: Failed to get/create session: {e}")
                await send_error_to_user(chat_id, str(e), "creating session")
                return

            # Process CLI events
            async for event_data in cli_session.start_task(
                prompt, session_id=captured_session_id
            ):
                if not isinstance(event_data, dict):
                    continue

                # Handle session_info event to capture session ID
                if event_data.get("type") == "session_info":
                    real_session_id = event_data.get("session_id")
                    if real_session_id and temp_session_id:
                        # Register the real session ID
                        await cli_session_manager.register_real_session_id(
                            temp_session_id, real_session_id
                        )
                        captured_session_id = real_session_id
                        # Save to session store for Telegram reply tracking
                        session_store.save_session(
                            session_id=real_session_id,
                            chat_id=chat_id,
                            initial_msg_id=original_msg_id,
                        )
                        logger.info(
                            f"BOT: Registered session {temp_session_id} -> {real_session_id}"
                        )
                    continue

                parsed = CLIParser.parse_event(event_data)

                if event_data.get("type") == "raw":
                    raw_line = event_data.get("content")
                    if not raw_line:
                        continue
                    if "login" in raw_line.lower():
                        await client.send_message(
                            chat_id,
                            "⚠️ **Claude requires login. Run `claude` in terminal.**",
                        )
                    elif "error" in raw_line.lower():
                        message_parts.append(("error", raw_line[:200]))
                    continue

                if not parsed:
                    continue

                if parsed["type"] == "thinking":
                    thinking_text = parsed["text"]
                    message_parts.append(("thinking", thinking_text))
                    await update_bot_ui("🧠 **Claude is thinking...**")

                elif parsed["type"] == "content":
                    if parsed.get("thinking"):
                        thinking_text = parsed["thinking"]
                        logger.debug(f"BOT: Got thinking: {len(thinking_text)} chars")
                        message_parts.append(("thinking", thinking_text))
                    if parsed.get("text"):
                        logger.debug(
                            f"BOT: Got text content: {len(parsed['text'])} chars"
                        )
                        if message_parts and message_parts[-1][0] == "content":
                            prev_type, prev_content = message_parts[-1]
                            message_parts[-1] = (
                                "content",
                                prev_content + parsed["text"],
                            )
                        else:
                            message_parts.append(("content", parsed["text"]))
                        await update_bot_ui("🧠 **Claude is working...**")

                elif parsed["type"] == "tool_start":
                    names = [t.get("name") for t in parsed["tools"]]
                    message_parts.append(("tool", ", ".join(names)))
                    await update_bot_ui("⏳ **Executing tools...**")

                elif parsed["type"] == "subagent_start":
                    tasks = parsed["tasks"]
                    message_parts.append(("subagent", ", ".join(tasks)))
                    await update_bot_ui("🔎 **Subagent working...**")

                elif parsed["type"] == "complete":
                    logger.debug(
                        f"BOT: Complete event, parts count: {len(message_parts)}"
                    )
                    if parsed.get("status") == "failed":
                        await update_bot_ui("❌ **Failed**", force=True)
                    else:
                        if not message_parts:
                            message_parts.append(("content", "Done."))
                        await update_bot_ui("✅ **Complete**", force=True)

                    # Update session's last message so replies to THIS response also work
                    if captured_session_id and status_msg:
                        session_store.update_last_message(
                            captured_session_id, status_msg.id
                        )

                elif parsed["type"] == "error":
                    error_msg = parsed.get("message", "Unknown error")
                    if "timeout" in error_msg.lower():
                        message_parts.append(("error", f"⏱️ {error_msg}"))
                    else:
                        message_parts.append(("error", f"**CLI Error:** {error_msg}"))
                    await update_bot_ui("❌ **Error**", force=True)

        except asyncio.CancelledError:
            logger.info(
                f"BOT: Task cancelled for session {captured_session_id or temp_session_id}"
            )
            message_parts.append(("error", "Task was cancelled"))
            await update_bot_ui("❌ **Failed**", force=True)
        except Exception as e:
            import traceback

            logger.error(f"Bot task failed: {e}\n{traceback.format_exc()}")
            try:
                error_text = str(e)[:300]
                await status_msg.edit(
                    f"💥 **Task Failed**\n\n```\n{error_text}\n```",
                    parse_mode="markdown",
                )
            except:
                await send_error_to_user(chat_id, str(e), "task execution")

    @client.on(events.NewMessage())
    async def handle_telegram_message(event):
        sender_id = str(event.sender_id)
        text_preview = event.text[:50] if event.text else "(empty)"
        logger.info(f"BOT_EVENT: From {sender_id} | Text: {text_preview}")

        target_id = str(ALLOWED_USER_ID).strip()
        if sender_id != target_id:
            logger.debug(f"BOT_SECURITY: Ignored message from {sender_id}")
            return

        # 1. Handle Commands
        if event.text == "/stop":
            # 1. Cancel all queued and running messages in the message queue
            cancelled_msgs = await message_queue.cancel_all()

            # 2. Stop all CLI sessions (subprocesses)
            await cli_session_manager.stop_all()

            # 3. Inform the user and update status of cancelled messages
            await event.reply("⏹ **All Claude sessions stopped.**")

            for msg in cancelled_msgs:
                try:
                    # We might not be able to edit if it's too old or already done,
                    # but we try to mark them as failed as requested.
                    status_msg = await client.get_messages(
                        msg.chat_id, ids=msg.reply_msg_id
                    )
                    if status_msg:
                        await status_msg.edit(
                            "❌ **Failed** (Stopped by user)", parse_mode="markdown"
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not update status for cancelled msg {msg.msg_id}: {e}"
                    )
            return

        if event.text == "/stats":
            stats = cli_session_manager.get_stats()
            await event.reply(
                f"📊 **Session Stats**\n\n"
                f"• Active: {stats['active_sessions']}\n"
                f"• Pending: {stats['pending_sessions']}\n"
                f"• Busy: {stats['busy_count']}\n"
                f"• Max: {stats['max_sessions']}"
            )
            return

        if event.text == "/queue":
            stats = cli_session_manager.get_stats()
            await event.reply(
                f"📋 **Queue Status**\n\n"
                f"Active sessions: {stats['active_sessions']}/{stats['max_sessions']}\n"
                f"Reply to old messages to continue conversations."
            )
            return

        # 2. Filter out bot's own status messages and empty text
        if not event.text or any(
            event.text.startswith(p)
            for p in ["⏳", "💭", "🔧", "✅", "❌", "🚀", "🤖", "📋", "📊", "🔄"]
        ):
            return

        logger.info(f"BOT_TASK: {event.text}")

        # 3. Check if this is a reply to an existing conversation
        session_id_to_resume = None
        reply_to_msg_id = event.reply_to_msg_id

        if reply_to_msg_id:
            # User is replying to a previous message - try to find the session
            session_id_to_resume = session_store.get_session_by_msg(
                event.chat_id, reply_to_msg_id
            )
            if session_id_to_resume:
                logger.info(
                    f"BOT: Found session {session_id_to_resume} for reply to msg {reply_to_msg_id}"
                )
            else:
                logger.info(
                    f"BOT: No session found for reply to msg {reply_to_msg_id}, starting new session"
                )

        # 4. Send initial status message
        try:
            if session_id_to_resume:
                if message_queue.is_session_busy(session_id_to_resume):
                    queue_size = message_queue.get_queue_size(session_id_to_resume) + 1
                    status_msg = await event.reply(
                        f"📋 **Queued** (position {queue_size}) - waiting for previous request..."
                    )
                else:
                    status_msg = await event.reply("🔄 **Continuing conversation...**")
            else:
                stats = cli_session_manager.get_stats()
                if stats["active_sessions"] >= stats["max_sessions"]:
                    status_msg = await event.reply(
                        f"⏳ **Waiting for slot...** ({stats['active_sessions']}/{stats['max_sessions']} sessions active)"
                    )
                else:
                    status_msg = await event.reply(
                        "⏳ **Launching new Claude CLI instance...**"
                    )
        except Exception as e:
            logger.error(f"Failed to send status message: {e}")
            return

        # 5. Create queued message
        queued_msg = QueuedMessage(
            prompt=event.text,
            chat_id=event.chat_id,
            msg_id=event.id,
            reply_msg_id=status_msg.id,
            event=event,
        )

        # 6. Process or queue based on session state
        if session_id_to_resume and message_queue.is_session_busy(session_id_to_resume):
            # Session is busy, queue the message for that specific session
            await message_queue.enqueue(
                session_id=session_id_to_resume,
                message=queued_msg,
                processor=process_claude_task,
            )
            logger.info(f"BOT: Message queued for busy session {session_id_to_resume}")
        elif session_id_to_resume:
            # Resuming a free existing session - use queue to track busy state
            await message_queue.enqueue(
                session_id=session_id_to_resume,
                message=queued_msg,
                processor=process_claude_task,
            )
        else:
            # NEW session - create a temporary ID based on the trigger message
            temp_session_id = f"pending_{event.id}"
            logger.info(f"BOT: Starting NEW session {temp_session_id}")

            # Pre-register in session store so replies to this NEW message or its status
            # can be identified and enqueued immediately.
            session_store.save_session(
                session_id=temp_session_id,
                chat_id=event.chat_id,
                initial_msg_id=event.id,
            )
            session_store.update_last_message(temp_session_id, status_msg.id)

            # Process via queue to ensure we track busy state even for new sessions
            await message_queue.enqueue(
                session_id=temp_session_id,
                message=queued_msg,
                processor=process_claude_task,
            )


FAST_PREFIX_DETECTION = os.getenv("FAST_PREFIX_DETECTION", "true").lower() == "true"


app = FastAPI(title="Claude Code Proxy", version="2.0.0", lifespan=lifespan)


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    """Handle provider-specific errors and return Anthropic format."""
    logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_anthropic_format(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle general errors and return Anthropic format."""
    logger.error(f"General Error: {str(exc)}")
    import traceback

    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "An unexpected error occurred.",
            },
        },
    )


def extract_command_prefix(command: str) -> str:
    import shlex

    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command)
        if not parts:
            return "none"

        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if "=" in part and not part.startswith("-"):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {
            "git",
            "npm",
            "docker",
            "kubectl",
            "cargo",
            "go",
            "pip",
            "yarn",
        }

        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            if not second_word.startswith("-"):
                return f"{first_word} {second_word}"
            return first_word
        return first_word if not env_prefix else " ".join(env_prefix) + " " + first_word

    except ValueError:
        return command.split()[0] if command.split() else "none"


def is_prefix_detection_request(request_data: MessagesRequest) -> tuple[bool, str]:
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, ""

    msg = request_data.messages[0]
    content = ""
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            if hasattr(block, "text"):
                content += block.text

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except Exception:
            pass

    return False, ""


def get_token_count(messages, system=None, tools=None) -> int:
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += len(ENCODER.encode(block.text))

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                # Handle dictionary or Pydantic model
                b_type = getattr(block, "type", None)

                if b_type == "text":
                    total_tokens += len(ENCODER.encode(getattr(block, "text", "")))
                elif b_type == "thinking":
                    # Thinking tokens are part of context if they are in history
                    total_tokens += len(ENCODER.encode(getattr(block, "thinking", "")))
                elif b_type == "tool_use":
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {})
                    # Add tokens for definitions
                    total_tokens += len(ENCODER.encode(name))
                    total_tokens += len(ENCODER.encode(json.dumps(inp)))
                    total_tokens += 10  # Control tokens approximate
                elif b_type == "tool_result":
                    content = getattr(block, "content", "")
                    if isinstance(content, str):
                        total_tokens += len(ENCODER.encode(content))
                    else:
                        total_tokens += len(ENCODER.encode(json.dumps(content)))
                    total_tokens += 5  # Control tokens approximate

    if tools:
        for tool in tools:
            # Approximate tool definition tokens
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(ENCODER.encode(tool_str))

    # Add some overhead for message formatting (approx 3 tokens per message)
    total_tokens += len(messages) * 3
    if tools:
        total_tokens += len(tools) * 5  # Extra overhead for tool definitions

    return max(1, total_tokens)


# log_request_details removed - now using log_request_compact from logging_utils


@app.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    provider: NvidiaNimProvider = Depends(get_provider),
):
    try:
        if FAST_PREFIX_DETECTION:
            is_prefix_req, command = is_prefix_detection_request(request_data)
            if is_prefix_req:
                import uuid

                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=request_data.model,
                    content=[{"type": "text", "text": extract_command_prefix(command)}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=5),
                )

        import uuid as uuid_mod

        request_id = f"req_{uuid_mod.uuid4().hex[:12]}"
        log_request_compact(logger, request_id, request_data)

        if request_data.stream:
            input_tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            return StreamingResponse(
                provider.stream_response(request_data, input_tokens=input_tokens),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response_json = await provider.complete(request_data)
            return provider.convert_response(response_json, request_data)

    except ProviderError:
        # Re-raise ProviderError to be handled by the specialized exception handler
        raise
    except Exception as e:
        import traceback

        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    try:
        return TokenCountResponse(
            input_tokens=get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "status": "ok",
        "provider": "nvidia_nim",
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "bot_running": tele_client is not None}


@app.post("/stop")
async def stop_cli():
    await cli_session_manager.stop_all()
    return {"status": "terminated"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="debug")
