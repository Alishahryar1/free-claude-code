# Architecture Improvement Plan

## Overview

This plan addresses 7 categories of improvements found during the code review:
provider code duplication, mislocated shared utilities, encapsulation leaks,
dead code, performance issues, directory structure, and minor fixes.

Changes are ordered by dependency: foundational moves first, then refactors
that build on them, then independent cleanup.

**CI requirements**: Every step must pass all 5 CI checks:
1. No `# type: ignore` / `# ty: ignore`
2. `uv run ruff format`
3. `uv run ruff check`
4. `uv run ty check`
5. `uv run pytest`

---

## Phase 1: Extract shared provider utilities into `providers/common/`

**Goal**: Eliminate the coupling where OpenRouter and LMStudio import from
`providers.nvidia_nim.utils` and `providers.nvidia_nim.errors`.

### Step 1.1: Create `providers/common/` package

Move these files from `providers/nvidia_nim/utils/` → `providers/common/`:
- `sse_builder.py` → `providers/common/sse_builder.py`
- `message_converter.py` → `providers/common/message_converter.py`
- `think_parser.py` → `providers/common/think_parser.py`
- `heuristic_tool_parser.py` → `providers/common/heuristic_tool_parser.py`

Move from `providers/nvidia_nim/`:
- `errors.py` → `providers/common/error_mapping.py`

Create `providers/common/__init__.py` with the same re-exports that
`providers/nvidia_nim/utils/__init__.py` currently has, plus `map_error`.

### Step 1.2: Update `providers/nvidia_nim/utils/__init__.py`

Change it to re-export from `providers.common` for backward compatibility:
```python
from providers.common import (
    SSEBuilder, ContentBlockManager, map_stop_reason,
    ThinkTagParser, ContentType, ContentChunk,
    HeuristicToolParser,
    AnthropicToOpenAIConverter, get_block_attr, get_block_type,
)
```

Similarly update `providers/nvidia_nim/errors.py` to re-export:
```python
from providers.common.error_mapping import map_error
```

### Step 1.3: Update direct consumers to import from `providers.common`

**Source files** (change imports):
- `providers/open_router/client.py` (lines 13-19) → import from `providers.common`
- `providers/lmstudio/client.py` (lines 13-19) → import from `providers.common`
- `providers/open_router/request.py` (line 5) → import from `providers.common.message_converter`
- `providers/lmstudio/request.py` (line 5) → import from `providers.common.message_converter`
- `providers/nvidia_nim/client.py` (lines 14-21, relative imports `from .errors` and `from .utils`) → import from `providers.common`
- `providers/nvidia_nim/request.py` (line 6, `from .utils.message_converter`) → import from `providers.common.message_converter`

**Test files** (change imports):
- `tests/test_sse_builder.py` (line 7)
- `tests/test_lmstudio.py` (inline imports at ~8 locations)
- `tests/test_subagent_interception.py` (line 5)
- `tests/test_parsers.py` (lines 3-4)
- `tests/test_streaming_errors.py` (inline imports at ~8 locations)
- `tests/test_converter.py` (lines 3, 276)
- `tests/test_error_mapping.py` (line 9)

### Step 1.4: Verify

- `uv run pytest` — all tests pass
- `uv run ty check` — no type errors
- `uv run ruff check && uv run ruff format`

---

## Phase 2: Extract shared streaming base class (`OpenAICompatibleProvider`)

**Goal**: Eliminate ~400 lines of duplicated streaming logic across the 3
provider clients.

### Step 2.1: Create `providers/openai_compat.py`

Create `OpenAICompatibleProvider(BaseProvider)` that contains the shared logic:

```python
class OpenAICompatibleProvider(BaseProvider):
    _client: AsyncOpenAI
    _global_rate_limiter: GlobalRateLimiter
    _provider_name: str  # "NIM", "OPENROUTER", "LMSTUDIO" — used in log tags

    def __init__(self, config, *, provider_name, base_url, api_key, nim_settings=None):
        # shared __init__: create AsyncOpenAI client, rate limiter

    def _build_request_body(self, request) -> dict:
        raise NotImplementedError  # each provider implements

    async def stream_response(self, request, input_tokens, *, request_id):
        # shared: logger.contextualize + delegate to _stream_response_impl

    async def _stream_response_impl(self, request, input_tokens, request_id):
        # THE shared ~180-line streaming loop, currently duplicated 3x

    def _handle_extra_reasoning(self, delta, sse) -> Iterator[str]:
        """Hook for OpenRouter's reasoning_details. Default: no-op."""
        return iter(())

    def _process_tool_call(self, tc, sse):
        # shared ~40-line method

    def _flush_task_arg_buffers(self, sse):
        # shared 3-line method
```

### Step 2.2: Refactor `NvidiaNimProvider`

Reduce to:
```python
class NvidiaNimProvider(OpenAICompatibleProvider):
    def __init__(self, config):
        super().__init__(config, provider_name="NIM",
                         base_url=config.base_url or NVIDIA_NIM_BASE_URL,
                         api_key=config.api_key,
                         nim_settings=config.nim_settings)

    def _build_request_body(self, request):
        return build_request_body(request, self._nim_settings)
```

### Step 2.3: Refactor `OpenRouterProvider`

Reduce to:
```python
class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, config):
        super().__init__(config, provider_name="OPENROUTER",
                         base_url=config.base_url or OPENROUTER_BASE_URL,
                         api_key=config.api_key)

    def _build_request_body(self, request):
        return build_request_body(request)

    def _handle_extra_reasoning(self, delta, sse):
        # Handle reasoning_details for StepFun models (8 lines)
        ...
```

### Step 2.4: Refactor `LMStudioProvider`

Reduce to:
```python
class LMStudioProvider(OpenAICompatibleProvider):
    def __init__(self, config):
        super().__init__(config, provider_name="LMSTUDIO",
                         base_url=config.base_url or LMSTUDIO_DEFAULT_BASE_URL,
                         api_key=config.api_key or "lm-studio")

    def _build_request_body(self, request):
        return build_request_body(request)
```

### Step 2.5: Verify

- All existing tests must pass without modification (public interface unchanged)
- `uv run pytest && uv run ty check && uv run ruff format && uv run ruff check`

---

## Phase 3: Fix encapsulation violations

### Step 3.1: Add `MessageTree.set_current_task(task)` method

In `messaging/tree_data.py`, add:
```python
def set_current_task(self, task: Optional[asyncio.Task]) -> None:
    """Set the current processing task. Caller must hold lock."""
    self._current_task = task
```

Update `messaging/tree_processor.py` lines 117 and 155:
```python
# Before:  tree._current_task = asyncio.create_task(...)
# After:   tree.set_current_task(asyncio.create_task(...))
```

### Step 3.2: Move `nim_settings` out of `ProviderConfig` base

In `providers/base.py`, remove `nim_settings` from `ProviderConfig`.

Add it as a field in `NvidiaNimProvider.__init__` or pass it directly
in the provider-specific config. The `OpenAICompatibleProvider` base class
stores it as `Optional[NimSettings]` only if passed.

Update `api/dependencies.py` where `ProviderConfig` is constructed — only
pass `nim_settings` for the NIM provider.

### Step 3.3: Verify

- `uv run pytest && uv run ty check`

---

## Phase 4: Remove dead code

### Step 4.1: Remove legacy `SessionRecord` system

In `messaging/session.py`:
- Remove `SessionRecord` dataclass (lines 18-28)
- Remove `self._sessions` dict and `self._msg_to_session` dict (lines 41-44)
- Remove `self._make_key()` method (lines 56-58) — note: keep `_make_chat_key()` which is still used
- Remove legacy session loading from `_load()` (lines 73-89) — keep tree
  and message_log loading
- Remove `self._sessions` from `_save()` serialization (line 138)
- Remove `self._sessions.clear()` and `self._msg_to_session.clear()` from
  `clear_all()` (lines 247-248)
- Remove the unused `import` for `dataclasses.asdict` if no longer needed
  (currently used only to serialize `SessionRecord`)

### Step 4.2: Fix hardcoded provider in root endpoint

In `api/routes.py:102`:
```python
# Before:  "provider": "nvidia_nim",
# After:   "provider": settings.provider_type,
```

### Step 4.3: Verify

- `uv run pytest && uv run ty check`

---

## Phase 5: Performance improvements

### Step 5.1: Use list-based string accumulation in transcript segments

In `messaging/transcript.py`:

**`ThinkingSegment`** — change from `self.text += t` to list accumulation:
```python
def __init__(self):
    super().__init__(kind="thinking")
    self._parts: list[str] = []

def append(self, t: str) -> None:
    if t:
        self._parts.append(t)

@property
def text(self) -> str:
    return "".join(self._parts)
```

Do the same for **`TextSegment`**.

For **`ToolCallSegment.append_input_delta`** — same pattern. Also update
`set_initial_input()` to do `self._parts = [inp]` instead of
`self.input_text = inp`.

Update `render()` methods and any test that accesses `.text` or
`.input_text` directly to use the property.

### Step 5.2: Cache `MAX_MESSAGE_LOG_ENTRIES_PER_CHAT` at init time

In `messaging/session.py`, `SessionStore.__init__`:
```python
cap_raw = os.getenv("MAX_MESSAGE_LOG_ENTRIES_PER_CHAT", "").strip()
self._message_log_cap: int | None = int(cap_raw) if cap_raw else None
```

Replace the per-call `os.getenv()` in `record_message_id()` (lines 215-229)
with `self._message_log_cap`.

### Step 5.3: Use iterative DFS in `MessageTree.get_descendants`

In `messaging/tree_data.py`, replace the recursive implementation:
```python
def get_descendants(self, node_id: str) -> list[str]:
    if node_id not in self._nodes:
        return []
    result = []
    stack = [node_id]
    while stack:
        nid = stack.pop()
        result.append(nid)
        node = self._nodes.get(nid)
        if node:
            stack.extend(node.children_ids)
    return result
```

### Step 5.4: Verify

- `uv run pytest` — all tests pass (especially transcript and tree tests)
- `uv run ty check`

---

## Phase 6: Minor fixes and cleanup

### Step 6.1: Remove `if False: yield ""` hack in `BaseProvider`

In `providers/base.py`, replace the abstract method body:
```python
@abstractmethod
async def stream_response(self, ...) -> AsyncIterator[str]:
    """Stream response in Anthropic SSE format."""
    ...
```

Note: This requires verifying that ty/mypy accepts `...` as a valid body
for an abstract async generator. If not, keep a minimal workaround but
add a comment explaining why.

### Step 6.2: Clean up `messaging/handler.py` log message naming

Lines 482, 491, 495, 497 and 507 say `TELEGRAM_EDIT` but the handler is
platform-agnostic. Rename to `PLATFORM_EDIT`:
```python
# line 482: TELEGRAM_EDIT → PLATFORM_EDIT
# line 491: TELEGRAM_EDIT_TEXT → PLATFORM_EDIT_TEXT
# line 495: TELEGRAM_EDIT_PREVIEW_HEAD → PLATFORM_EDIT_PREVIEW_HEAD
# line 497: TELEGRAM_EDIT_PREVIEW_TAIL → PLATFORM_EDIT_PREVIEW_TAIL
# line 507: Failed to update Telegram → Failed to update platform
```

### Step 6.3: Verify

- `uv run ruff format && uv run ruff check && uv run ty check && uv run pytest`

---

## Phase 7: Directory restructuring (messaging/ and tests/)

**Note**: This phase has the highest risk of merge conflicts. It should be
done last and in one commit to minimize churn.

### Step 7.1: Create `messaging/platforms/` sub-package

Move:
- `messaging/base.py` → `messaging/platforms/base.py`
- `messaging/discord.py` → `messaging/platforms/discord.py`
- `messaging/telegram.py` → `messaging/platforms/telegram.py`
- `messaging/factory.py` → `messaging/platforms/factory.py`

Create `messaging/platforms/__init__.py` re-exporting key symbols.
Update `messaging/__init__.py` to import from `messaging.platforms`.

### Step 7.2: Create `messaging/rendering/` sub-package

Move:
- `messaging/discord_markdown.py` → `messaging/rendering/discord_markdown.py`
- `messaging/telegram_markdown.py` → `messaging/rendering/telegram_markdown.py`

Create `messaging/rendering/__init__.py`.
Update `messaging/handler.py` imports.

### Step 7.3: Create `messaging/trees/` sub-package

Move:
- `messaging/tree_data.py` → `messaging/trees/data.py`
- `messaging/tree_repository.py` → `messaging/trees/repository.py`
- `messaging/tree_processor.py` → `messaging/trees/processor.py`
- `messaging/tree_queue.py` → `messaging/trees/queue_manager.py`

Create `messaging/trees/__init__.py` re-exporting `TreeQueueManager`,
`MessageTree`, `MessageNode`, `MessageState`.

Update `messaging/__init__.py` re-exports.

### Step 7.4: Organize `tests/` to mirror source

Create subdirectories:
```
tests/
  api/         ← test_api.py, test_routes_optimizations.py, test_app_lifespan_and_errors.py, etc.
  providers/   ← test_nvidia_nim.py, test_open_router.py, test_lmstudio.py, etc.
  messaging/   ← test_handler.py, test_tree_*.py, test_telegram.py, test_discord_*.py, etc.
  cli/         ← test_cli.py, test_cli_manager_edge_cases.py, test_process_registry.py
  config/      ← test_config.py, test_logging_config.py
```

Update `conftest.py` path if needed. Ensure pytest discovers all tests.

### Step 7.5: Maintain backward-compatible re-exports

Every moved module must have re-exports from the old location (via the
package `__init__.py`) so that any external consumer or existing import
path continues to work. These re-exports can be removed in a future
breaking version.

### Step 7.6: Verify

- `uv run pytest` — all 56+ test files discovered and passing
- `uv run ty check` — no broken imports
- `uv run ruff check && uv run ruff format`

---

## Execution Order & Dependencies

```
Phase 1 (shared utils extraction)
  └→ Phase 2 (shared base class) — depends on Phase 1
       └→ Phase 3 (encapsulation) — depends on Phase 2 for nim_settings
Phase 4 (dead code) — independent
Phase 5 (performance) — independent
Phase 6 (minor fixes) — independent
Phase 7 (directory restructure) — should be done LAST
```

Phases 4, 5, 6 are independent of each other and of Phase 2. They can be
done in any order or in parallel.

Phase 7 must come after all other phases to avoid rebasing moved files.

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|-----------|
| 1 | Import breakage in tests | Backward-compat re-exports in old location |
| 2 | Behavioral change in streaming | Tests cover all 3 providers; run full suite |
| 3 | `nim_settings` removal from base config | Check all `ProviderConfig` construction sites |
| 4 | Legacy session data stops loading | Only remove write path; keep read if needed |
| 5 | String accumulation changes rendering | Transcript tests exercise rendering thoroughly |
| 7 | Massive import churn, merge conflicts | Do in single commit, last phase |

---

## Estimated Scope

| Phase | Files Changed | Lines Changed (approx) |
|-------|--------------|----------------------|
| 1 | ~20 | +80 / -20 (new __init__ + import updates) |
| 2 | ~5 | +200 / -450 (net reduction ~250 lines) |
| 3 | ~4 | +15 / -10 |
| 4 | ~2 | +5 / -60 |
| 5 | ~3 | +30 / -15 |
| 6 | ~2 | +5 / -5 |
| 7 | ~60+ | +100 / -50 (mostly import changes) |
| **Total** | | **Net reduction ~200-300 lines** |
