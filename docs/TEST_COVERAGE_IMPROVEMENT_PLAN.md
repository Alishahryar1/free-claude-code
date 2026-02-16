# Test Coverage Improvement Plan

**Generated:** February 16, 2025  
**Current baseline:** 91% overall coverage (724 tests)  
**Tool:** pytest with pytest-cov

---

## Executive Summary

The codebase has strong overall coverage (91%), but several modules have significant gaps. This plan prioritizes improvements by impact and effort, focusing on the lowest-coverage modules first.

---

## Priority 1: Critical Gaps (7–52% coverage)

### 1.1 `messaging/discord_markdown.py` — **7% coverage** (269 lines missed)

**Impact:** High — Discord markdown rendering is user-facing and error-prone.

**Missing coverage:**
- `_is_gfm_table_header_line`, `_normalize_gfm_tables`
- `escape_discord`, `escape_discord_code`, `discord_bold`, `discord_code_inline`
- `format_status_discord`, `format_status`
- `render_markdown_to_discord` (main entry point)

**Actions:**
1. Add `tests/test_discord_markdown.py` mirroring `test_handler_markdown_and_status_edges.py` patterns for Telegram.
2. Test table detection, normalization, and code-block fallback.
3. Test escaping (backticks, backslashes, bold, inline code).
4. Test `format_status` variants.
5. Parametrize `render_markdown_to_discord` for common markdown structures (headers, lists, tables, code blocks).

---

### 1.2 `messaging/discord.py` — **40% coverage** (122 lines missed)

**Impact:** High — Discord platform integration.

**Missing coverage:**
- `_get_discord()` lazy import
- `_parse_allowed_channels` edge cases
- Message send/edit/delete flows
- Queue operations
- Event handlers (`on_message`, `on_ready`, etc.)
- Error handling paths

**Actions:**
1. Expand `tests/test_discord_platform.py` with mocked `discord.Client` flows.
2. Add tests for `_parse_allowed_channels` (empty, comma-separated, whitespace).
3. Add integration-style tests for send/edit/delete with mocked HTTP.
4. Test `fire_and_forget` and queue behavior.
5. Test channel allowlist filtering.

---

### 1.3 `providers/open_router/client.py` — **52% coverage** (84 lines missed)

**Impact:** Medium–High — OpenRouter is an alternative provider.

**Missing coverage:**
- Streaming response handling (lines 154–173, 180–205)
- Error mapping and retry logic
- Non-streaming paths
- Model mapping and request transformation

**Actions:**
1. Add `tests/test_open_router_client.py` (or extend `test_open_router.py`).
2. Mock `httpx.AsyncClient` for streaming and non-streaming responses.
3. Test error handling (rate limits, 5xx, timeouts).
4. Test model mapping and request building.
5. Align with patterns used in `test_nvidia_nim.py` and `test_lmstudio.py`.

---

## Priority 2: Moderate Gaps (68–85% coverage)

### 2.1 `messaging/telegram.py` — **68% coverage**

**Actions:**
- Cover `fire_and_forget` non-coroutine path (partially in `test_telegram_edge_cases`).
- Test `_with_retry` edge cases (message not found, parse errors).
- Test queue send/edit when limiter is present vs absent.
- Test `on_start` command and handler error propagation.
- Test message length limits (4096) and empty string handling (partially covered).

---

### 2.2 `messaging/base.py` — **77% coverage**

**Missing:** Protocol/ABC methods not exercised by concrete implementations.

**Actions:**
- Add tests that explicitly call protocol methods via `CLISession`, `SessionManagerInterface`, `MessagingPlatform` mocks.
- Ensure `mock_platform` in conftest exercises all interface methods.

---

### 2.3 `messaging/transcript.py` — **78% coverage**

**Missing:** Lines 272–340 (segment rendering), 372–380, 430–470, 496–508.

**Actions:**
- Add tests for `ThinkingSegment`, `ToolCallSegment`, `ToolResultSegment`, `SubagentSegment`, `ErrorSegment` edge cases.
- Test truncation with mixed segment types.
- Test `RenderCtx` state transitions.
- Test exception handling in `render_segment`.

---

### 2.4 `messaging/handler.py` — **85% coverage**

**Missing:** Lines 575–601, 827–828, 838, 849–850, 867, 877–884, 893–895, 902–903.

**Actions:**
- Identify and test the specific branches (likely error paths, cancellation, or edge conditions).
- Add tests for `process_node` cancellation and cleanup.
- Test handler behavior when platform methods raise.

---

### 2.5 `messaging/limiter.py` — **85% coverage**

**Actions:**
- Test `SlidingWindowLimiter` edge cases (window boundaries, concurrent requests).
- Test `MessagingRateLimiter` queue overflow and rejection paths.
- Test retry-after and backoff behavior.

---

### 2.6 `providers/nvidia_nim/client.py` & `providers/lmstudio/client.py` — **80% each**

**Missing:** Streaming chunks (144–163, 202–231), error handling (241, 257–258, 285–296).

**Actions:**
- Add streaming response tests with chunked mock data.
- Test 5xx, timeout, and connection error handling.
- Test partial stream failure and recovery.

---

### 2.7 `api/optimization_handlers.py` — **81% coverage**

**Missing:** Settings-gated branches (when features disabled), logger calls.

**Actions:**
- Add tests with `enable_*` settings toggled off to ensure handlers return `None`.
- Test `try_optimizations` short-circuit order.
- Optionally assert logger calls for optimization hits.

---

### 2.8 `cli/process_registry.py` — **84% coverage**

**Missing:** Lines 34, 42, 71–75 (likely `kill_all_best_effort`, signal handling).

**Actions:**
- Test `kill_all_best_effort` with mock processes.
- Test `register_pid`/`unregister_pid` edge cases.
- Test graceful vs forceful shutdown paths.

---

## Priority 3: Smaller Gaps (88–99% coverage)

### 3.1 Modules at 88–96%

| Module | Coverage | Focus |
|--------|----------|--------|
| `api/app.py` | 89% | Lifespan error paths, provider fallback |
| `config/logging_config.py` | 88% | Log level configuration, handler setup |
| `api/detection.py` | 91% | Exception paths in `is_prefix_detection_request`, `is_filepath_extraction_request` |
| `api/command_utils.py` | 92% | `ValueError` in `extract_command_prefix`, grep edge cases |
| `messaging/event_parser.py` | 92% | Malformed event handling |
| `messaging/tree_processor.py` | 92% | Error and cancellation paths |
| `messaging/tree_queue.py` | 93% | Queue overflow, concurrency edge cases |
| `api/routes.py` | 94% | Lines 86–95 (likely error or edge route) |
| `messaging/tree_repository.py` | 96% | Save/load error paths |
| `providers/nvidia_nim/utils/think_parser.py` | 96% | Flush and boundary cases |

**Actions:** Add targeted tests for the specific missing line ranges listed in the coverage report.

---

## CI Integration

### Add coverage to CI

1. **Update `.github/workflows/tests.yml`:**

```yaml
- name: Run tests
  run: uv run pytest -v --tb=short --cov=. --cov-report=xml --cov-fail-under=85
```

2. **Optional:** Add `[tool.coverage.run]` and `[tool.coverage.report]` to `pyproject.toml` for consistent config:

```toml
[tool.coverage.run]
source = ["api", "cli", "config", "messaging", "providers", "utils"]
omit = ["tests/*", "*.pyi"]

[tool.coverage.report]
fail_under = 85
exclude_lines = ["pragma: no cover", "def __repr__", "raise NotImplementedError"]
```

3. **Optional:** Add a coverage badge or report upload (e.g., Codecov, Coveralls).

---

## Implementation Order

| Phase | Scope | Estimated effort |
|-------|--------|------------------|
| **Phase 1** | Discord markdown (7% → 85%+) | 1–2 days |
| **Phase 2** | Discord platform (40% → 75%+) | 1–2 days |
| **Phase 3** | OpenRouter client (52% → 80%+) | 1 day |
| **Phase 4** | Telegram, transcript, handler (68–85% → 90%+) | 2 days |
| **Phase 5** | Provider clients (NIM, LMStudio), limiter, process_registry | 1–2 days |
| **Phase 6** | Remaining modules, CI integration | 1 day |

---

## Quick Wins

1. **Add `--cov` to CI** — Enforce coverage on every PR.
2. **`test_discord_markdown.py`** — Copy structure from Telegram markdown tests; high impact for relatively low effort.
3. **Targeted line coverage** — For modules at 90%+, add 1–3 tests per missing block to reach 95%+.

---

## Notes

- Use `uv run pytest --cov=. --cov-report=term-missing` for local development.
- Exclude `tests/` and `conftest.py` from coverage targets if desired (or keep for test quality metrics).
- Consider `pytest-cov`’s `--cov-branch` for branch coverage in critical paths.
