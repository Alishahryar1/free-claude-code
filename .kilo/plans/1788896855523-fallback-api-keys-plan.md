# Fallback API Keys Implementation Plan

## Overview
Add support for multiple API keys per provider with automatic fallback when a key fails (authentication error, rate limit, timeout, etc.). This complements the existing model-level fallbacks (`MODEL_FALLBACKS`) by adding key-level fallbacks within the same provider.

## Current Architecture
- **Settings** (`config/settings.py`): Single API key per provider (e.g., `nvidia_nim_api_key`)
- **Provider Catalog** (`config/provider_catalog.py`): Maps env vars to settings attributes
- **Provider Config** (`providers/runtime/config.py`): Builds `ProviderConfig` with single `api_key`
- **Provider Runtime** (`providers/runtime/runtime.py`): Creates/caches providers per provider ID
- **OpenAIChatProvider** (`providers/openai_chat/provider.py`): Uses single API key for `AsyncOpenAI` client
- **Execution** (`application/execution.py`): Handles model-level fallbacks via `MODEL_FALLBACKS`

## Two-Layer Fallback Architecture (Critical Separation)

The feature adds a **new inner layer** that operates completely independently from the existing outer layer:

| Layer | Location | Scope | Trigger |
|-------|----------|-------|---------|
| **Key-level (NEW)** | `OpenAIChatProvider._create_stream()` | Single provider, multiple keys | 401/403/429/timeout on a key |
| **Model-level (EXISTING)** | `ProviderExecutor._stream_candidates()` | Multiple providers/models | Provider raises `ExecutionFailure` after ALL keys exhausted |

**Flow:**
```
Request → ProviderExecutor._stream_candidates()
  → Candidate 1: provider A/model X
    → OpenAIChatProvider.stream_messages()  ← KEY FALLBACK HERE (key1→key2→key3)
      → If all keys fail → raise ExecutionFailure
  → Candidate 2: provider B/model Y  ← MODEL FALLBACK HERE
    → OpenAIChatProvider.stream_messages()  ← KEY FALLBACK HERE
      → ...
```

**Key guarantee:** Key rotation never crosses provider boundaries. Model fallback only activates after a provider has exhausted ALL its keys.

## Implementation Strategy

### 1. Configuration Layer (Settings)
**Files:** `config/settings.py`, `config/constants.py`, `.env.example`

- Add support for comma-separated API keys in existing env vars (backward compatible)
- New pattern: `NVIDIA_NIM_API_KEYS=key1,key2,key3` (plural form)
- Keep existing `NVIDIA_NIM_API_KEY` for single key (takes precedence if both set)
- Parse and validate as `tuple[str, ...]` in Settings model
- Update `_validate_model_format` equivalent for API keys

### 2. Provider Catalog & Config
**Files:** `config/provider_catalog.py`, `providers/runtime/config.py`

- No catalog changes needed (env var names stay same)
- Modify `provider_credential()` to return `tuple[str, ...]` instead of `str | None`
- Update `ProviderConfig` dataclass to hold `api_keys: tuple[str, ...]` (replace `api_key: str | None`)
- Update `build_provider_config()` to pass all keys

### 3. Provider Runtime
**Files:** `providers/runtime/runtime.py`, `providers/runtime/factory.py`

- Update `ProviderConstructor` type hint to accept `api_keys: tuple[str, ...]`
- Pass keys to provider factory functions
- Special providers (NVIDIA NIM, OpenRouter, etc.) need updated factory signatures

### 4. OpenAIChatProvider (Core Implementation)
**Files:** `providers/openai_chat/provider.py`

Key changes:
- Accept `api_keys: tuple[str, ...]` in constructor
- Store current key index: `self._key_index = 0`
- Create `AsyncOpenAI` client with current key
- On authentication failure (401), rate limit (429), or other retryable errors:
  - Increment key index
  - If more keys available, create new client with next key and retry
  - If no more keys, propagate failure
- Add `_rotate_api_key()` method to handle key switching
- Ensure thread-safety for concurrent requests (use asyncio locks)

### 5. Special Provider Factories
**Files:** `providers/runtime/factory.py`

Update each special provider factory to:
- Accept `api_keys` parameter
- Pass to provider constructor
- Providers: NVIDIA NIM, OpenRouter, Mistral, Kilo, DeepSeek, LM Studio, Cloudflare, Gemini, Vertex, Groq, OpenCode Zen/Go

### 6. Admin UI
**Files:** `config/admin/provider_manifest.py`, `config/admin/specs.py`

- Update credential field type from "secret" to "textarea" for multi-key input
- Add description explaining comma-separated format
- Keep secret=true for masking in UI

### 7. Environment File Example
**Files:** `.env.example`

- Add examples for each provider showing plural form
- Document fallback behavior

## Detailed Task Breakdown

### Phase 1: Configuration & Types (No Breaking Changes)
1. [ ] Update `config/settings.py`:
   - Add `api_keys` field type (tuple of strings) for each provider
   - Create validator for comma-separated parsing
   - Maintain backward compatibility with singular `*_api_key`

2. [ ] Update `providers/runtime/config.py`:
   - Change `provider_credential()` return type to `tuple[str, ...]`
   - Update `ProviderConfig` dataclass: `api_keys: tuple[str, ...]`
   - Update `build_provider_config()` to populate all keys

3. [ ] Update `providers/base.py`:
   - Update `ProviderConfig` dataclass definition

### Phase 2: Provider Implementation
4. [ ] Update `providers/openai_chat/provider.py`:
   - Modify `__init__` to accept `api_keys: tuple[str, ...]`
   - Add `_key_index` and `_api_keys` instance variables
   - Implement `_get_current_key()` and `_rotate_api_key()` methods
   - Modify `_create_stream()` to catch auth/rate-limit errors and rotate keys
   - Update client creation to use current key

5. [ ] Update `providers/runtime/factory.py`:
   - Update `ProviderConstructor` type alias
   - Update all special provider factory functions to accept/pass `api_keys`

6. [ ] Update each special provider (NVIDIA NIM, OpenRouter, etc.):
   - Modify `__init__` to accept `api_keys`
   - Pass to parent `OpenAIChatProvider`

### Phase 3: Admin UI & Documentation
7. [ ] Update `config/admin/provider_manifest.py`:
   - Change credential field type to "textarea"
   - Add helpful description for multi-key format

8. [ ] Update `.env.example`:
   - Add examples for plural API key format

### Phase 4: Testing
9. [ ] Add unit tests for:
   - Settings parsing of comma-separated keys
   - Provider config with multiple keys
   - Key rotation on 401/429 errors
   - Exhaustion of all keys

10. [ ] Add integration tests:
    - Mock provider with multiple keys
    - Verify fallback works end-to-end

### Phase 5: Validation
11. [ ] Run existing test suite: `pytest tests/ -x`
12. [ ] Run type checking: `ty check`
13. [ ] Run linting: `ruff check`
14. [ ] Manual verification with multiple keys

## Key Design Decisions

### Key Rotation Trigger Conditions
Rotate to next key on:
- **401 Unauthorized** - Invalid/expired key
- **403 Forbidden** - Key lacks permissions
- **429 Rate Limited** - Key exhausted quota
- **Timeout/Network errors** - Transient failures (optional, configurable)

### Key Exhaustion Behavior
- When all keys for a provider exhausted, propagate the last failure as `ExecutionFailure`
- Log which keys were tried
- **This `ExecutionFailure` is what triggers the existing model-level fallback** (provider B, provider C, etc.)
- Key rotation NEVER falls back to a different provider — that's exclusively `MODEL_FALLBACKS` job

### Concurrency Safety
- Each request gets its own key index tracking
- Use per-request key rotation, not global
- Multiple concurrent requests can use different keys simultaneously

### Backward Compatibility
- Single `*_API_KEY` still works
- If both `*_API_KEY` and `*_API_KEYS` set, prefer `*_API_KEYS`
- No migration needed for existing users

## Affected Files Summary

### Core Changes (Required)
1. `src/free_claude_code/config/settings.py` - Settings model
2. `src/free_claude_code/providers/runtime/config.py` - Provider config building
3. `src/free_claude_code/providers/base.py` - ProviderConfig dataclass
4. `src/free_claude_code/providers/openai_chat/provider.py` - Key rotation logic
5. `src/free_claude_code/providers/runtime/factory.py` - Factory signatures
6. `src/free_claude_code/providers/runtime/runtime.py` - Runtime constructor

### Special Providers (Required)
7. `src/free_claude_code/providers/nvidia_nim/client.py`
8. `src/free_claude_code/providers/open_router/client.py`
9. `src/free_claude_code/providers/mistral/client.py`
10. `src/free_claude_code/providers/kilo/client.py`
11. `src/free_claude_code/providers/deepseek/client.py`
12. `src/free_claude_code/providers/lmstudio/client.py`
13. `src/free_claude_code/providers/cloudflare/client.py`
14. `src/free_claude_code/providers/gemini/client.py`
15. `src/free_claude_code/providers/vertex/client.py`
16. `src/free_claude_code/providers/groq/client.py`
17. `src/free_claude_code/providers/opencode/provider.py`

### Admin & Docs (Optional but Recommended)
18. `src/free_claude_code/config/admin/provider_manifest.py`
19. `src/free_claude_code/config/admin/specs.py`
20. `.env.example`

### Tests (Required for Validation)
21. `tests/providers/test_fallback_keys.py` (new)
22. `tests/config/test_settings.py` (extend)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing single-key configs | Low | High | Backward compatibility maintained |
| Concurrent request key conflicts | Medium | Medium | Per-request key tracking |
| Infinite retry loops | Low | High | Max retries = number of keys |
| Special provider incompatibility | Medium | Medium | Update all factory functions |
| Admin UI validation gaps | Low | Low | Add textarea with clear hints |

## Success Criteria
- [ ] User can configure `NVIDIA_NIM_API_KEYS=key1,key2,key3`
- [ ] First key fails with 401 → second key tried automatically
- [ ] All keys exhausted → proper error propagated
- [ ] Existing single-key configs continue working
- [ ] All existing tests pass
- [ ] New tests cover key rotation scenarios
- [ ] Type checking and linting pass

## Out of Scope
- Per-model API key configuration (different keys for different models)
- Key health monitoring/metrics UI
- Automatic key rotation scheduling (time-based)
- Connected accounts (OpenAI, GitHub Copilot) - they use OAuth
- Local providers (LM Studio, Ollama, llama.cpp) - no API keys needed