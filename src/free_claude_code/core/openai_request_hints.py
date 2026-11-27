"""OpenAI request fields that other protocols can safely ignore."""

# These steer prompt caching or abuse tracking and never change the answer.
OPENAI_REQUEST_HINTS = frozenset(
    {"prompt_cache_key", "prompt_cache_retention", "safety_identifier", "user"}
)
