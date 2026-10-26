"""Best-effort text token estimates for Gemini input."""

import json

from free_claude_code.core.token_estimation import estimate_text_tokens

from .models import GeminiRequest


def estimate_gemini_input_tokens(request: GeminiRequest) -> int:
    total = 0
    contents = [*request.contents]
    if request.system_instruction is not None:
        contents.append(request.system_instruction)
    for content in contents:
        for part in content.parts:
            if part.text is not None:
                total += estimate_text_tokens(part.text[:1_000_000])
            if part.function_call is not None:
                total += estimate_text_tokens(
                    part.function_call.model_dump_json()[:1_000_000]
                )
            if part.function_response is not None:
                total += estimate_text_tokens(part.function_response.name)
                total += estimate_text_tokens(
                    json.dumps(part.function_response.response)[:1_000_000]
                )
    for tool in request.tools:
        total += estimate_text_tokens(
            json.dumps(tool.model_dump(mode="json"))[:1_000_000]
        )
    return total
