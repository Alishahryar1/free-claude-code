"""Pydantic models for Anthropic-compatible requests."""

from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal

from pydantic import BaseModel, field_validator, model_validator

from config.settings import get_settings
from providers.model_utils import normalize_model_name
from loguru import logger


# =============================================================================
# Content Block Types
# =============================================================================


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


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


# =============================================================================
# Message Types
# =============================================================================


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


# =============================================================================
# Request Models
# =============================================================================


class MessagesRequest(BaseModel):
    model: str
    max_tokens: Optional[int] = None
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = True
    temperature: Optional[float] = None
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
        """Map any Claude model name to the configured model."""
        settings = get_settings()
        if self.original_model is None:
            self.original_model = self.model

        # Use centralized model normalization
        normalized = normalize_model_name(self.model, settings.model)
        if normalized != self.model:
            self.model = normalized

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
        """Map any Claude model name to the configured model."""
        settings = get_settings()
        # Use centralized model normalization
        return normalize_model_name(v, settings.model)
