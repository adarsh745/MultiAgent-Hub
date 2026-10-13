from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Payload for chat endpoint request."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Prompt or message to send to the LLM.",
    )

    @field_validator("message")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or contain only whitespace.")
        return stripped


class ChatResponse(BaseModel):
    """Response payload returned from chat endpoint."""

    response: str
    model: Optional[str] = None
