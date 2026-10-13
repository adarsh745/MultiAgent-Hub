from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, model_validator


class ToolDecision(BaseModel):
    action: Literal["tool", "final"]
    tool: Optional[str] = None
    arguments: Dict[str, Any] = {}
    response: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action == "tool":
            if not self.tool or not isinstance(self.tool, str) or not self.tool.strip():
                raise ValueError("When action is 'tool', a non-empty 'tool' string is required.")
        elif self.action == "final":
            if self.response is None:
                raise ValueError("When action is 'final', a 'response' string is required.")
        return self
