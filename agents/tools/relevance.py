import logging
from typing import Any, Dict, Tuple
from agents.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRelevanceGuard:
    """Deterministic guard to check whether a selected tool is relevant to a user request."""

    # Keyword mappings for supported tools
    TOOL_KEYWORD_MAP: Dict[str, list] = {
        "calculator": [
            "calculate", "calc", "add", "plus", "sum", "subtract", "minus",
            "multiply", "multiplied", "times", "divide", "divided", "math",
            "arithmetic", "+", "-", "*", "/", "×", "÷"
        ],
        "unit_converter": [
            "convert", "conversion", "celsius", "fahrenheit", "km", "kilometer",
            "kilometers", "mile", "miles", "degree", "degrees", "°c", "°f"
        ],
        "text_analyzer": [
            "analyze", "analysis", "count", "word", "words", "character",
            "characters", "sentence", "sentences", "paragraph", "measure text"
        ],
    }

    def validate_relevance(
        self,
        user_message: str,
        tool: BaseTool,
        arguments: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate whether the selected tool is relevant to the user's request.

        Returns:
            Tuple[bool, str]: (is_relevant, reason)
        """
        msg_lower = user_message.lower().strip()
        tool_name = tool.name.lower()

        # 1. Mapped tool relevance check
        if tool_name in self.TOOL_KEYWORD_MAP:
            expected_keywords = self.TOOL_KEYWORD_MAP[tool_name]
            has_trigger = any(kw in msg_lower for kw in expected_keywords)
            if not has_trigger:
                reason = (
                    f"Selected tool '{tool.name}' is unrelated to the user's request. "
                    f"Request does not contain keywords matching tool description: '{tool.description}'."
                )
                return False, reason
            return True, "Tool is relevant."

        # 2. Generic fallback for arbitrary/future tools (e.g., jira, github, slack, database, rag_search)
        name_terms = [w.lower() for w in tool_name.split("_") if len(w) > 2]
        desc_terms = [w.lower() for w in tool.description.replace(".", "").split() if len(w) > 3]
        all_terms = set(name_terms + desc_terms)

        has_overlap = any(term in msg_lower for term in all_terms)
        if not has_overlap:
            reason = (
                f"Selected tool '{tool.name}' is unrelated to user request. "
                f"Tool description: '{tool.description}'."
            )
            return False, reason

        return True, "Tool is relevant."
