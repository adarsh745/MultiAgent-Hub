from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMService(ABC):
    """Abstract interface for LLM provider implementations."""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        tools:
            Optional list of tool definitions that the LLM can use.
        """
        pass