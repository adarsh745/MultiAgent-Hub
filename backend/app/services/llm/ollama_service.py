import logging
from typing import Any, Dict, List, Optional
import httpx
from fastapi import HTTPException, status

from app.core.config import settings
from app.services.llm.base import BaseLLMService
logger = logging.getLogger(__name__)


class OllamaLLMService(BaseLLMService):
    """Service implementation for communicating with local Ollama API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT_SECONDS

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Send prompt to local Ollama API and return generated text."""
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload)

            if response.status_code == 404:
                logger.error("Ollama model '%s' not found.", self.model)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Configured model '{self.model}' not found in local Ollama instance.",
                )
            elif response.status_code != 200:
                logger.error("Ollama API error HTTP %s: %s", response.status_code, response.text)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error returned from local LLM provider.",
                )

            data = response.json()
            llm_text = data.get("response")
            if llm_text is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Invalid response structure returned from LLM service.",
                )

            return llm_text.strip()

        except httpx.TimeoutException:
            logger.error("Ollama request timed out after %s seconds.", self.timeout)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM request timed out.",
            )
        except (httpx.ConnectError, httpx.NetworkError, httpx.RequestError) as exc:
            logger.error("Failed to connect to Ollama at %s: %s", self.base_url, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama LLM service is unavailable. Please check that Ollama is running locally.",
            )
