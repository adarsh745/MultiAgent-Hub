import logging
from typing import Annotated
from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.models.user import User
from app.schemas.ai import ChatRequest, ChatResponse
from app.services.llm.base import BaseLLMService
from app.services.llm.ollama_service import OllamaLLMService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI / LLM"])


def get_llm_service() -> BaseLLMService:
    """Dependency providing the active LLM service instance."""
    return OllamaLLMService()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_llm(
    payload: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    llm_service: Annotated[BaseLLMService, Depends(get_llm_service)],
) -> ChatResponse:
    """JWT-protected endpoint to generate LLM completions using local Ollama service."""
    logger.info("User ID %s initiated LLM chat request.", current_user.id)

    llm_output = await llm_service.generate_response(prompt=payload.message)

    return ChatResponse(
        response=llm_output,
        model=settings.OLLAMA_MODEL,
    )
