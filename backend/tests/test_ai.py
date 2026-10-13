import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException, status

from app.api.ai import get_llm_service
from app.api.dependencies import get_current_user
from app.main import app
from app.models.user import User, UserRole
from app.services.llm.base import BaseLLMService


class MockLLMService(BaseLLMService):
    """Mock LLM Service for deterministic testing."""

    async def generate_response(self, prompt: str, system_prompt=None) -> str:
        return f"Mocked response for: {prompt}"


def mock_authenticated_user() -> User:
    """Mock authenticated user instance."""
    return User(
        id=1,
        organization_id=1,
        email="test_user@example.com",
        full_name="Test User",
        role=UserRole.MEMBER,
        is_active=True,
    )


@pytest.mark.asyncio
async def test_ai_chat_unauthenticated(client):
    """Unauthenticated request to /api/v1/ai/chat should return HTTP 401."""
    app.dependency_overrides.clear()
    res = await client.post("/api/v1/ai/chat", json={"message": "Hello AI"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_ai_chat_authenticated_success(client):
    """Authenticated user successfully calls /api/v1/ai/chat and gets ChatResponse."""
    app.dependency_overrides[get_current_user] = mock_authenticated_user
    app.dependency_overrides[get_llm_service] = lambda: MockLLMService()

    try:
        res = await client.post(
            "/api/v1/ai/chat",
            json={"message": "What is WorkPilot AI?"},
            headers={"Authorization": "Bearer mock_token"},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert "response" in data
        assert "Mocked response for: What is WorkPilot AI?" in data["response"]
        assert "model" in data
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ai_chat_empty_message_validation(client):
    """Empty or whitespace-only message should be rejected with 422 Unprocessable Entity."""
    app.dependency_overrides[get_current_user] = mock_authenticated_user

    try:
        # Empty string
        res1 = await client.post(
            "/api/v1/ai/chat",
            json={"message": ""},
            headers={"Authorization": "Bearer mock_token"},
        )
        assert res1.status_code == 422

        # Whitespace-only string
        res2 = await client.post(
            "/api/v1/ai/chat",
            json={"message": "    "},
            headers={"Authorization": "Bearer mock_token"},
        )
        assert res2.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ai_chat_message_too_large_validation(client):
    """Message exceeding 4000 characters should be rejected with 422."""
    app.dependency_overrides[get_current_user] = mock_authenticated_user

    try:
        large_message = "A" * 4001
        res = await client.post(
            "/api/v1/ai/chat",
            json={"message": large_message},
            headers={"Authorization": "Bearer mock_token"},
        )
        assert res.status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ai_chat_ollama_unavailable_handling(client):
    """When Ollama is unavailable, API should return 503 Service Unavailable."""
    app.dependency_overrides[get_current_user] = mock_authenticated_user

    class FailingOllamaService(BaseLLMService):
        async def generate_response(self, prompt: str, system_prompt=None) -> str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama LLM service is unavailable. Please check that Ollama is running locally.",
            )

    app.dependency_overrides[get_llm_service] = lambda: FailingOllamaService()

    try:
        res = await client.post(
            "/api/v1/ai/chat",
            json={"message": "Test Ollama Down"},
            headers={"Authorization": "Bearer mock_token"},
        )
        assert res.status_code == 503
        assert "Ollama LLM service is unavailable" in res.json()["detail"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ai_chat_llm_internal_error_handling(client):
    """When LLM service encounters internal errors, API should return HTTP 500."""
    app.dependency_overrides[get_current_user] = mock_authenticated_user

    class InternalErrorLLMService(BaseLLMService):
        async def generate_response(self, prompt: str, system_prompt=None) -> str:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error returned from local LLM provider.",
            )

    app.dependency_overrides[get_llm_service] = lambda: InternalErrorLLMService()

    try:
        res = await client.post(
            "/api/v1/ai/chat",
            json={"message": "Test Internal Error"},
            headers={"Authorization": "Bearer mock_token"},
        )
        assert res.status_code == 500
        assert "Error returned from local LLM provider" in res.json()["detail"]
    finally:
        app.dependency_overrides.clear()
