"""Unit tests for SlackService with mocked HTTP responses."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.services.slack_service import (
    SlackService,
    SlackServiceError,
    SlackConfigurationError,
    SlackAuthenticationError,
    SlackPermissionError,
    SlackNotInChannelError,
    SlackNotFoundError,
    SlackRateLimitError,
    SlackServerError,
    SlackNetworkError,
    SlackAPIError,
)


# 1. Configuration validation
def test_slack_config_validation():
    # Unconfigured service
    unconfigured = SlackService(bot_token="")
    is_valid, error = unconfigured.validate_config()
    assert is_valid is False
    assert "SLACK_BOT_TOKEN" in error
    assert unconfigured.is_configured is False

    # Configured service
    configured = SlackService(bot_token="xoxb-mock-token", default_channel="general")
    is_valid, error = configured.validate_config()
    assert is_valid is True
    assert error is None
    assert configured.is_configured is True


# 2. Successful authentication (auth.test)
def test_successful_authentication():
    service = SlackService(bot_token="xoxb-mock-token")
    mock_payload = {
        "ok": True,
        "team": "Acme Corp",
        "team_id": "T12345",
        "user": "workpilot_bot",
        "user_id": "U67890",
        "bot_id": "B11122",
        "url": "https://acme.slack.com/",
    }
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(mock_payload).encode("utf-8"),
        request=httpx.Request("POST", "https://slack.com/api/auth.test"),
    )

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = service.auth_test_sync()
        assert res["ok"] is True
        assert res["team"] == "Acme Corp"
        assert res["user"] == "workpilot_bot"
        assert res["bot_id"] == "B11122"


# 3. Channel listing
def test_channel_listing():
    service = SlackService(bot_token="xoxb-mock-token")
    mock_payload = {
        "ok": True,
        "channels": [
            {
                "id": "C01",
                "name": "general",
                "is_channel": True,
                "is_private": False,
                "is_member": True,
                "num_members": 15,
                "topic": {"value": "General chat"},
                "purpose": {"value": "Company-wide announcements"},
            },
            {
                "id": "C02",
                "name": "random",
                "is_channel": True,
                "is_private": False,
                "is_member": False,
                "num_members": 10,
                "topic": {"value": ""},
                "purpose": {"value": ""},
            },
        ],
    }
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(mock_payload).encode("utf-8"),
        request=httpx.Request("GET", "https://slack.com/api/conversations.list"),
    )

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        channels = service.list_channels_sync()
        assert len(channels) == 2
        assert channels[0]["id"] == "C01"
        assert channels[0]["name"] == "general"
        assert channels[0]["is_member"] is True
        assert channels[1]["id"] == "C02"
        assert channels[1]["is_member"] is False


# 4. Message history retrieval
def test_message_history_retrieval():
    service = SlackService(bot_token="xoxb-mock-token")
    mock_channels = {
        "ok": True,
        "channels": [{"id": "C01", "name": "general", "is_member": True}],
    }
    mock_history = {
        "ok": True,
        "messages": [
            {
                "text": "Deployment completed successfully.",
                "user": "U1234",
                "ts": "1728500000.000100",
            },
            {
                "text": "Investigating payment issue PAY-123 on checkout service.",
                "user": "U5678",
                "ts": "1728500100.000200",
            },
        ],
    }

    def mock_get(url, *args, **kwargs):
        if "conversations.list" in str(url):
            return httpx.Response(200, content=json.dumps(mock_channels).encode("utf-8"), request=httpx.Request("GET", url))
        elif "conversations.history" in str(url):
            return httpx.Response(200, content=json.dumps(mock_history).encode("utf-8"), request=httpx.Request("GET", url))
        return httpx.Response(404, request=httpx.Request("GET", url))

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        messages = service.get_channel_history_sync("general", limit=10)
        assert len(messages) == 2
        assert messages[0]["channel_id"] == "C01"
        assert messages[0]["channel_name"] == "general"
        assert "Deployment completed" in messages[0]["text"]
        assert "PAY-123" in messages[1]["text"]


# 5. Search behavior: search.messages with token type limitation
def test_search_messages_token_type_limitation():
    service = SlackService(bot_token="xoxb-mock-token")
    mock_payload = {
        "ok": False,
        "error": "not_allowed_token_type",
    }
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(mock_payload).encode("utf-8"),
        request=httpx.Request("GET", "https://slack.com/api/search.messages"),
    )

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        with pytest.raises(SlackPermissionError) as exc_info:
            service.search_messages_sync("payment")
        assert "not_allowed_token_type" in exc_info.value.message


# 6. Channel history search fallback
def test_channel_history_search_fallback():
    service = SlackService(bot_token="xoxb-mock-token")
    mock_channels = {
        "ok": True,
        "channels": [
            {"id": "C01", "name": "engineering", "is_member": True},
            {"id": "C02", "name": "random", "is_member": False},
        ],
    }
    mock_history = {
        "ok": True,
        "messages": [
            {"text": "Discussing PAY-123 retry timeout bug in gateway", "user": "U01", "ts": "1728500200.000100"},
            {"text": "Frontend design review notes", "user": "U02", "ts": "1728500300.000200"},
        ],
    }

    def mock_get(url, *args, **kwargs):
        if "conversations.list" in str(url):
            return httpx.Response(200, content=json.dumps(mock_channels).encode("utf-8"), request=httpx.Request("GET", url))
        elif "conversations.history" in str(url):
            return httpx.Response(200, content=json.dumps(mock_history).encode("utf-8"), request=httpx.Request("GET", url))
        return httpx.Response(404, request=httpx.Request("GET", url))

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        result = service.search_channel_history_sync("PAY-123")
        assert result["count"] == 1
        assert len(result["items"]) == 1
        assert "PAY-123" in result["items"][0]["text"]
        assert "engineering" in result["channels_searched"]


# 7. Empty results
def test_empty_results():
    service = SlackService(bot_token="xoxb-mock-token")
    mock_channels = {
        "ok": True,
        "channels": [{"id": "C01", "name": "general", "is_member": True}],
    }
    mock_history = {
        "ok": True,
        "messages": [],
    }

    def mock_get(url, *args, **kwargs):
        if "conversations.list" in str(url):
            return httpx.Response(200, content=json.dumps(mock_channels).encode("utf-8"), request=httpx.Request("GET", url))
        elif "conversations.history" in str(url):
            return httpx.Response(200, content=json.dumps(mock_history).encode("utf-8"), request=httpx.Request("GET", url))
        return httpx.Response(404, request=httpx.Request("GET", url))

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        res = service.search_channel_history_sync("nonexistent-term")
        assert res["count"] == 0
        assert res["items"] == []


# 8. Invalid or expired token (200 with ok: false AND 401)
def test_invalid_token_handling():
    service = SlackService(bot_token="xoxb-invalid")

    # Slack ok: false with invalid_auth
    mock_payload = {"ok": False, "error": "invalid_auth"}
    mock_resp = httpx.Response(200, content=json.dumps(mock_payload).encode("utf-8"), request=httpx.Request("POST", "url"))
    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        with pytest.raises(SlackAuthenticationError) as exc:
            service.auth_test_sync()
        assert "invalid_auth" in exc.value.message

    # HTTP 401
    mock_resp_401 = httpx.Response(401, request=httpx.Request("POST", "url"))
    with patch("httpx.AsyncClient.post", return_value=mock_resp_401):
        with pytest.raises(SlackAuthenticationError):
            service.auth_test_sync()


# 9. Missing scopes
def test_missing_scopes():
    service = SlackService(bot_token="xoxb-mock")
    mock_payload = {
        "ok": False,
        "error": "missing_scope",
        "needed": "channels:history",
        "provided": "channels:read",
    }
    mock_resp = httpx.Response(200, content=json.dumps(mock_payload).encode("utf-8"), request=httpx.Request("GET", "url"))

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        with pytest.raises(SlackPermissionError) as exc:
            service.get_channel_history_sync("C01")
        assert exc.value.needed_scope == "channels:history"
        assert "channels:history" in exc.value.message


# 10. not_in_channel
def test_not_in_channel():
    service = SlackService(bot_token="xoxb-mock")
    mock_payload = {"ok": False, "error": "not_in_channel"}
    mock_resp = httpx.Response(200, content=json.dumps(mock_payload).encode("utf-8"), request=httpx.Request("GET", "url"))

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        with pytest.raises(SlackNotInChannelError) as exc:
            service.get_channel_history_sync("C01")
        assert "not in the channel" in exc.value.message.lower()


# 11. channel_not_found
def test_channel_not_found():
    service = SlackService(bot_token="xoxb-mock")
    mock_channels = {"ok": True, "channels": []}
    mock_resp = httpx.Response(200, content=json.dumps(mock_channels).encode("utf-8"), request=httpx.Request("GET", "url"))

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        with pytest.raises(SlackNotFoundError) as exc:
            service.get_channel_history_sync("nonexistent-channel")
        assert "not found" in exc.value.message.lower()


# 12. Rate limiting (429 and ok: false with ratelimited)
def test_rate_limiting():
    service = SlackService(bot_token="xoxb-mock")

    # HTTP 429
    resp_429 = httpx.Response(429, headers={"Retry-After": "4"}, request=httpx.Request("GET", "url"))
    with patch("httpx.AsyncClient.get", return_value=resp_429):
        with pytest.raises(SlackRateLimitError) as exc:
            service.list_channels_sync()
        assert exc.value.retry_after == 4

    # 200 with error: ratelimited
    payload = {"ok": False, "error": "ratelimited"}
    resp_200 = httpx.Response(200, content=json.dumps(payload).encode("utf-8"), headers={"Retry-After": "6"}, request=httpx.Request("GET", "url"))
    with patch("httpx.AsyncClient.get", return_value=resp_200):
        with pytest.raises(SlackRateLimitError) as exc:
            service.list_channels_sync()
        assert exc.value.retry_after == 6


# 13. Network timeouts and connectivity failures
def test_network_timeouts():
    service = SlackService(bot_token="xoxb-mock")

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timeout")):
        with pytest.raises(SlackNetworkError) as exc:
            service.auth_test_sync()
        assert "timeout" in exc.value.message.lower()

    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Failed to resolve host")):
        with pytest.raises(SlackNetworkError) as exc:
            service.list_channels_sync()
        assert "network error" in exc.value.message.lower()


# 14. Server errors and unclassified ok: false
def test_server_errors_and_unclassified_api_error():
    service = SlackService(bot_token="xoxb-mock")

    # 500 error
    resp_500 = httpx.Response(500, request=httpx.Request("GET", "url"))
    with patch("httpx.AsyncClient.get", return_value=resp_500):
        with pytest.raises(SlackServerError):
            service.list_channels_sync()

    # 200 with ok: false and unclassified error
    payload = {"ok": False, "error": "some_internal_fatal_error"}
    resp_custom = httpx.Response(200, content=json.dumps(payload).encode("utf-8"), request=httpx.Request("GET", "url"))
    with patch("httpx.AsyncClient.get", return_value=resp_custom):
        with pytest.raises(SlackAPIError) as exc:
            service.list_channels_sync()
        assert "some_internal_fatal_error" in exc.value.message


# 15. Async methods
@pytest.mark.asyncio
async def test_async_methods():
    service = SlackService(bot_token="xoxb-mock")
    auth_payload = {"ok": True, "team": "Acme", "user": "bot"}
    resp = httpx.Response(200, content=json.dumps(auth_payload).encode("utf-8"), request=httpx.Request("POST", "url"))

    with patch("httpx.AsyncClient.post", return_value=resp):
        res = await service.auth_test()
        assert res["team"] == "Acme"
