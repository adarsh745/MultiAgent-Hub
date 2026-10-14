"""Unit tests for JiraService and Jira LangChain tools with mocked HTTP responses."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.services.jira_service import (
    JiraService,
    JiraServiceError,
    JiraAuthenticationError,
    JiraPermissionError,
    JiraNotFoundError,
    JiraRateLimitError,
    JiraServerError,
    JiraNetworkError,
    build_jql_from_query,
)


# 1. Jira authentication and configuration validation
def test_jira_config_validation():
    # Empty service
    service = JiraService(base_url="", email="", api_token="")
    is_valid, error = service.validate_config()
    assert is_valid is False
    assert "JIRA_BASE_URL" in error
    assert "JIRA_EMAIL" in error
    assert "JIRA_API_TOKEN" in error

    # Configured service
    configured = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )
    is_valid, error = configured.validate_config()
    assert is_valid is True
    assert error is None
    assert configured.is_configured is True


# 2. Successful issue-search response parsing and normalization
def test_successful_search_parsing():
    raw_response = {
        "issues": [
            {
                "id": "1001",
                "key": "SAM1-7",
                "fields": {
                    "summary": "Create Wallet Integration",
                    "status": {"name": "In Progress"},
                    "priority": {"name": "High"},
                    "assignee": {"displayName": "Adarsh Janjirala"},
                    "created": "2026-09-14T10:00:00.000Z",
                    "updated": "2026-09-14T12:00:00.000Z",
                },
            }
        ],
        "isLast": True,
        "nextPageToken": None,
    }

    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_response).encode("utf-8"),
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        result = service.search_issues_sync("project = SAM1")
        assert result["total"] == 1
        assert result["is_last"] is True
        issue = result["issues"][0]
        assert issue["key"] == "SAM1-7"
        assert issue["summary"] == "Create Wallet Integration"
        assert issue["status"] == "In Progress"
        assert issue["priority"] == "High"
        assert issue["assignee"] == "Adarsh Janjirala"
        assert issue["created"] == "2026-09-14T10:00:00.000Z"
        assert issue["updated"] == "2026-09-14T12:00:00.000Z"


# 3. Empty search results
def test_empty_search_results():
    raw_response = {
        "issues": [],
        "isLast": True,
    }

    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_response).encode("utf-8"),
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        result = service.search_issues_sync("text ~ 'nonexistent'")
        assert result["total"] == 0
        assert result["issues"] == []
        assert result["is_last"] is True


# 4. HTTP 401 and 403 responses
def test_http_401_unauthorized():
    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="bad_token",
    )

    mock_resp = httpx.Response(
        status_code=401,
        content=b'{"errorMessages": ["Unauthorized"]}',
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(JiraAuthenticationError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "401" in exc_info.value.message
        assert "bad_token" not in exc_info.value.message


def test_http_403_forbidden():
    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=403,
        content=b'{"errorMessages": ["Forbidden"]}',
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(JiraPermissionError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "403" in exc_info.value.message


# 5. HTTP 404, rate limiting (429), and server errors (500)
def test_http_404_not_found():
    service = JiraService(
        base_url="https://invalid-site.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=404,
        content=b'{"errorMessage": "Site temporarily unavailable"}',
        request=httpx.Request("POST", "https://invalid-site.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(JiraNotFoundError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "404" in exc_info.value.message


def test_http_429_rate_limited():
    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=429,
        content=b'{"errorMessages": ["Rate limit exceeded"]}',
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(JiraRateLimitError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "429" in exc_info.value.message


def test_http_500_server_error():
    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=500,
        content=b'{"errorMessages": ["Internal Server Error"]}',
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(JiraServerError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "500" in exc_info.value.message


# 6. Network timeout and connection failures
def test_network_timeout():
    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(JiraNetworkError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "timed out" in exc_info.value.message.lower()


def test_network_connection_error():
    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(JiraNetworkError) as exc_info:
            service.search_issues_sync("project = SAM1")
        assert "network connection" in exc_info.value.message.lower()


# 7. Asynchronous method search_issues
def test_async_search_issues():
    raw_response = {
        "issues": [
            {
                "id": "1002",
                "key": "SAM1-8",
                "fields": {
                    "summary": "Develop Transaction History Feature",
                    "status": {"name": "To Do"},
                    "priority": None,
                    "assignee": None,
                    "created": "2026-09-14T11:00:00.000Z",
                    "updated": "2026-09-14T11:30:00.000Z",
                },
            }
        ],
        "isLast": True,
    }

    service = JiraService(
        base_url="https://test.atlassian.net",
        email="test@example.com",
        api_token="mock_token",
    )

    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_response).encode("utf-8"),
        request=httpx.Request("POST", "https://test.atlassian.net/rest/api/3/search/jql"),
    )

    async def _run():
        with patch.object(httpx.AsyncClient, "post", return_value=mock_resp):
            return await service.search_issues("project = SAM1")

    result = asyncio.run(_run())
    assert result["total"] == 1
    assert result["issues"][0]["key"] == "SAM1-8"
    assert result["issues"][0]["status"] == "To Do"
    assert result["issues"][0]["assignee"] == "Unassigned"
    assert result["issues"][0]["priority"] == "None"


# 8. Deterministic JQL query builder tests
def test_build_jql_from_query():
    # Natural language with intent
    jql1 = build_jql_from_query("Find unresolved payment bugs in Jira")
    assert "statusCategory != Done" in jql1
    assert 'text ~ "payment"' in jql1
    assert "ORDER BY updated DESC" in jql1

    # Issue key
    jql2 = build_jql_from_query("Check issue SAM1-7")
    assert jql2 == 'issue = "SAM1-7"'

    # Natural-language project mentions are text, not trusted project keys.
    jql3 = build_jql_from_query("Find bugs in SAM1")
    assert "project =" not in jql3
    assert 'text ~ "sam1"' in jql3

    # Words following "project" or "in" must not become inferred project keys.
    suspicious_project_names = build_jql_from_query(
        "Investigate the RAG project in Jira updated"
    )
    assert 'project = "IN"' not in suspicious_project_names
    assert 'project = "UPDATED"' not in suspicious_project_names
    assert "text ~" in suspicious_project_names

    generated_source_query = build_jql_from_query("RAG project issues tasks bugs")
    assert "project =" not in generated_source_query
    assert 'text ~ "rag project"' in generated_source_query

    # Empty query
    jql4 = build_jql_from_query("")
    assert "project is not EMPTY" in jql4

    # Raw JQL query
    jql5 = build_jql_from_query("project = SAM1 AND statusCategory != Done")
    assert jql5 == "project = SAM1 AND statusCategory != Done"
