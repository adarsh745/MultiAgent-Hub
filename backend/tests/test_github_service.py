"""Unit tests for GitHubService with mocked HTTP responses."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import patch

from app.services.github_service import (
    GitHubService,
    GitHubServiceError,
    GitHubAuthenticationError,
    GitHubPermissionError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubServerError,
    GitHubNetworkError,
)


# 1. Configuration validation
def test_github_config_validation():
    # Empty service
    service = GitHubService(token="", owner="", repo="")
    is_valid, error = service.validate_config()
    assert is_valid is False
    assert "GITHUB_TOKEN" in error
    assert "GITHUB_OWNER" in error
    assert "GITHUB_REPO" in error

    # Properly configured service
    configured = GitHubService(
        token="test_token",
        owner="test_owner",
        repo="test_repo",
    )
    is_valid, error = configured.validate_config()
    assert is_valid is True
    assert error is None
    assert configured.is_configured is True


# 2. Successful repository retrieval
def test_successful_repository_retrieval():
    raw_repo = {
        "name": "test_repo",
        "full_name": "test_owner/test_repo",
        "owner": {"login": "test_owner"},
        "description": "A test repository",
        "visibility": "public",
        "default_branch": "main",
        "html_url": "https://github.com/test_owner/test_repo",
        "stargazers_count": 42,
        "forks_count": 5,
        "open_issues_count": 2,
    }

    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_repo).encode("utf-8"),
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        repo = service.get_repository_sync()
        assert repo["name"] == "test_repo"
        assert repo["full_name"] == "test_owner/test_repo"
        assert repo["visibility"] == "public"
        assert repo["default_branch"] == "main"
        assert repo["stars_count"] == 42


# 3. Pull-request retrieval
def test_pull_request_retrieval():
    raw_pulls = [
        {
            "number": 101,
            "title": "Add feature X",
            "state": "open",
            "user": {"login": "contributor1"},
            "html_url": "https://github.com/test_owner/test_repo/pull/101",
            "created_at": "2026-10-01T12:00:00Z",
            "updated_at": "2026-10-02T12:00:00Z",
            "draft": False,
            "merged_at": None,
        }
    ]

    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_pulls).encode("utf-8"),
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo/pulls"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        pulls = service.list_pull_requests_sync(state="open")
        assert len(pulls) == 1
        assert pulls[0]["number"] == 101
        assert pulls[0]["title"] == "Add feature X"
        assert pulls[0]["author"] == "contributor1"
        assert pulls[0]["state"] == "open"


# 4. Commit retrieval
def test_commit_retrieval():
    raw_commits = [
        {
            "sha": "a1b2c3d4e5f67890",
            "commit": {
                "message": "feat: initialize project\n\nDetailed description here",
                "author": {"name": "Developer", "date": "2026-10-01T10:00:00Z"},
            },
            "html_url": "https://github.com/test_owner/test_repo/commit/a1b2c3d4e5f67890",
        }
    ]

    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_commits).encode("utf-8"),
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo/commits"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        commits = service.list_commits_sync()
        assert len(commits) == 1
        assert commits[0]["sha"] == "a1b2c3d"
        assert commits[0]["message"] == "feat: initialize project"
        assert commits[0]["author"] == "Developer"


# 5. Issue retrieval and EXCLUSION of pull requests
def test_issue_retrieval_excludes_pull_requests():
    # In GitHub REST API, /issues returns both issues and PRs (PRs have 'pull_request' key)
    raw_items = [
        {
            "number": 1,
            "title": "Bug in login flow",
            "state": "open",
            "user": {"login": "user1"},
            "labels": [{"name": "bug"}],
            "html_url": "https://github.com/test_owner/test_repo/issues/1",
            "created_at": "2026-10-01T10:00:00Z",
            "updated_at": "2026-10-01T10:00:00Z",
            # Note: No 'pull_request' key -> real issue!
        },
        {
            "number": 2,
            "title": "Fix login bug PR",
            "state": "open",
            "user": {"login": "user2"},
            "labels": [],
            "html_url": "https://github.com/test_owner/test_repo/pull/2",
            "created_at": "2026-10-02T10:00:00Z",
            "updated_at": "2026-10-02T10:00:00Z",
            "pull_request": {"url": "https://api.github.com/repos/test_owner/test_repo/pulls/2"},
            # Note: Contains 'pull_request' key -> must be filtered out!
        },
    ]

    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_items).encode("utf-8"),
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo/issues"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        issues = service.list_issues_sync()
        # Must only contain issue #1, issue #2 (the PR) must be excluded!
        assert len(issues) == 1
        assert issues[0]["number"] == 1
        assert issues[0]["title"] == "Bug in login flow"
        assert issues[0]["labels"] == ["bug"]


# 6. Empty results
def test_empty_results():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=200,
        content=b"[]",
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo/pulls"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        pulls = service.list_pull_requests_sync()
        assert pulls == []

        issues = service.list_issues_sync()
        assert issues == []

        commits = service.list_commits_sync()
        assert commits == []


# 7. HTTP 401 Unauthorized
def test_http_401_unauthorized():
    service = GitHubService(token="bad_token", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=401,
        content=b'{"message": "Bad credentials"}',
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        with pytest.raises(GitHubAuthenticationError) as exc_info:
            service.get_repository_sync()
        assert "401" in exc_info.value.message
        assert "bad_token" not in exc_info.value.message


# 8. HTTP 403 Forbidden
def test_http_403_forbidden():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=403,
        content=b'{"message": "Resource not accessible by personal access token"}',
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        with pytest.raises(GitHubPermissionError) as exc_info:
            service.get_repository_sync()
        assert "403" in exc_info.value.message


# 9. HTTP 404 Not Found
def test_http_404_not_found():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=404,
        content=b'{"message": "Not Found"}',
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        with pytest.raises(GitHubNotFoundError) as exc_info:
            service.get_repository_sync()
        assert "404" in exc_info.value.message


# 10. HTTP 429 Rate Limit
def test_http_429_rate_limited():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=429,
        content=b'{"message": "API rate limit exceeded"}',
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        with pytest.raises(GitHubRateLimitError) as exc_info:
            service.get_repository_sync()
        assert "rate limit exceeded" in exc_info.value.message.lower()


# 11. HTTP 500 Server Error
def test_http_500_server_error():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=500,
        content=b'{"message": "Internal Server Error"}',
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    with patch.object(httpx.Client, "get", return_value=mock_resp):
        with pytest.raises(GitHubServerError) as exc_info:
            service.get_repository_sync()
        assert "500" in exc_info.value.message


# 12. Network timeout and connection failures
def test_network_timeout():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")

    with patch.object(httpx.Client, "get", side_effect=httpx.TimeoutException("Timed out")):
        with pytest.raises(GitHubNetworkError) as exc_info:
            service.get_repository_sync()
        assert "timed out" in exc_info.value.message.lower()


def test_network_connection_error():
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")

    with patch.object(httpx.Client, "get", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(GitHubNetworkError) as exc_info:
            service.get_repository_sync()
        assert "network connection" in exc_info.value.message.lower()


# 13. Asynchronous methods
def test_async_methods():
    raw_repo = {
        "name": "test_repo",
        "full_name": "test_owner/test_repo",
        "visibility": "public",
        "default_branch": "main",
    }
    service = GitHubService(token="mock", owner="test_owner", repo="test_repo")
    mock_resp = httpx.Response(
        status_code=200,
        content=json.dumps(raw_repo).encode("utf-8"),
        request=httpx.Request("GET", "https://api.github.com/repos/test_owner/test_repo"),
    )

    async def _run():
        with patch.object(httpx.AsyncClient, "get", return_value=mock_resp):
            return await service.get_repository()

    res = asyncio.run(_run())
    assert res["name"] == "test_repo"
