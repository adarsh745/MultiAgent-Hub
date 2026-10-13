"""GitHub REST API service for repository, pull request, commit, and issue retrieval."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_GITHUB_API_URL = "https://api.github.com"
DEFAULT_API_VERSION = "2022-11-28"


class GitHubServiceError(Exception):
    """Base exception for GitHub service operations."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GitHubAuthenticationError(GitHubServiceError):
    """Raised when GitHub authentication fails (HTTP 401)."""


class GitHubPermissionError(GitHubServiceError):
    """Raised when GitHub access is forbidden (HTTP 403)."""


class GitHubNotFoundError(GitHubServiceError):
    """Raised when the repository or resource is not found (HTTP 404)."""


class GitHubRateLimitError(GitHubServiceError):
    """Raised when GitHub API rate limits are exceeded (HTTP 429 or 403 rate-limit)."""


class GitHubServerError(GitHubServiceError):
    """Raised when GitHub returns a 5xx server error."""


class GitHubNetworkError(GitHubServiceError):
    """Raised when network timeout or connectivity fails."""


class GitHubService:
    """Service for interacting with GitHub REST API v3."""

    def __init__(
        self,
        token: Optional[str] = None,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        # Load environment variables if not already present
        _env_path = find_dotenv(usecwd=True)
        if not _env_path:
            _env_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(_env_path)

        raw_token = token if token is not None else os.getenv("GITHUB_TOKEN", "")
        raw_owner = owner if owner is not None else os.getenv("GITHUB_OWNER", "")
        raw_repo = repo if repo is not None else os.getenv("GITHUB_REPO", "")
        raw_api_url = api_base_url if api_base_url is not None else os.getenv("GITHUB_API_BASE_URL", DEFAULT_GITHUB_API_URL)

        self._token = (raw_token or "").strip()
        self.owner = (raw_owner or "").strip()
        self.repo = (raw_repo or "").strip()
        self.api_base_url = (raw_api_url or DEFAULT_GITHUB_API_URL).rstrip("/")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if all required GitHub credentials and repository settings are present."""
        return bool(self._token and self.owner and self.repo)

    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration and return missing variable names if any."""
        missing = []
        if not self._token:
            missing.append("GITHUB_TOKEN")
        if not self.owner:
            missing.append("GITHUB_OWNER")
        if not self.repo:
            missing.append("GITHUB_REPO")

        if missing:
            return False, f"Missing required GitHub configuration: {', '.join(missing)}"
        return True, None

    def _get_headers(self) -> Dict[str, str]:
        """Build standard GitHub REST API request headers."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": DEFAULT_API_VERSION,
            "User-Agent": "WorkPilot-AI",
        }

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Map HTTP error status codes to typed exceptions without exposing tokens."""
        status = response.status_code
        error_msg = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                error_msg = body.get("message") or ""
        except Exception:
            error_msg = response.text[:200]

        # Rate limit check (GitHub uses 403 or 429 for rate limiting)
        remaining = response.headers.get("x-ratelimit-remaining")
        if status == 429 or (status == 403 and (remaining == "0" or "rate limit" in error_msg.lower())):
            raise GitHubRateLimitError(
                f"GitHub API rate limit exceeded (HTTP {status}). Please wait before retrying.",
                status_code=status,
            )

        if status == 401:
            raise GitHubAuthenticationError(
                "GitHub authentication failed (HTTP 401). Verify GITHUB_TOKEN.",
                status_code=401,
            )
        elif status == 403:
            raise GitHubPermissionError(
                f"GitHub access forbidden (HTTP 403). Check token scopes/permissions for repository '{self.owner}/{self.repo}'. Details: {error_msg or 'Forbidden'}",
                status_code=403,
            )
        elif status == 404:
            raise GitHubNotFoundError(
                f"GitHub repository or resource not found (HTTP 404): '{self.owner}/{self.repo}'. Verify GITHUB_OWNER and GITHUB_REPO.",
                status_code=404,
            )
        elif 500 <= status < 600:
            raise GitHubServerError(
                f"GitHub server error (HTTP {status}): {error_msg or 'Internal Server Error'}",
                status_code=status,
            )
        else:
            raise GitHubServiceError(
                f"GitHub API error (HTTP {status}): {error_msg or response.text[:200]}",
                status_code=status,
            )

    # -------------------------------------------------------------------------
    # Normalizers
    # -------------------------------------------------------------------------

    def normalize_repository(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw GitHub repository response."""
        return {
            "name": data.get("name", self.repo),
            "full_name": data.get("full_name", f"{self.owner}/{self.repo}"),
            "owner": (data.get("owner") or {}).get("login", self.owner),
            "description": data.get("description") or "",
            "visibility": data.get("visibility", "public"),
            "default_branch": data.get("default_branch", "main"),
            "html_url": data.get("html_url", ""),
            "stars_count": data.get("stargazers_count", 0),
            "forks_count": data.get("forks_count", 0),
            "open_issues_count": data.get("open_issues_count", 0),
        }

    def normalize_pull_request(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw GitHub pull request response."""
        user_obj = item.get("user") or {}
        return {
            "number": item.get("number"),
            "title": item.get("title", ""),
            "state": item.get("state", "open"),
            "author": user_obj.get("login") or "unknown",
            "html_url": item.get("html_url", ""),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "draft": item.get("draft", False),
            "merged_at": item.get("merged_at"),
        }

    def normalize_commit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw GitHub commit response."""
        commit_obj = item.get("commit") or {}
        author_obj = commit_obj.get("author") or {}
        message = commit_obj.get("message") or ""
        short_msg = message.splitlines()[0] if message else ""
        sha = item.get("sha") or ""

        return {
            "sha": sha[:7] if sha else "unknown",
            "full_sha": sha,
            "message": short_msg,
            "author": author_obj.get("name") or (item.get("author") or {}).get("login") or "unknown",
            "date": author_obj.get("date"),
            "html_url": item.get("html_url", ""),
        }

    def normalize_issue(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw GitHub issue response."""
        user_obj = item.get("user") or {}
        labels = [label.get("name") for label in item.get("labels", []) if isinstance(label, dict)]

        return {
            "number": item.get("number"),
            "title": item.get("title", ""),
            "state": item.get("state", "open"),
            "author": user_obj.get("login") or "unknown",
            "html_url": item.get("html_url", ""),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "labels": labels,
        }

    # -------------------------------------------------------------------------
    # Asynchronous API Methods
    # -------------------------------------------------------------------------

    async def get_repository(self) -> Dict[str, Any]:
        """Asynchronously retrieve repository metadata."""
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers())

            if response.status_code != 200:
                self._handle_response_error(response)

            return self.normalize_repository(response.json())

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while connecting to GitHub API for '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to establish network connection to GitHub API at '{self.api_base_url}'."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    async def list_pull_requests(
        self,
        state: str = "open",
        per_page: int = 10,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve pull requests with state filtering."""
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}/pulls"
        params = {"state": state, "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers(), params=params)

            if response.status_code != 200:
                self._handle_response_error(response)

            raw_pulls = response.json()
            if not isinstance(raw_pulls, list):
                return []

            return [self.normalize_pull_request(pr) for pr in raw_pulls]

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while fetching pull requests from '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to connect to GitHub API while fetching pull requests."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    async def list_commits(
        self,
        per_page: int = 10,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve recent repository commits."""
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}/commits"
        params = {"per_page": per_page, "page": page}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers(), params=params)

            if response.status_code != 200:
                self._handle_response_error(response)

            raw_commits = response.json()
            if not isinstance(raw_commits, list):
                return []

            return [self.normalize_commit(c) for c in raw_commits]

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while fetching commits from '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to connect to GitHub API while fetching commits."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    async def list_issues(
        self,
        state: str = "open",
        per_page: int = 10,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieve repository issues.
        EXCLUDES pull requests from issue results as required.
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}/issues"
        params = {"state": state, "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers(), params=params)

            if response.status_code != 200:
                self._handle_response_error(response)

            raw_items = response.json()
            if not isinstance(raw_items, list):
                return []

            # Filter out pull requests (in GitHub REST API, PRs contain the 'pull_request' key)
            issues_only = [item for item in raw_items if "pull_request" not in item]
            return [self.normalize_issue(i) for i in issues_only]

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while fetching issues from '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to connect to GitHub API while fetching issues."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    # -------------------------------------------------------------------------
    # Synchronous API Wrappers (for synchronous LangChain tools)
    # -------------------------------------------------------------------------

    def get_repository_sync(self) -> Dict[str, Any]:
        """Synchronously retrieve repository metadata."""
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self._get_headers())

            if response.status_code != 200:
                self._handle_response_error(response)

            return self.normalize_repository(response.json())

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while connecting to GitHub API for '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to establish network connection to GitHub API at '{self.api_base_url}'."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    def list_pull_requests_sync(
        self,
        state: str = "open",
        per_page: int = 10,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Synchronously retrieve pull requests."""
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}/pulls"
        params = {"state": state, "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self._get_headers(), params=params)

            if response.status_code != 200:
                self._handle_response_error(response)

            raw_pulls = response.json()
            if not isinstance(raw_pulls, list):
                return []

            return [self.normalize_pull_request(pr) for pr in raw_pulls]

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while fetching pull requests from '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to connect to GitHub API while fetching pull requests."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    def list_commits_sync(
        self,
        per_page: int = 10,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """Synchronously retrieve recent commits."""
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}/commits"
        params = {"per_page": per_page, "page": page}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self._get_headers(), params=params)

            if response.status_code != 200:
                self._handle_response_error(response)

            raw_commits = response.json()
            if not isinstance(raw_commits, list):
                return []

            return [self.normalize_commit(c) for c in raw_commits]

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while fetching commits from '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to connect to GitHub API while fetching commits."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc

    def list_issues_sync(
        self,
        state: str = "open",
        per_page: int = 10,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Synchronously retrieve repository issues.
        EXCLUDES pull requests from issue results.
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise GitHubServiceError(error or "GitHub service not configured.")

        url = f"{self.api_base_url}/repos/{self.owner}/{self.repo}/issues"
        params = {"state": state, "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self._get_headers(), params=params)

            if response.status_code != 200:
                self._handle_response_error(response)

            raw_items = response.json()
            if not isinstance(raw_items, list):
                return []

            # Filter out pull requests
            issues_only = [item for item in raw_items if "pull_request" not in item]
            return [self.normalize_issue(i) for i in issues_only]

        except (GitHubServiceError, GitHubAuthenticationError, GitHubPermissionError, GitHubNotFoundError, GitHubRateLimitError, GitHubServerError):
            raise
        except httpx.TimeoutException as exc:
            raise GitHubNetworkError(
                f"Connection timed out while fetching issues from '{self.owner}/{self.repo}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise GitHubNetworkError(
                f"Failed to connect to GitHub API while fetching issues."
            ) from exc
        except httpx.RequestError as exc:
            raise GitHubNetworkError(
                f"Network request error contacting GitHub: {type(exc).__name__}."
            ) from exc
