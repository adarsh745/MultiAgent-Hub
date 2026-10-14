"""Jira Cloud service for retrieving and searching issues."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

# Search fields requested from Jira Cloud
DEFAULT_JIRA_FIELDS = [
    "summary",
    "status",
    "priority",
    "assignee",
    "created",
    "updated",
]

# Intent and stop words for natural language to JQL conversion
STOP_WORDS = {
    "find", "search", "for", "in", "jira", "the", "a", "an", "all", "show", "me",
    "get", "issues", "issue", "bugs", "bug", "tickets", "ticket", "tasks", "task",
    "unresolved", "resolved", "open", "closed", "done", "high", "low", "critical",
    "priority", "about", "related", "to", "of", "with", "any", "some", "my", "please",
    "type", "component", "status", "query", "filter", "label", "labels", "category",
}


def build_jql_from_query(query: str) -> str:
    """
    Deterministically convert a natural language query or structured string into safe JQL.
    Natural-language terms use a bounded text search; project constraints are accepted only
    as explicit JQL rather than inferred from arbitrary words.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return "project is not EMPTY ORDER BY updated DESC"

    # 1. Direct JQL check: if query already contains standard JQL operators
    jql_operators = [
        " = ", " != ", " ~ ", " !~ ", " in (", " IN (",
        " is not ", " IS NOT ", " order by ", " ORDER BY ",
    ]
    if any(op in clean_query for op in jql_operators):
        upper_query = clean_query.upper()
        if upper_query.startswith("ORDER BY"):
            return f"project is not EMPTY {clean_query}"
        return clean_query

    # 2. Check for explicit issue key pattern (e.g. SAM1-7, KAN-12)
    key_match = re.search(r"\b([A-Z][A-Z0-9]+-\d+)\b", clean_query)
    if key_match:
        return f'issue = "{key_match.group(1)}"'

    # 3. Natural language parsing
    lower_query = clean_query.lower()
    clauses = []

    # Status intent
    if any(term in lower_query for term in ["unresolved", "open", "in progress", "pending", "active", "to do", "todo"]):
        clauses.append("statusCategory != Done")
    elif any(term in lower_query for term in ["resolved", "closed", "done", "completed"]):
        clauses.append("statusCategory = Done")

    # Priority intent
    if any(term in lower_query for term in ["high priority", "critical", "highest", "blocker"]):
        clauses.append("priority in (High, Highest, Critical)")
    elif any(term in lower_query for term in ["low priority", "lowest"]):
        clauses.append("priority in (Low, Lowest)")

    # Keywords extraction: strip pseudo-qualifier prefixes (e.g. component:wallet -> wallet)
    raw_words = re.findall(r"[A-Za-z0-9_:-]+", lower_query)
    normalized_words = [
        re.sub(r"^(?:type|component|status|project|query|label|priority|category):", "", w)
        for w in raw_words
    ]
    keywords = [
        w for w in normalized_words
        if w and w not in STOP_WORDS
    ]
    if keywords:
        text_term = " ".join(keywords)
        text_term_safe = text_term.replace('"', '\\"')
        clauses.append(f'text ~ "{text_term_safe}"')

    # If no specific clauses were generated, ensure query is bounded
    if not clauses:
        clauses.append("project is not EMPTY")

    return " AND ".join(clauses) + " ORDER BY updated DESC"


class JiraServiceError(Exception):
    """Base exception for Jira service operations."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class JiraAuthenticationError(JiraServiceError):
    """Raised when Jira authentication fails (HTTP 401)."""


class JiraPermissionError(JiraServiceError):
    """Raised when access to Jira resources is forbidden (HTTP 403)."""


class JiraNotFoundError(JiraServiceError):
    """Raised when the Jira site or resource is not found (HTTP 404)."""


class JiraRateLimitError(JiraServiceError):
    """Raised when Jira API rate limit is exceeded (HTTP 429)."""


class JiraServerError(JiraServiceError):
    """Raised when Atlassian Jira Cloud returns a 5xx server error."""


class JiraNetworkError(JiraServiceError):
    """Raised when network timeout or connectivity fails."""


class JiraService:
    """Service for interacting with Jira Cloud REST API v3."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = 15.0,
    ):
        # Load environment variables if not already present
        _env_path = find_dotenv(usecwd=True)
        if not _env_path:
            _env_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(_env_path)

        # Allow passing config or reading from environment
        raw_base_url = base_url if base_url is not None else os.getenv("JIRA_BASE_URL", "")
        raw_email = email if email is not None else os.getenv("JIRA_EMAIL", "")
        raw_token = api_token if api_token is not None else os.getenv("JIRA_API_TOKEN", "")

        self.base_url = (raw_base_url or "").rstrip("/")
        self.email = (raw_email or "").strip()
        self._api_token = (raw_token or "").strip()
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if all required Jira credentials are set."""
        return bool(self.base_url and self.email and self._api_token)

    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration and return missing variable names if any."""
        missing = []
        if not self.base_url:
            missing.append("JIRA_BASE_URL")
        if not self.email:
            missing.append("JIRA_EMAIL")
        if not self._api_token:
            missing.append("JIRA_API_TOKEN")

        if missing:
            return False, f"Missing required Jira configuration: {', '.join(missing)}"
        return True, None

    def _ensure_bounded_jql(self, jql: str) -> str:
        """
        Ensure JQL query is bounded.
        Jira Cloud's /rest/api/3/search/jql rejects unbounded queries with HTTP 400.
        """
        cleaned = (jql or "").strip()
        if not cleaned:
            return "project is not EMPTY ORDER BY updated DESC"

        # Check if query only specifies ordering without any search filter
        upper = cleaned.upper()
        if upper.startswith("ORDER BY"):
            return f"project is not EMPTY {cleaned}"

        return cleaned

    def normalize_issue(self, raw_issue: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Jira issue payload into a clean, structured dictionary."""
        fields = raw_issue.get("fields", {}) or {}

        # Status
        status_val = fields.get("status")
        status_name = "Unknown"
        if isinstance(status_val, dict):
            status_name = status_val.get("name", "Unknown")
        elif isinstance(status_val, str):
            status_name = status_val

        # Priority
        priority_val = fields.get("priority")
        priority_name = "None"
        if isinstance(priority_val, dict):
            priority_name = priority_val.get("name", "None")
        elif isinstance(priority_val, str):
            priority_name = priority_val

        # Assignee
        assignee_val = fields.get("assignee")
        assignee_name = "Unassigned"
        if isinstance(assignee_val, dict):
            assignee_name = (
                assignee_val.get("displayName")
                or assignee_val.get("emailAddress")
                or "Unassigned"
            )
        elif isinstance(assignee_val, str):
            assignee_name = assignee_val

        return {
            "key": raw_issue.get("key", raw_issue.get("id", "UNKNOWN")),
            "summary": fields.get("summary", ""),
            "status": status_name,
            "priority": priority_name,
            "assignee": assignee_name,
            "created": fields.get("created"),
            "updated": fields.get("updated"),
        }

    def _build_request_payload(
        self,
        jql: str,
        max_results: int,
        next_page_token: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Construct the POST payload for /rest/api/3/search/jql."""
        payload: Dict[str, Any] = {
            "jql": self._ensure_bounded_jql(jql),
            "fields": fields or DEFAULT_JIRA_FIELDS,
            "maxResults": max_results,
        }
        if next_page_token:
            payload["nextPageToken"] = next_page_token
        return payload

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Map HTTP error status codes to typed exceptions without leaking credentials."""
        status = response.status_code

        # Attempt to parse error messages from response body
        error_msg = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                error_messages = body.get("errorMessages") or []
                field_errors = body.get("errors") or {}
                parts = list(error_messages)
                if isinstance(field_errors, dict):
                    parts.extend(f"{k}: {v}" for k, v in field_errors.items())
                if parts:
                    error_msg = "; ".join(parts)
                elif "errorMessage" in body:
                    error_msg = str(body["errorMessage"])
        except Exception:
            error_msg = response.text[:200]

        if status == 401:
            raise JiraAuthenticationError(
                f"Jira authentication failed (HTTP 401). Verify JIRA_EMAIL and JIRA_API_TOKEN.",
                status_code=401,
            )
        elif status == 403:
            raise JiraPermissionError(
                f"Jira permission denied (HTTP 403). Check account permissions and site access.",
                status_code=403,
            )
        elif status == 404:
            raise JiraNotFoundError(
                f"Jira site or endpoint not found (HTTP 404) at '{self.base_url}'. "
                f"Verify JIRA_BASE_URL is active and provisioned. Details: {error_msg or 'Not Found'}",
                status_code=404,
            )
        elif status == 429:
            raise JiraRateLimitError(
                "Jira API rate limit exceeded (HTTP 429). Please retry after a cooldown.",
                status_code=429,
            )
        elif 500 <= status < 600:
            raise JiraServerError(
                f"Atlassian Cloud server error (HTTP {status}): {error_msg or 'Internal Server Error'}",
                status_code=status,
            )
        else:
            raise JiraServiceError(
                f"Jira API error (HTTP {status}): {error_msg or response.text[:200]}",
                status_code=status,
            )

    def _parse_search_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize raw Jira search response into standard result dict."""
        raw_issues = data.get("issues", [])
        normalized_issues = [self.normalize_issue(i) for i in raw_issues]
        is_last = data.get("isLast", True)
        next_page_token = data.get("nextPageToken")

        return {
            "issues": normalized_issues,
            "total": len(normalized_issues),
            "is_last": is_last,
            "next_page_token": next_page_token,
        }

    async def search_issues(
        self,
        jql: str,
        max_results: int = 50,
        next_page_token: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously search Jira issues using JQL via /rest/api/3/search/jql.

        Returns:
            Dict containing:
                - 'issues': List of normalized issue dictionaries
                - 'total': Number of issues in this page
                - 'is_last': Boolean indicating whether this is the final page
                - 'next_page_token': Optional token for fetching the next page
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise JiraServiceError(error or "Jira service not configured.")

        url = f"{self.base_url}/rest/api/3/search/jql"
        payload = self._build_request_payload(
            jql=jql,
            max_results=max_results,
            next_page_token=next_page_token,
            fields=fields,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    auth=(self.email, self._api_token),
                    json=payload,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                )

            if response.status_code != 200:
                self._handle_response_error(response)

            data = response.json()
            return self._parse_search_response(data)

        except (JiraServiceError, JiraAuthenticationError, JiraPermissionError, JiraNotFoundError, JiraRateLimitError, JiraServerError):
            raise
        except httpx.TimeoutException as exc:
            raise JiraNetworkError(
                f"Connection timed out while connecting to Jira Cloud at '{self.base_url}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise JiraNetworkError(
                f"Failed to establish network connection to Jira Cloud at '{self.base_url}'."
            ) from exc
        except httpx.RequestError as exc:
            raise JiraNetworkError(
                f"Network request error contacting Jira: {type(exc).__name__}."
            ) from exc

    def search_issues_sync(
        self,
        jql: str,
        max_results: int = 50,
        next_page_token: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronously search Jira issues using JQL via /rest/api/3/search/jql.
        Useful for synchronous tool invocations and compatibility.
        """
        is_valid, error = self.validate_config()
        if not is_valid:
            raise JiraServiceError(error or "Jira service not configured.")

        url = f"{self.base_url}/rest/api/3/search/jql"
        payload = self._build_request_payload(
            jql=jql,
            max_results=max_results,
            next_page_token=next_page_token,
            fields=fields,
        )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    auth=(self.email, self._api_token),
                    json=payload,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                )

            if response.status_code != 200:
                self._handle_response_error(response)

            data = response.json()
            return self._parse_search_response(data)

        except (JiraServiceError, JiraAuthenticationError, JiraPermissionError, JiraNotFoundError, JiraRateLimitError, JiraServerError):
            raise
        except httpx.TimeoutException as exc:
            raise JiraNetworkError(
                f"Connection timed out while connecting to Jira Cloud at '{self.base_url}'."
            ) from exc
        except httpx.ConnectError as exc:
            raise JiraNetworkError(
                f"Failed to establish network connection to Jira Cloud at '{self.base_url}'."
            ) from exc
        except httpx.RequestError as exc:
            raise JiraNetworkError(
                f"Network request error contacting Jira: {type(exc).__name__}."
            ) from exc
