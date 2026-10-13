"""Slack Web API service for channels, message history, and search retrieval."""

import asyncio
import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_SLACK_API_URL = "https://slack.com/api"


class SlackServiceError(Exception):
    """Base exception for Slack service operations."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        slack_error: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.slack_error = slack_error


class SlackConfigurationError(SlackServiceError):
    """Raised when required Slack configuration is missing or invalid."""


class SlackAuthenticationError(SlackServiceError):
    """Raised when Slack authentication fails (e.g., invalid_auth, token_expired)."""


class SlackPermissionError(SlackServiceError):
    """Raised when permissions/scopes are insufficient (e.g., missing_scope, not_allowed_token_type)."""

    def __init__(
        self,
        message: str,
        needed_scope: Optional[str] = None,
        provided_scopes: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.needed_scope = needed_scope
        self.provided_scopes = provided_scopes


class SlackNotInChannelError(SlackServiceError):
    """Raised when the bot is not a member of the requested channel."""


class SlackNotFoundError(SlackServiceError):
    """Raised when a channel, user, or resource is not found (e.g., channel_not_found)."""


class SlackRateLimitError(SlackServiceError):
    """Raised when Slack rate limits are exceeded (HTTP 429 or error='ratelimited')."""

    def __init__(self, message: str, retry_after: int = 1, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class SlackServerError(SlackServiceError):
    """Raised when Slack returns a server error (HTTP 5xx or fatal_error)."""


class SlackNetworkError(SlackServiceError):
    """Raised when network timeout or connectivity fails."""


class SlackAPIError(SlackServiceError):
    """Raised when Slack returns ok=false with an unclassified error code."""


def _format_slack_timestamp(ts: Optional[str]) -> str:
    """Format a Slack timestamp (epoch seconds . sequence) into ISO 8601 string."""
    if not ts:
        return ""
    try:
        epoch = float(ts.split(".")[0])
        return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return ts


class SlackService:
    """Reusable service for interacting with the Slack Web API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_channel: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        # Load environment variables if not already loaded
        _env_path = find_dotenv(usecwd=True)
        if not _env_path:
            _env_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(_env_path)

        raw_token = bot_token if bot_token is not None else os.getenv("SLACK_BOT_TOKEN", "")
        raw_default_channel = default_channel if default_channel is not None else os.getenv("SLACK_DEFAULT_CHANNEL", "")
        raw_api_url = api_base_url if api_base_url is not None else os.getenv("SLACK_API_BASE_URL", DEFAULT_SLACK_API_URL)

        self._token = (raw_token or "").strip()
        self.default_channel = (raw_default_channel or "").strip()
        self.api_base_url = (raw_api_url or DEFAULT_SLACK_API_URL).rstrip("/")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Check if required Slack bot token is present."""
        return bool(self._token)

    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration and return missing variable names if any."""
        if not self._token:
            return False, "Missing required Slack configuration: SLACK_BOT_TOKEN"
        return True, None

    def _get_headers(self) -> Dict[str, str]:
        """Build standard Slack Web API request headers."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "WorkPilot-AI",
        }

    def _handle_error_response(self, response: httpx.Response, data: Optional[Dict[str, Any]] = None):
        """Map HTTP error status codes and Slack JSON error codes to typed exceptions."""
        status = response.status_code

        # HTTP level errors
        if status == 429:
            retry_after = int(response.headers.get("Retry-After", "1"))
            raise SlackRateLimitError(
                f"Slack API rate limit exceeded. Retry after {retry_after} seconds.",
                status_code=429,
                retry_after=retry_after,
                slack_error="ratelimited",
            )
        elif status == 401:
            raise SlackAuthenticationError(
                "Slack authentication failed: Invalid or expired token (HTTP 401).",
                status_code=401,
                slack_error="invalid_auth",
            )
        elif status == 403:
            raise SlackPermissionError(
                "Slack access forbidden (HTTP 403).",
                status_code=403,
            )
        elif status == 404:
            raise SlackNotFoundError(
                "Slack API endpoint not found (HTTP 404).",
                status_code=404,
            )
        elif status >= 500:
            raise SlackServerError(
                f"Slack server error: HTTP {status}.",
                status_code=status,
            )

        # Slack JSON level errors (even on HTTP 200)
        if data and not data.get("ok", False):
            error_code = data.get("error", "unknown_error")
            needed = data.get("needed")
            provided = data.get("provided")

            if error_code in ("invalid_auth", "account_inactive", "token_revoked", "token_expired"):
                raise SlackAuthenticationError(
                    f"Slack authentication failed: {error_code}.",
                    status_code=status,
                    slack_error=error_code,
                )
            elif error_code == "not_in_channel":
                raise SlackNotInChannelError(
                    "Bot is not in the channel. The bot must be invited to the channel to read its messages.",
                    status_code=status,
                    slack_error=error_code,
                )
            elif error_code in ("channel_not_found", "user_not_found"):
                raise SlackNotFoundError(
                    f"Slack resource not found: {error_code}.",
                    status_code=status,
                    slack_error=error_code,
                )
            elif error_code in ("missing_scope", "not_allowed_token_type", "restricted_action", "cannot_read"):
                scope_info = f" (needed: '{needed}', provided: '{provided}')" if needed else ""
                raise SlackPermissionError(
                    f"Slack permission error: {error_code}{scope_info}.",
                    status_code=status,
                    slack_error=error_code,
                    needed_scope=needed,
                    provided_scopes=provided,
                )
            elif error_code == "ratelimited":
                retry_after = int(response.headers.get("Retry-After", "1"))
                raise SlackRateLimitError(
                    f"Slack rate limit exceeded. Retry after {retry_after} seconds.",
                    status_code=status,
                    retry_after=retry_after,
                    slack_error=error_code,
                )
            else:
                raise SlackAPIError(
                    f"Slack API error: {error_code}.",
                    status_code=status,
                    slack_error=error_code,
                )

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform an HTTP request against the Slack Web API."""
        is_valid, err_msg = self.validate_config()
        if not is_valid:
            raise SlackConfigurationError(err_msg or "Invalid configuration")

        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, params=params)
                else:
                    response = await client.post(url, headers=headers, params=params, json=json_data)
        except httpx.TimeoutException as exc:
            raise SlackNetworkError(
                f"Timeout connecting to Slack API at {endpoint}: {str(exc)}"
            ) from exc
        except httpx.RequestError as exc:
            raise SlackNetworkError(
                f"Network error connecting to Slack API at {endpoint}: {str(exc)}"
            ) from exc

        try:
            data = response.json()
        except Exception:
            data = None

        if response.status_code != 200 or (data and not data.get("ok", False)):
            self._handle_error_response(response, data)

        return data or {}

    # -------------------------------------------------------------------------
    # Asynchronous API Methods
    # -------------------------------------------------------------------------

    async def auth_test(self) -> Dict[str, Any]:
        """
        Verify authentication and retrieve bot identity and team information.
        Calls POST /api/auth.test.
        """
        data = await self._request("POST", "auth.test")
        return {
            "ok": True,
            "team": data.get("team"),
            "team_id": data.get("team_id"),
            "user": data.get("user"),
            "user_id": data.get("user_id"),
            "bot_id": data.get("bot_id"),
            "url": data.get("url"),
        }

    async def list_channels(
        self,
        types: str = "public_channel",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List accessible channels.
        Calls GET /api/conversations.list with specified channel types.
        Normalizes into structured channel dictionaries.
        """
        params = {"types": types, "limit": min(limit, 200)}
        data = await self._request("GET", "conversations.list", params=params)
        raw_channels = data.get("channels", [])

        normalized = []
        for ch in raw_channels:
            topic_val = ch.get("topic", {})
            topic_str = topic_val.get("value", "") if isinstance(topic_val, dict) else str(topic_val or "")

            purpose_val = ch.get("purpose", {})
            purpose_str = purpose_val.get("value", "") if isinstance(purpose_val, dict) else str(purpose_val or "")

            normalized.append({
                "id": ch.get("id"),
                "name": ch.get("name", ""),
                "is_channel": ch.get("is_channel", True),
                "is_private": ch.get("is_private", False),
                "is_member": ch.get("is_member", False),
                "num_members": ch.get("num_members", 0),
                "topic": topic_str,
                "purpose": purpose_str,
            })
        return normalized

    async def resolve_channel_id(self, channel_identifier: str) -> Tuple[str, str]:
        """
        Resolve a channel name or ID to (channel_id, channel_name).
        If channel_identifier looks like a Slack ID (starts with C, G, D), searches by ID.
        Otherwise searches by name.
        """
        clean = (channel_identifier or "").strip().lstrip("#")
        if not clean:
            if self.default_channel:
                clean = self.default_channel.lstrip("#")
            else:
                raise SlackNotFoundError("No channel identifier specified and SLACK_DEFAULT_CHANNEL is not set.")

        # Check accessible channels by ID or name
        channels = await self.list_channels(types="public_channel")
        for ch in channels:
            if ch.get("id") == clean:
                return ch["id"], ch.get("name", clean)
            if ch.get("name", "").lower() == clean.lower():
                return ch["id"], ch.get("name", clean)

        # If it matches Slack ID pattern (starts with C, G, D), return it as ID
        if re.match(r"^[CGD][A-Z0-9]+$", clean):
            return clean, clean

        raise SlackNotFoundError(f"Slack channel '{clean}' not found among accessible public channels.")


    async def get_channel_history(
        self,
        channel: str,
        limit: int = 20,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve message history for an accessible channel.
        Calls GET /api/conversations.history.
        Normalizes into structured message dictionaries.
        """
        channel_id, channel_name = await self.resolve_channel_id(channel)

        params: Dict[str, Any] = {
            "channel": channel_id,
            "limit": min(limit, 100),
        }
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest

        data = await self._request("GET", "conversations.history", params=params)
        raw_messages = data.get("messages", [])

        normalized = []
        for msg in raw_messages:
            # Skip subtype notifications like channel_join or channel_leave unless relevant
            ts = msg.get("ts", "")
            text = msg.get("text", "")
            user = msg.get("user") or msg.get("username") or msg.get("bot_id") or "unknown"
            iso_time = _format_slack_timestamp(ts)

            # Generate approximate permalink or fallback
            permalink = f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}" if ts else ""

            normalized.append({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "text": text,
                "user": user,
                "ts": ts,
                "timestamp": iso_time,
                "permalink": permalink,
                "subtype": msg.get("subtype", ""),
            })
        return normalized

    async def search_messages(
        self,
        query: str,
        count: int = 20,
    ) -> Dict[str, Any]:
        """
        Attempt to search messages across the workspace using Slack's search.messages API.
        Note: The search.messages API requires a User OAuth Token (xoxp-...) with the 'search:read' scope.
        When invoked with a Bot token (xoxb-...), it raises SlackPermissionError, directing callers
        to search_channel_history.
        """
        params = {"query": query, "count": min(count, 50)}
        data = await self._request("GET", "search.messages", params=params)

        messages_data = data.get("messages", {})
        matches = messages_data.get("matches", [])

        normalized = []
        for item in matches:
            channel_info = item.get("channel", {})
            channel_id = channel_info.get("id", "") if isinstance(channel_info, dict) else ""
            channel_name = channel_info.get("name", "") if isinstance(channel_info, dict) else ""
            ts = item.get("ts", "")
            normalized.append({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "text": item.get("text", ""),
                "user": item.get("user") or item.get("username") or "unknown",
                "ts": ts,
                "timestamp": _format_slack_timestamp(ts),
                "permalink": item.get("permalink", ""),
            })

        return {
            "total": messages_data.get("total", len(normalized)),
            "matches": normalized,
            "query": query,
        }

    async def search_channel_history(
        self,
        query: str,
        channel: Optional[str] = None,
        max_messages_per_channel: int = 25,
    ) -> Dict[str, Any]:
        """
        Search accessible channel message history for messages matching a query.
        This provides a reliable, read-only search method working within the 'channels:read'
        and 'channels:history' bot scopes.
        """
        clean_query = (query or "").strip().lower()
        channels_searched = []
        channels_skipped_not_in_channel = []
        matched_messages = []

        stop_words = {
            "find", "search", "for", "in", "the", "a", "an", "all", "show", "get",
            "discussions", "discussion", "messages", "message", "related", "relevant",
            "issues", "issue", "slack", "channel", "any", "some", "my", "about", "to",
        }
        keywords = [
            w for w in re.findall(r"[A-Za-z0-9_-]+", clean_query)
            if w not in stop_words
        ]

        def _is_match(text: str) -> bool:
            if not clean_query:
                return True
            t_lower = text.lower()
            if clean_query in t_lower:
                return True
            if keywords and any(kw in t_lower for kw in keywords):
                return True
            return False

        if channel:
            # Search only in the specified channel
            try:
                ch_id, ch_name = await self.resolve_channel_id(channel)
                messages = await self.get_channel_history(ch_id, limit=max_messages_per_channel)
                channels_searched.append({"id": ch_id, "name": ch_name})
                for msg in messages:
                    if _is_match(msg.get("text", "")):
                        matched_messages.append(msg)
            except SlackNotInChannelError:
                channels_skipped_not_in_channel.append(channel)
        else:
            # Search across accessible public channels where the bot is a member
            public_channels = await self.list_channels(types="public_channel")
            member_channels = [ch for ch in public_channels if ch.get("is_member")]

            if not member_channels:
                # Bot is not a member of any public channel
                all_channel_names = [ch["name"] for ch in public_channels]
                return {
                    "query": query,
                    "count": 0,
                    "items": [],
                    "channels_searched": [],
                    "available_channels": all_channel_names,
                    "note": (
                        "Bot is currently not a member of any public channel. "
                        f"Found public channels: {', '.join(all_channel_names) if all_channel_names else 'none'}. "
                        "To search channel messages, invite the bot to the channel in Slack using '/invite @workpilot_ai'."
                    ),
                }

            for ch in member_channels[:5]:  # Limit to 5 member channels to respect rate limits
                try:
                    messages = await self.get_channel_history(ch["id"], limit=max_messages_per_channel)
                    channels_searched.append({"id": ch["id"], "name": ch["name"]})
                    for msg in messages:
                        if _is_match(msg.get("text", "")):
                            matched_messages.append(msg)
                except SlackNotInChannelError:
                    channels_skipped_not_in_channel.append(ch["name"])
                except Exception as exc:
                    logger.warning("Failed to fetch history for channel %s: %s", ch["name"], exc)


        return {
            "query": query,
            "count": len(matched_messages),
            "items": matched_messages,
            "channels_searched": [c["name"] for c in channels_searched],
            "channels_skipped_not_in_channel": channels_skipped_not_in_channel,
            "note": "Search performed across accessible channel history within configured bot permissions.",
        }

    # -------------------------------------------------------------------------
    # Synchronous Wrappers for LangChain Tools and Synchronous Code
    # -------------------------------------------------------------------------

    def _run_sync(self, coro):
        """Run an async coroutine synchronously, handling existing event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def auth_test_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for auth_test."""
        return self._run_sync(self.auth_test())

    def list_channels_sync(
        self,
        types: str = "public_channel",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for list_channels."""
        return self._run_sync(self.list_channels(types=types, limit=limit))

    def resolve_channel_id_sync(self, channel_identifier: str) -> Tuple[str, str]:
        """Synchronous wrapper for resolve_channel_id."""
        return self._run_sync(self.resolve_channel_id(channel_identifier))

    def get_channel_history_sync(
        self,
        channel: str,
        limit: int = 20,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_channel_history."""
        return self._run_sync(self.get_channel_history(channel, limit=limit, oldest=oldest, latest=latest))

    def search_messages_sync(
        self,
        query: str,
        count: int = 20,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for search_messages."""
        return self._run_sync(self.search_messages(query, count=count))

    def search_channel_history_sync(
        self,
        query: str,
        channel: Optional[str] = None,
        max_messages_per_channel: int = 25,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for search_channel_history."""
        return self._run_sync(self.search_channel_history(query, channel=channel, max_messages_per_channel=max_messages_per_channel))
