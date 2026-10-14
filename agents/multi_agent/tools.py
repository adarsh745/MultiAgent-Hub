"""LangChain tools for multi-agent workflows including real Jira Cloud search."""

import json
import logging
import re
import sys
from pathlib import Path

from langchain_core.tools import tool

# Ensure project root and backend are on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.services.jira_service import (
    JiraService,
    JiraServiceError,
    build_jql_from_query,
)
from app.services.github_service import (
    GitHubService,
    GitHubServiceError,
)
from app.services.slack_service import (
    SlackService,
    SlackServiceError,
    SlackNotInChannelError,
    SlackNotFoundError,
    SlackPermissionError,
)

logger = logging.getLogger(__name__)

_RECENT_SLACK_INTENT_PATTERNS = (
    re.compile(r"\b(?:recent|latest|most recent|last)\b.*\b(?:messages?|conversations?|discussions?|chats?)\b", re.I),
    re.compile(r"\b(?:messages?|conversations?|discussions?|chats?)\b.*\b(?:recent|latest|recently|lately)\b", re.I),
    re.compile(r"\bwhat did (?:the )?team (?:discuss|talk about|say)\b.*\b(?:recently|lately)\b", re.I),
)


def is_recent_slack_intent(query: str) -> bool:
    """Return whether a request asks for recent Slack/team conversation history."""
    return any(pattern.search(query) for pattern in _RECENT_SLACK_INTENT_PATTERNS)


def extract_slack_channel(query: str) -> str:
    """Extract an explicitly named Slack channel without treating source names as channels."""
    hash_match = re.search(r"#([A-Za-z0-9_-]+)", query)
    if hash_match:
        return hash_match.group(1)

    channel_match = re.search(
        r"\b(?:in|from|within)\s+(?:the\s+)?([A-Za-z0-9_-]+)\s+channel\b",
        query,
        re.IGNORECASE,
    )
    return channel_match.group(1) if channel_match else ""


@tool
def search_jira_issues(query: str) -> str:
    """
    Search real Jira Cloud issues using a natural language query or JQL.
    Retrieves real issue data: key, summary, status, priority, assignee, created, and updated dates.
    Safe and read-only.
    """
    service = JiraService()
    is_valid, error = service.validate_config()
    if not is_valid:
        return f"Error: Jira is not configured properly. {error}"

    jql = build_jql_from_query(query)
    logger.info("Executing Jira JQL search: %s", jql)

    try:
        result = service.search_issues_sync(jql=jql, max_results=25)
        issues = result.get("issues", [])

        if not issues:
            return f"No matching Jira issues found for query: '{query}' (JQL executed: '{jql}')."

        return json.dumps(issues, indent=2)

    except JiraServiceError as exc:
        return f"Error searching Jira: {exc.message}"
    except Exception as exc:
        return f"Unexpected error during Jira search: {type(exc).__name__}"


@tool
def search_issues(query: str) -> str:
    """
    Legacy mock tool for searching simulated software issues without connecting to Jira.
    Retained for backward compatibility and offline testing.
    """
    mock_issues = [
        {
            "id": "BUG-101",
            "title": "Payment timeout during checkout",
            "priority": "HIGH",
            "status": "OPEN",
        },
        {
            "id": "BUG-102",
            "title": "Payment callback failure",
            "priority": "HIGH",
            "status": "RESOLVED",
        },
        {
            "id": "BUG-103",
            "title": "Payment retry creates duplicate transaction",
            "priority": "CRITICAL",
            "status": "OPEN",
        },
    ]

    results = []
    query_lower = query.lower()
    query_words = [
        w
        for w in query_lower.split()
        if w not in ("bug", "bugs", "issue", "issues", "the", "a", "an", "for", "in", "of", "find", "search")
    ]

    for issue in mock_issues:
        title_lower = issue["title"].lower()
        if query_lower in title_lower or (query_words and any(word in title_lower for word in query_words)):
            results.append(issue)

    if not results:
        return "No matching issues found."

    return str(results)


@tool
def search_github_repository(query: str) -> str:
    """
    Search GitHub repository resources: pull requests, commits, issues, and metadata.
    Accepts natural language queries like 'find recent pull requests', 'list recent commits',
    or 'search repository issues'.
    Safe and read-only.
    """
    service = GitHubService()
    is_valid, error = service.validate_config()
    if not is_valid:
        return f"Error: GitHub is not configured properly. {error}"

    clean_query = (query or "").strip().lower()

    try:
        # Route 1: Commits
        if any(w in clean_query for w in ["commit", "commits", "git log", "recent changes", "code history"]):
            commits = service.list_commits_sync(per_page=10)
            keywords = [
                w for w in re.findall(r"[A-Za-z0-9_-]+", clean_query)
                if w not in ["find", "search", "list", "get", "show", "commit", "commits", "recent", "in", "my", "the", "github", "repo", "repository"]
            ]
            if keywords:
                filtered = [c for c in commits if any(k in c["message"].lower() or k in c["author"].lower() for k in keywords)]
            else:
                filtered = commits

            if not filtered:
                return f"No matching commits found in GitHub repository '{service.owner}/{service.repo}' for query: '{query}'."

            return json.dumps({
                "repository": f"{service.owner}/{service.repo}",
                "resource": "commits",
                "count": len(filtered),
                "items": filtered,
            }, indent=2)

        # Route 2: Pull Requests
        elif any(w in clean_query for w in ["pull request", "pull requests", "pr", "prs", "pulls", "merge request"]):
            state = "closed" if "closed" in clean_query else ("all" if "all" in clean_query or "recent" in clean_query else "open")
            pulls = service.list_pull_requests_sync(state=state, per_page=10)
            keywords = [
                w for w in re.findall(r"[A-Za-z0-9_-]+", clean_query)
                if w not in ["find", "search", "list", "get", "show", "pull", "request", "requests", "pr", "prs", "recent", "in", "my", "the", "github", "repo", "repository", "all", "open", "closed"]
            ]
            if keywords:
                filtered = [pr for pr in pulls if any(k in pr["title"].lower() or k in pr["author"].lower() for k in keywords)]
            else:
                filtered = pulls

            if not filtered:
                return f"No matching pull requests found in GitHub repository '{service.owner}/{service.repo}' for query: '{query}'."

            return json.dumps({
                "repository": f"{service.owner}/{service.repo}",
                "resource": "pull_requests",
                "count": len(filtered),
                "items": filtered,
            }, indent=2)

        # Route 3: Issues
        elif any(w in clean_query for w in ["issue", "issues", "bug", "bugs", "ticket", "tickets"]):
            state = "closed" if "closed" in clean_query else ("all" if "all" in clean_query or "recent" in clean_query else "open")
            issues = service.list_issues_sync(state=state, per_page=10)
            keywords = [
                w for w in re.findall(r"[A-Za-z0-9_-]+", clean_query)
                if w not in ["find", "search", "list", "get", "show", "issue", "issues", "bug", "bugs", "recent", "in", "my", "the", "github", "repo", "repository", "all", "open", "closed"]
            ]
            if keywords:
                filtered = [i for i in issues if any(k in i["title"].lower() or k in i["author"].lower() for k in keywords)]
            else:
                filtered = issues

            if not filtered:
                return f"No matching issues found in GitHub repository '{service.owner}/{service.repo}' for query: '{query}'."

            return json.dumps({
                "repository": f"{service.owner}/{service.repo}",
                "resource": "issues",
                "count": len(filtered),
                "items": filtered,
            }, indent=2)

        # Route 4: General repository overview or fallback
        else:
            repo_info = service.get_repository_sync()
            recent_commits = service.list_commits_sync(per_page=3)
            return json.dumps({
                "repository": f"{service.owner}/{service.repo}",
                "resource": "overview",
                "metadata": repo_info,
                "recent_commits": recent_commits,
            }, indent=2)
    except GitHubServiceError as exc:
        return f"GitHub API error: {exc.message}"
    except Exception as exc:
        return f"Unexpected error during GitHub search: {type(exc).__name__}"


@tool
def search_slack_messages(query: str, channel: str = "") -> str:
    """
    Search Slack channel messages and team discussions for a query topic.
    Parameters:
    - query: Topic, keyword, or issue to search (e.g. 'payment', 'wallet', 'PAY-123').
    - channel: Optional channel name to search within (e.g. 'new-channel', 'all-adarsh').
    Safe and read-only. Operates within configured Slack bot permissions (channels:read, channels:history).
    """
    service = SlackService()
    is_valid, error = service.validate_config()
    if not is_valid:
        return f"Error: Slack is not configured properly. {error}"

    clean_query = (query or "").strip()
    target_channel = channel.strip() if channel else None

    # Extract channel from query if mentioned, e.g. "in #new-channel", "in new-channel", "channel:new-channel"
    ch_match = None
    if not target_channel:
        ch_match = re.search(
            r"(?:in\s+#?|channel[:\s]+#?)([A-Za-z0-9_-]+)",
            clean_query,
            re.IGNORECASE,
        )
    if ch_match:
        extracted = ch_match.group(1).lower()
        if extracted not in ("slack", "workspace", "my", "the", "a", "an", "this"):
            target_channel = extracted
            clean_query = re.sub(r"(?:in\s+#?|channel[:\s]+#?)" + re.escape(ch_match.group(1)), "", clean_query, flags=re.IGNORECASE).strip()

    # If query is solely a channel name (like "new-channel" or "#new-channel")
    if clean_query.lower().lstrip("#") in ("new-channel", "all-adarsh", "social"):
        target_channel = clean_query.lower().lstrip("#")
        clean_query = ""

    # If target_channel is a generic placeholder like 'general' that doesn't exist, check if query contains a real channel
    if target_channel and target_channel.lower() in ("general", "default", "none", "channel"):
        try:
            pub_channels = service.list_channels_sync(types="public_channel")
            pub_names = [c["name"].lower() for c in pub_channels if isinstance(c, dict) and "name" in c]
            if target_channel.lower() not in pub_names:
                found_real = next((name for name in pub_names if name in clean_query.lower()), None)
                if found_real:
                    target_channel = found_real
                    clean_query = re.sub(r"\b" + re.escape(found_real) + r"\b", "", clean_query, flags=re.IGNORECASE).strip()
                else:
                    target_channel = None
        except Exception:
            target_channel = None


    try:
        result = service.search_channel_history_sync(
            query=clean_query,
            channel=target_channel,
            max_messages_per_channel=20,
        )
        items = result.get("items", [])
        if not items:
            note = result.get("note", "")
            channel_str = f" in channel '{target_channel}'" if target_channel else ""
            return f"No matching Slack messages found for query: '{clean_query or 'recent discussions'}'{channel_str}. {note}"

        return json.dumps({
            "source": "slack",
            "query": clean_query or "all_messages",
            "count": len(items),
            "channels_searched": result.get("channels_searched", []),
            "items": items,
            "note": result.get("note", ""),
        }, indent=2)

    except SlackNotInChannelError as exc:
        return f"Slack error: {exc.message}"
    except SlackNotFoundError as exc:
        return f"Slack error: {exc.message}"
    except SlackPermissionError as exc:
        return f"Slack permission error: {exc.message}"
    except SlackServiceError as exc:
        return f"Slack API error: {exc.message}"
    except Exception as exc:
        return f"Unexpected error during Slack search: {type(exc).__name__}"


@tool
def get_recent_slack_messages(query: str, channel: str = "") -> str:
    """
    Retrieve recent Slack channel history for recent-conversation requests.
    Requires a specified channel or configured SLACK_DEFAULT_CHANNEL.
    """
    service = SlackService()
    is_valid, error = service.validate_config()
    if not is_valid:
        return f"Error: Slack is not configured properly. {error}"

    target_channel = (
        channel.strip().lstrip("#")
        or extract_slack_channel(query)
        or service.default_channel.lstrip("#")
    )
    if not target_channel:
        return (
            "A Slack channel is required to retrieve recent messages. "
            "Please specify a channel (for example, #engineering)."
        )

    try:
        messages = service.get_channel_history_sync(target_channel, limit=20)
    except SlackNotInChannelError as exc:
        return f"Slack error: {exc.message}"
    except SlackNotFoundError as exc:
        return f"Slack error: {exc.message}"
    except SlackPermissionError as exc:
        return f"Slack permission error: {exc.message}"
    except SlackServiceError as exc:
        return f"Slack API error: {exc.message}"
    except Exception as exc:
        return f"Unexpected error during recent Slack history retrieval: {type(exc).__name__}"

    messages = sorted(
        messages,
        key=lambda item: float(item.get("ts") or 0),
        reverse=True,
    )
    items = [
        {
            "author": message.get("user") or message.get("username") or "unknown",
            "timestamp": message.get("timestamp") or message.get("ts", ""),
            "text": message.get("text", ""),
            "permalink": message.get("permalink", ""),
        }
        for message in messages
    ]
    return json.dumps(
        {
            "source": "slack",
            "mode": "recent_history",
            "channel": target_channel,
            "count": len(items),
            "items": items,
            "note": "Recent channel history retrieved within configured Slack permissions.",
        },
        indent=2,
    )