"""Live End-to-End Cross-Source Investigation Test.

Executes real read-only API calls across:
1. Jira Cloud (search_jira_issues)
2. GitHub REST API (search_github_repository)
3. Slack Web API (search_slack_messages in #new-channel)
4. Full Multi-Agent workflow / Research Subgraph

Never prints, logs, or exposes tokens or secrets.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = _project_root / ".env"
load_dotenv(dotenv_path)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.multi_agent.tools import (
    search_jira_issues,
    search_github_repository,
    search_slack_messages,
)
from agents.multi_agent.research_graph import research_graph


SLACK_SEARCH_CHANNEL = "new-channel"
_QUERY_FILLER_WORDS = {
    "across",
    "and",
    "about",
    "find",
    "for",
    "from",
    "github",
    "investigate",
    "investigation",
    "in",
    "jira",
    "of",
    "on",
    "related",
    "relevant",
    "search",
    "slack",
    "the",
    "using",
}


def parse_investigation_query(
    argv: Sequence[str],
    input_fn: Callable[[str], str] = input,
) -> str:
    parser = argparse.ArgumentParser(description="Run a live cross-source investigation.")
    parser.add_argument(
        "query",
        nargs="*",
        help="Investigation query (quote it to keep it as one shell argument).",
    )
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip()

    if not query:
        query = input_fn("Enter investigation query: ").strip()
    if not query:
        raise ValueError("An investigation query is required.")
    return query


def build_source_queries(query: str) -> dict[str, str]:
    topic_words = [
        word
        for word in re.findall(r"[A-Za-z0-9_-]+", query)
        if word.lower() not in _QUERY_FILLER_WORDS
    ]
    topic = " ".join(topic_words) or query.strip()
    return {
        "Jira": f"{topic} issues tasks bugs",
        "GitHub commits": f"{topic} commits",
        "GitHub pull requests": f"{topic} pull requests",
        "Slack": f"{topic} discussions",
    }


def _source_search_status(result: str) -> str:
    lowered = result.lower()
    if _is_tool_error(result):
        return "Integration/search failure; no search conclusion can be drawn."
    if "no matching" in lowered or "found 0 " in lowered:
        return "Search succeeded with zero matches; this limited search does not establish absence."

    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return "Search response returned; inspect the evidence below."

    if isinstance(parsed, list):
        return f"Search succeeded with {len(parsed)} returned result(s)." if parsed else (
            "Search succeeded with zero matches; this limited search does not establish absence."
        )
    if isinstance(parsed, dict):
        items = parsed.get("items")
        count = parsed.get("count")
        if isinstance(items, list):
            return f"Search succeeded with {len(items)} returned result(s)." if items else (
                "Search succeeded with zero matches; this limited search does not establish absence."
            )
        if isinstance(count, int):
            return f"Search succeeded with {count} returned result(s)." if count else (
                "Search succeeded with zero matches; this limited search does not establish absence."
            )
    return "Search response returned; inspect the evidence below."


def _is_tool_error(result: str) -> bool:
    error_prefixes = (
        "error:",
        "error searching jira:",
        "unexpected error",
        "jira api error:",
        "github api error:",
        "slack error:",
        "slack permission error:",
        "slack api error:",
    )
    return result.lstrip().lower().startswith(error_prefixes)


def format_source_report(
    source: str,
    user_query: str,
    search_query: str,
    result: str,
    channel: str | None = None,
) -> str:
    if source == "Jira":
        search_scope = "Up to 25 Jira issues matching the generated bounded JQL text search."
    elif source.startswith("GitHub"):
        search_scope = "Up to 10 recent repository records of the requested type."
    else:
        channel_scope = f"#{channel}" if channel else "accessible public channels"
        search_scope = f"At most 20 recent channel-history messages in {channel_scope}."

    integration_status = (
        f"FAILURE: {result}" if _is_tool_error(result) else "SUCCESS: tool returned a search response"
    )
    return (
        f"Source: {source}\n"
        f"User query: {user_query}\n"
        f"Search query: {search_query}\n"
        f"Search scope: {search_scope}\n"
        f"Integration: {integration_status}\n"
        f"Findings status: {_source_search_status(result)}\n"
        f"Actual result:\n{result}"
    )


def run_cross_source_live_test(query: str) -> bool:
    print("=" * 70)
    print("Cross-Source Investigation Live Integration Test")
    print("=" * 70)
    print(f"Query: {query}\n")

    # -------------------------------------------------------------
    # Step 1: Individual Tool Executions against Live APIs
    # -------------------------------------------------------------
    source_queries = build_source_queries(query)
    source_searches = (
        (
            "Jira",
            search_jira_issues,
            source_queries["Jira"],
            {},
            None,
        ),
        (
            "GitHub commits",
            search_github_repository,
            source_queries["GitHub commits"],
            {},
            None,
        ),
        (
            "GitHub pull requests",
            search_github_repository,
            source_queries["GitHub pull requests"],
            {},
            None,
        ),
        (
            "Slack",
            search_slack_messages,
            source_queries["Slack"],
            {"channel": SLACK_SEARCH_CHANNEL},
            SLACK_SEARCH_CHANNEL,
        ),
    )
    connectivity_succeeded = True
    for index, (source, tool, source_query, extra_args, channel) in enumerate(source_searches, 1):
        print(f"--- {index}. Testing Live {source} Tool ({tool.name}) ---")
        print(f"Search query: {source_query}")
        try:
            result = tool.invoke({"query": source_query, **extra_args})
        except Exception as exc:
            result = f"Error: tool invocation raised {type(exc).__name__}: {exc}"
            connected = False
        else:
            connected = not _is_tool_error(result)

        print(
            format_source_report(
                source,
                query,
                source_query,
                result,
                channel=channel,
            )
        )
        connectivity_succeeded = connectivity_succeeded and connected

    # -------------------------------------------------------------
    # Step 2: Research Subgraph Execution (LLM -> Tools -> LLM)
    # -------------------------------------------------------------
    print("--- 5. Executing Research Subgraph with Live APIs & Ollama ---")
    try:
        source_search_plan = [
            {
                "name": search_jira_issues.name,
                "args": {"query": source_queries["Jira"]},
            },
            {
                "name": search_github_repository.name,
                "args": {"query": source_queries["GitHub commits"]},
            },
            {
                "name": search_github_repository.name,
                "args": {"query": source_queries["GitHub pull requests"]},
            },
            {
                "name": search_slack_messages.name,
                "args": {
                    "query": source_queries["Slack"],
                    "channel": SLACK_SEARCH_CHANNEL,
                },
            },
        ]
        graph_result = research_graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "source_search_plan": source_search_plan,
            }
        )
        messages = graph_result.get("messages", [])

        # Check tool execution
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]

        print(f"Total messages in conversation: {len(messages)}")
        print(f"Tool executions recorded:       {len(tool_messages)}")

        for tm in tool_messages:
            print(f"\n  - Tool Result ({getattr(tm, 'name', 'tool')}):\n{tm.content}")

        final_msg = ai_messages[-1] if ai_messages else None
        if final_msg:
            print("\nFinal Research Summary:")
            print("-" * 50)
            print(final_msg.content)
            print("-" * 50)

        if len(tool_messages) < 1:
            raise AssertionError("At least one live tool must be called by Research Agent")
        print("[PASS] Research subgraph completed live cross-source investigation.\n")
    except Exception as exc:
        print(f"[FAIL] Research subgraph execution failed: {exc}")
        return False

    print("=" * 70)
    if connectivity_succeeded:
        print("[SUCCESS] All source integrations connected; inspect search status above for evidence.")
    else:
        print("[FAILURE] One or more source integrations failed; see connectivity statuses above.")
    print("=" * 70)
    return connectivity_succeeded


if __name__ == "__main__":
    try:
        investigation_query = parse_investigation_query(sys.argv[1:])
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if run_cross_source_live_test(investigation_query) else 1)
