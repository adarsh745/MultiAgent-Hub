"""Live End-to-End Cross-Source Investigation Test.

Executes real read-only API calls across:
1. Jira Cloud (search_jira_issues)
2. GitHub REST API (search_github_repository)
3. Slack Web API (search_slack_messages in #new-channel)
4. Full Multi-Agent workflow / Research Subgraph

Never prints, logs, or exposes tokens or secrets.
"""

import json
import os
import sys
from pathlib import Path

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


QUERY = (
    "Investigate payment-related issues using Jira, find related commits and pull requests "
    "in GitHub, and search Slack for relevant discussions in new-channel."
)


def run_cross_source_live_test() -> bool:
    print("=" * 70)
    print("Cross-Source Investigation Live Integration Test")
    print("=" * 70)
    print(f"Query: '{QUERY}'\n")

    # -------------------------------------------------------------
    # Step 1: Individual Tool Executions against Live APIs
    # -------------------------------------------------------------
    print("--- 1. Testing Live Jira Tool (search_jira_issues) ---")
    try:
        jira_res = search_jira_issues.invoke({"query": "payment-related issues"})
        print(f"Jira Output: {jira_res[:200]}...")
        assert "Error: Jira is not configured" not in jira_res
        print("[PASS] Jira tool executed against live Jira Cloud.\n")
    except Exception as exc:
        print(f"[FAIL] Jira tool execution failed: {exc}")
        return False

    print("--- 2. Testing Live GitHub Tool (search_github_repository) ---")
    try:
        gh_res = search_github_repository.invoke({"query": "payment-related commits and pull requests"})
        print(f"GitHub Output: {gh_res[:200]}...")
        assert "Error: GitHub is not configured" not in gh_res
        print("[PASS] GitHub tool executed against live GitHub REST API.\n")
    except Exception as exc:
        print(f"[FAIL] GitHub tool execution failed: {exc}")
        return False

    print("--- 3. Testing Live Slack Tool (search_slack_messages in new-channel) ---")
    try:
        slack_res = search_slack_messages.invoke({"query": "payment", "channel": "new-channel"})
        print(f"Slack Output: {slack_res[:200]}...")
        assert "Error: Slack is not configured" not in slack_res
        print("[PASS] Slack tool executed against live Slack Web API in #new-channel.\n")
    except Exception as exc:
        print(f"[FAIL] Slack tool execution failed: {exc}")
        return False

    # -------------------------------------------------------------
    # Step 2: Research Subgraph Execution (LLM -> Tools -> LLM)
    # -------------------------------------------------------------
    print("--- 4. Executing Research Subgraph with Live APIs & Ollama ---")
    try:
        graph_result = research_graph.invoke({"messages": [HumanMessage(content=QUERY)]})
        messages = graph_result.get("messages", [])

        # Check tool execution
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]

        print(f"Total messages in conversation: {len(messages)}")
        print(f"Tool executions recorded:       {len(tool_messages)}")

        for tm in tool_messages:
            preview = tm.content.replace("\n", " ")[:120]
            print(f"  - Tool Result: {preview}...")

        final_msg = ai_messages[-1] if ai_messages else None
        if final_msg:
            print("\nFinal Research Summary:")
            print("-" * 50)
            print(final_msg.content)
            print("-" * 50)

        assert len(tool_messages) >= 1, "At least one live tool must be called by Research Agent"
        print("[PASS] Research subgraph completed live cross-source investigation.\n")
    except Exception as exc:
        print(f"[FAIL] Research subgraph execution failed: {exc}")
        return False

    print("=" * 70)
    print("[SUCCESS] All Live Cross-Source Investigation Checks Passed!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_cross_source_live_test()
    sys.exit(0 if success else 1)
