"""Unit tests for the cross-source investigation workflow.

Validates:
1. Tool selection: Research Agent calls Jira, GitHub, and Slack tools.
2. Actual tool results passed to LLM before research summary is generated.
3. Strict evidence correlation without hallucinated links.
4. Honest reporting of empty results with scope limitations.
5. Clear distinction between tool errors/failures and empty search results.
6. Full multi-agent workflow (Supervisor -> Research -> Analysis -> Verification).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
_local_dir = str(Path(__file__).resolve().parent)
if _local_dir in sys.path:
    sys.path.remove(_local_dir)

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.multi_agent.research_graph import research_graph, tools
from agents.multi_agent.tools import (
    search_jira_issues,
    search_github_repository,
    search_slack_messages,
)
from agents.multi_agent.verification import (
    is_empty_search_result,
    has_unsupported_quality_claims,
    verify_analysis_claims,
    format_factual_analysis,
)


CROSS_SOURCE_QUERY = (
    "Investigate payment-related issues using Jira, find related commits and pull requests "
    "in GitHub, and search Slack for relevant discussions in new-channel."
)


# 1. Verify all three tools are registered in the Research subgraph
def test_all_three_tools_registered_in_research_graph():
    tool_names = [t.name for t in tools]
    assert "search_jira_issues" in tool_names, "search_jira_issues must be registered"
    assert "search_github_repository" in tool_names, "search_github_repository must be registered"
    assert "search_slack_messages" in tool_names, "search_slack_messages must be registered"


# 2. Tool selection: Verify LLM calls Jira, GitHub, and Slack tools for cross-source request
def test_cross_source_tool_selection():
    mock_tool_calls_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_jira_issues",
                "args": {"query": "payment-related issues"},
                "id": "call_jira_001",
            },
            {
                "name": "search_github_repository",
                "args": {"query": "payment-related commits and pull requests"},
                "id": "call_github_001",
            },
            {
                "name": "search_slack_messages",
                "args": {"query": "payment", "channel": "new-channel"},
                "id": "call_slack_001",
            },
        ],
    )
    mock_summary_msg = AIMessage(
        content=(
            "Investigation Summary:\n"
            "- Jira: No matching payment issues found.\n"
            "- GitHub: 1 commit related to payment gateway; 0 pull requests.\n"
            "- Slack: No matching payment discussions found in new-channel."
        )
    )

    with patch("agents.multi_agent.research_graph.model_with_tools") as mock_model:
        mock_model.invoke.side_effect = [mock_tool_calls_msg, mock_summary_msg]

        with patch("agents.multi_agent.tools.JiraService") as MockJira, \
             patch("agents.multi_agent.tools.GitHubService") as MockGithub, \
             patch("agents.multi_agent.tools.SlackService") as MockSlack:

            # Mock Jira
            jira_inst = MagicMock()
            jira_inst.validate_config.return_value = (True, None)
            jira_inst.search_issues_sync.return_value = {"issues": []}
            MockJira.return_value = jira_inst

            # Mock GitHub
            gh_inst = MagicMock()
            gh_inst.validate_config.return_value = (True, None)
            gh_inst.owner = "adarsh745"
            gh_inst.repo = "Production-ready-RAG-"
            gh_inst.list_commits_sync.return_value = [
                {"sha": "abc1234", "message": "payment gateway retry", "author": "adarsh", "date": "2026-10-10"}
            ]
            gh_inst.list_pull_requests_sync.return_value = []
            MockGithub.return_value = gh_inst

            # Mock Slack
            slack_inst = MagicMock()
            slack_inst.validate_config.return_value = (True, None)
            slack_inst.list_channels_sync.return_value = [{"id": "C01", "name": "new-channel", "is_member": True}]
            slack_inst.search_channel_history_sync.return_value = {
                "query": "payment",
                "count": 0,
                "items": [],
                "channels_searched": ["new-channel"],
                "note": "No messages found.",
            }
            MockSlack.return_value = slack_inst

            result = research_graph.invoke({"messages": [HumanMessage(content=CROSS_SOURCE_QUERY)]})
            messages = result["messages"]

            # Verify tool calls were triggered
            ai_calls = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
            assert len(ai_calls) == 1
            called_tool_names = [tc["name"] for tc in ai_calls[0].tool_calls]
            assert "search_jira_issues" in called_tool_names
            assert "search_github_repository" in called_tool_names
            assert "search_slack_messages" in called_tool_names


# 3. Tool results passed to LLM before research summary is generated
def test_tool_results_passed_to_llm_before_summary():
    mock_tool_calls_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_jira_issues", "args": {"query": "payment"}, "id": "call_j1"},
            {"name": "search_slack_messages", "args": {"query": "payment", "channel": "new-channel"}, "id": "call_s1"},
        ],
    )
    mock_summary_msg = AIMessage(content="Final summary based on tool results.")

    received_messages_in_second_call = []

    def mock_invoke(messages):
        if len(received_messages_in_second_call) == 0:
            received_messages_in_second_call.append(messages)
            return mock_tool_calls_msg
        else:
            received_messages_in_second_call.append(messages)
            return mock_summary_msg

    with patch("agents.multi_agent.research_graph.model_with_tools") as mock_model:
        mock_model.invoke.side_effect = mock_invoke

        with patch("agents.multi_agent.tools.JiraService") as MockJira, \
             patch("agents.multi_agent.tools.SlackService") as MockSlack:

            jira_inst = MagicMock()
            jira_inst.validate_config.return_value = (True, None)
            jira_inst.search_issues_sync.return_value = {"issues": []}
            MockJira.return_value = jira_inst

            slack_inst = MagicMock()
            slack_inst.validate_config.return_value = (True, None)
            slack_inst.list_channels_sync.return_value = [{"id": "C01", "name": "new-channel", "is_member": True}]
            slack_inst.search_channel_history_sync.return_value = {
                "query": "payment", "count": 0, "items": [], "channels_searched": ["new-channel"], "note": "None."
            }
            MockSlack.return_value = slack_inst

            result = research_graph.invoke({"messages": [HumanMessage(content="Investigate Jira and Slack")]})
            messages = result["messages"]

            # ToolMessage instances must exist in the state
            tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_msgs) == 2, "Both tool results must be present in messages"

            # The final call to the model must include these ToolMessages
            second_call_messages = received_messages_in_second_call[1]
            tool_msgs_in_call = [m for m in second_call_messages if isinstance(m, ToolMessage)]
            assert len(tool_msgs_in_call) == 2, "LLM must receive tool messages before generating summary"


# 4. Strict correlation: No invented relationships between distinct sources
def test_strict_correlation_and_no_invented_relationships():
    research_with_no_links = (
        "Jira: Found 0 issues for query 'payment-related issues'.\n"
        "GitHub: Found commit 4e14d77 'fix(rag): deterministic fast strategy' (author: adarsh).\n"
        "Slack: Found 0 messages in new-channel matching 'payment'."
    )
    # Analysis that incorrectly invents a relationship
    bad_analysis = (
        "The commit 4e14d77 resolved the payment issue discussed in Slack new-channel, "
        "and Jira has closed the bug accordingly."
    )
    # Analysis that adheres strictly to facts
    factual_analysis = (
        "Research revealed 0 matching payment issues in Jira and 0 matching discussions in Slack new-channel. "
        "One commit 4e14d77 exists in GitHub but it does not reference any payment issue or Slack thread. "
        "This result is limited to the search performed."
    )

    is_valid_bad, report_bad = verify_analysis_claims(research_with_no_links, bad_analysis)
    assert is_valid_bad is False or "Contradiction" in report_bad or "Governs" in report_bad

    is_valid_good, report_good = verify_analysis_claims(research_with_no_links, factual_analysis)
    assert is_valid_good is True


# 5. Distinguish tool errors from empty results
def test_distinguish_tool_failure_from_empty_results():
    with patch("agents.multi_agent.tools.SlackService") as MockSlack:
        slack_inst = MagicMock()
        slack_inst.validate_config.return_value = (True, None)
        slack_inst.list_channels_sync.return_value = [{"id": "C01", "name": "new-channel", "is_member": False}]
        from app.services.slack_service import SlackNotInChannelError
        slack_inst.search_channel_history_sync.side_effect = SlackNotInChannelError("Bot is not in channel 'new-channel'.")
        MockSlack.return_value = slack_inst

        tool_result = search_slack_messages.invoke({"query": "payment", "channel": "new-channel"})
        # Must clearly indicate error/failure, not 0 matches found cleanly
        assert "Slack error:" in tool_result or "not in channel" in tool_result
        assert "0 matching messages" not in tool_result


# 6. Full multi-agent workflow integration (Supervisor -> Research -> Analysis -> Verification)
def test_full_cross_source_multi_agent_workflow():
    from agents.multi_agent.graph import graph

    # Mock research_graph output with multi-source findings
    mock_research_output = {
        "messages": [
            HumanMessage(content=CROSS_SOURCE_QUERY),
            AIMessage(
                content=(
                    "Investigation Findings across sources:\n"
                    "- Jira: No matching payment-related issues found.\n"
                    "- GitHub: Found 30 recent commits in repository adarsh745/Production-ready-RAG-; 0 pull requests found.\n"
                    "- Slack: No matching payment messages found in channel 'new-channel'.\n"
                    "Note: No correlated cross-source ticket keys were identified."
                )
            )
        ]
    }

    with patch("agents.multi_agent.agents.research_graph.invoke", return_value=mock_research_output):
        initial_state = {
            "user_request": CROSS_SOURCE_QUERY,
            "research_result": "",
            "analysis_result": "",
            "verification_result": "",
            "next_agent": "",
            "final_response": "",
        }

        final_state = graph.invoke(initial_state)

        assert final_state["research_result"] != ""
        assert "Jira" in final_state["research_result"]
        assert "GitHub" in final_state["research_result"]
        assert "Slack" in final_state["research_result"]

        assert final_state["analysis_result"] != ""
        assert final_state["verification_result"] != ""
        assert final_state["final_response"] != ""

        # Verification must not allow unsupported claims
        assert not has_unsupported_quality_claims(final_state["analysis_result"])
