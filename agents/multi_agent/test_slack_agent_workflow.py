"""Unit tests for Slack LangChain tool and multi-agent workflow."""

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

from agents.multi_agent.tools import search_slack_messages
from app.services.slack_service import (
    SlackServiceError,
    SlackNotInChannelError,
    SlackPermissionError,
)


# 1. LangChain tool invocation - Successful message search
def test_search_slack_messages_tool_success():
    mock_result = {
        "query": "payment bug",
        "count": 1,
        "items": [
            {
                "channel_id": "C01",
                "channel_name": "engineering",
                "text": "Discussing payment bug PAY-123 on checkout gateway",
                "user": "U12345",
                "ts": "1728500200.000100",
                "timestamp": "2024-10-09T18:30:00+00:00",
                "permalink": "https://slack.com/archives/C01/p1728500200000100",
            }
        ],
        "channels_searched": ["engineering"],
        "channels_skipped_not_in_channel": [],
        "note": "Search performed across accessible channel history.",
    }

    with patch("agents.multi_agent.tools.SlackService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_channel_history_sync.return_value = mock_result
        MockService.return_value = instance

        result = search_slack_messages.invoke({"query": "payment bug"})
        data = json.loads(result)
        assert data["source"] == "slack"
        assert data["count"] == 1
        assert "PAY-123" in data["items"][0]["text"]
        assert data["channels_searched"] == ["engineering"]


# 2. LangChain tool invocation with channel parameter
def test_search_slack_messages_with_specific_channel():
    mock_result = {
        "query": "deployment",
        "count": 1,
        "items": [
            {
                "channel_id": "C02",
                "channel_name": "deployments",
                "text": "v1.2.0 deployed to production",
                "user": "U999",
                "ts": "1728500300.000200",
            }
        ],
        "channels_searched": ["deployments"],
        "channels_skipped_not_in_channel": [],
        "note": "Search performed in specified channel.",
    }

    with patch("agents.multi_agent.tools.SlackService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_channel_history_sync.return_value = mock_result
        MockService.return_value = instance

        result = search_slack_messages.invoke({"query": "deployment", "channel": "deployments"})
        data = json.loads(result)
        assert data["count"] == 1
        assert "v1.2.0 deployed" in data["items"][0]["text"]
        instance.search_channel_history_sync.assert_called_once_with(
            query="deployment",
            channel="deployments",
            max_messages_per_channel=20,
        )


# 3. LangChain tool invocation - Empty search results
def test_search_slack_messages_empty_results():
    mock_result = {
        "query": "unrelated-query",
        "count": 0,
        "items": [],
        "channels_searched": ["engineering"],
        "channels_skipped_not_in_channel": [],
        "note": "No matching messages found in accessible channels.",
    }

    with patch("agents.multi_agent.tools.SlackService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_channel_history_sync.return_value = mock_result
        MockService.return_value = instance

        result = search_slack_messages.invoke({"query": "unrelated-query"})
        assert "No matching Slack messages found for query: 'unrelated-query'" in result


# 4. LangChain tool invocation - Unconfigured service
def test_search_slack_messages_unconfigured():
    with patch("agents.multi_agent.tools.SlackService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (False, "Missing SLACK_BOT_TOKEN")
        MockService.return_value = instance

        result = search_slack_messages.invoke({"query": "any query"})
        assert "Error: Slack is not configured properly." in result
        assert "Missing SLACK_BOT_TOKEN" in result


# 5. LangChain tool invocation - Service error handling
def test_search_slack_messages_error_handling():
    with patch("agents.multi_agent.tools.SlackService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_channel_history_sync.side_effect = SlackNotInChannelError(
            "Bot is not in channel 'private-eng'. Invite using /invite @workpilot_ai."
        )
        MockService.return_value = instance

        result = search_slack_messages.invoke({"query": "status", "channel": "private-eng"})
        assert "Slack error:" in result
        assert "Bot is not in channel" in result


# 6. Research subgraph tool registration and ToolNode loop
def test_research_subgraph_has_slack_tool():
    from agents.multi_agent.research_graph import tools

    tool_names = [t.name for t in tools]
    assert "search_slack_messages" in tool_names
    assert "search_jira_issues" in tool_names
    assert "search_github_repository" in tool_names


# 7. Research subgraph tool-call cycle with mocked LLM
def test_research_subgraph_slack_tool_call_execution():
    from agents.multi_agent.research_graph import research_graph

    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_slack_messages",
                "args": {"query": "payment discussion"},
                "id": "call_slack_001",
            }
        ],
    )
    final_ai_message = AIMessage(
        content="Slack discussions reveal that payment issue PAY-123 was investigated in #engineering."
    )

    with patch("agents.multi_agent.research_graph.model_with_tools") as mock_model:
        mock_model.invoke.side_effect = [tool_call_message, final_ai_message]

        with patch("agents.multi_agent.tools.SlackService") as MockService:
            instance = MagicMock()
            instance.validate_config.return_value = (True, None)
            instance.search_channel_history_sync.return_value = {
                "query": "payment discussion",
                "count": 1,
                "items": [{"text": "PAY-123 discussion", "user": "U1", "channel_name": "engineering"}],
                "channels_searched": ["engineering"],
                "note": "",
            }
            MockService.return_value = instance

            initial_state = {
                "messages": [HumanMessage(content="Search Slack for discussions about payment issue")]
            }
            output = research_graph.invoke(initial_state)

            assert len(output["messages"]) >= 2
            # Verify tool execution occurred
            messages = output["messages"]
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert "PAY-123 discussion" in tool_messages[0].content
            assert output["messages"][-1].content == final_ai_message.content
