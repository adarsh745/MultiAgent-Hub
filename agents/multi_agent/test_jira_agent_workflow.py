"""Unit tests for Jira LangChain tool, Research subgraph, and Multi-Agent workflow."""

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

from agents.multi_agent.tools import (
    build_jql_from_query,
    search_jira_issues,
    search_issues,
)
from app.services.jira_service import JiraServiceError, JiraAuthenticationError


# Test 7. LangChain tool invocation
def test_search_jira_issues_tool_success():
    mock_issues = [
        {
            "key": "SAM1-7",
            "summary": "Create Wallet Integration",
            "status": "In Progress",
            "priority": "None",
            "assignee": "Unassigned",
            "created": "2026-09-14T15:13:54.617+0530",
            "updated": "2026-09-14T15:13:57.370+0530",
        }
    ]

    with patch("agents.multi_agent.tools.JiraService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_issues_sync.return_value = {
            "issues": mock_issues,
            "total": 1,
            "is_last": True,
        }
        MockService.return_value = instance

        result = search_jira_issues.invoke({"query": "Find wallet issues"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key"] == "SAM1-7"
        assert parsed[0]["summary"] == "Create Wallet Integration"
        assert parsed[0]["status"] == "In Progress"


def test_search_jira_issues_tool_empty():
    with patch("agents.multi_agent.tools.JiraService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_issues_sync.return_value = {
            "issues": [],
            "total": 0,
            "is_last": True,
        }
        MockService.return_value = instance

        result = search_jira_issues.invoke({"query": "Find unresolved payment bugs"})
        assert "No matching Jira issues found" in result


def test_search_jira_issues_tool_missing_config():
    with patch("agents.multi_agent.tools.JiraService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (
            False,
            "Missing required Jira configuration: JIRA_BASE_URL",
        )
        MockService.return_value = instance

        result = search_jira_issues.invoke({"query": "payment"})
        assert "Error: Jira is not configured properly" in result
        assert "JIRA_BASE_URL" in result


def test_search_jira_issues_tool_error_handling():
    with patch("agents.multi_agent.tools.JiraService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.search_issues_sync.side_effect = JiraAuthenticationError(
            "Jira authentication failed (HTTP 401). Verify JIRA_EMAIL and JIRA_API_TOKEN."
        )
        MockService.return_value = instance

        result = search_jira_issues.invoke({"query": "payment"})
        assert "Error searching Jira:" in result
        assert "401" in result


def test_legacy_search_issues_tool():
    result = search_issues.invoke({"query": "payment timeout"})
    assert "BUG-101" in result


# Test 8. Research subgraph tool calling and final response generation
def test_research_subgraph_execution():
    from agents.multi_agent.research_graph import research_graph

    # Mock model_with_tools to simulate LLM making a tool call, then synthesizing final answer
    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_jira_issues",
                "args": {"query": "Find unresolved payment bugs"},
                "id": "call_123",
            }
        ],
    )
    final_ai_message = AIMessage(
        content="Research Findings: No unresolved payment bugs were found in Jira."
    )

    with patch("agents.multi_agent.research_graph.model_with_tools") as mock_model:
        mock_model.invoke.side_effect = [tool_call_message, final_ai_message]

        with patch("agents.multi_agent.tools.JiraService") as MockService:
            instance = MagicMock()
            instance.validate_config.return_value = (True, None)
            instance.search_issues_sync.return_value = {
                "issues": [],
                "total": 0,
                "is_last": True,
            }
            MockService.return_value = instance

            output = research_graph.invoke({
                "messages": [HumanMessage(content="Find unresolved payment bugs in Jira")]
            })

            messages = output["messages"]
            assert len(messages) >= 3
            # Check tool message was returned to LLM
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert "No matching Jira issues found" in tool_messages[0].content

            # Check final response
            assert messages[-1].content == "Research Findings: No unresolved payment bugs were found in Jira."


# Test 9. Complete Supervisor -> Research -> Analysis -> Verification workflow
def test_complete_supervisor_workflow():
    from agents.multi_agent.graph import graph
    from langchain_ollama import ChatOllama

    mock_research_output = {
        "messages": [
            AIMessage(content="Research: No unresolved payment bugs exist in Jira.")
        ]
    }
    mock_analysis_output = {
        "messages": [
            AIMessage(content="Analysis: Payment systems are clear of reported unresolved defects.")
        ]
    }
    mock_verification_response = AIMessage(
        content="Verified: The research and analysis accurately reflect that no payment bugs exist."
    )

    with patch("agents.multi_agent.agents.research_graph.invoke", return_value=mock_research_output):
        with patch("agents.multi_agent.agents.analysis_graph.invoke", return_value=mock_analysis_output):
            with patch.object(ChatOllama, "invoke", return_value=mock_verification_response):
                state = {
                    "user_request": "Find unresolved payment bugs in Jira",
                    "research_result": "",
                    "analysis_result": "",
                    "verification_result": "",
                    "next_agent": "",
                    "final_response": "",
                }

                result = graph.invoke(state)

                assert result["research_result"] == "Research: No unresolved payment bugs exist in Jira."
                assert "Analysis: Payment systems" in result["analysis_result"]
                assert "Verified:" in result["verification_result"]
                assert result["final_response"] == "Verified: The research and analysis accurately reflect that no payment bugs exist."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
