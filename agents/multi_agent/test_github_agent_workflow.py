"""Unit tests for GitHub LangChain tool and multi-agent workflow."""

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
from langchain_ollama import ChatOllama

from agents.multi_agent.tools import search_github_repository
from app.services.github_service import GitHubServiceError, GitHubAuthenticationError


# 1. LangChain tool invocation - Commits
def test_search_github_commits_tool():
    mock_commits = [
        {
            "sha": "4e14d77",
            "full_sha": "4e14d77483a840dbf2484772b37e7db92186762c",
            "message": "fix(rag): use deterministic fast strategy",
            "author": "adarsh",
            "date": "2026-10-10T12:00:00Z",
            "html_url": "https://github.com/adarsh745/Production-ready-RAG-/commit/4e14d77",
        }
    ]

    with patch("agents.multi_agent.tools.GitHubService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.owner = "adarsh745"
        instance.repo = "Production-ready-RAG-"
        instance.list_commits_sync.return_value = mock_commits
        MockService.return_value = instance

        result = search_github_repository.invoke({"query": "show recent commits in my repository"})
        data = json.loads(result)
        assert data["resource"] == "commits"
        assert data["count"] == 1
        assert data["items"][0]["sha"] == "4e14d77"


# 2. LangChain tool invocation - Pull Requests
def test_search_github_pull_requests_tool():
    mock_prs = [
        {
            "number": 12,
            "title": "Upgrade LLM model",
            "state": "open",
            "author": "contributor",
            "html_url": "https://github.com/adarsh745/Production-ready-RAG-/pull/12",
            "created_at": "2026-10-09T10:00:00Z",
            "updated_at": "2026-10-09T10:00:00Z",
            "draft": False,
            "merged_at": None,
        }
    ]

    with patch("agents.multi_agent.tools.GitHubService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.owner = "adarsh745"
        instance.repo = "Production-ready-RAG-"
        instance.list_pull_requests_sync.return_value = mock_prs
        MockService.return_value = instance

        result = search_github_repository.invoke({"query": "Find recent pull requests in my GitHub repository"})
        data = json.loads(result)
        assert data["resource"] == "pull_requests"
        assert data["count"] == 1
        assert data["items"][0]["number"] == 12


# 3. LangChain tool invocation - Empty Results
def test_search_github_tool_empty():
    with patch("agents.multi_agent.tools.GitHubService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.owner = "adarsh745"
        instance.repo = "Production-ready-RAG-"
        instance.list_pull_requests_sync.return_value = []
        MockService.return_value = instance

        result = search_github_repository.invoke({"query": "Find recent pull requests in my GitHub repository"})
        assert "No matching pull requests found in GitHub repository" in result


# 4. LangChain tool invocation - Missing Config
def test_search_github_tool_missing_config():
    with patch("agents.multi_agent.tools.GitHubService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (False, "Missing required GitHub configuration: GITHUB_TOKEN")
        MockService.return_value = instance

        result = search_github_repository.invoke({"query": "pull requests"})
        assert "Error: GitHub is not configured properly" in result
        assert "GITHUB_TOKEN" in result


# 5. LangChain tool invocation - Error Handling
def test_search_github_tool_error_handling():
    with patch("agents.multi_agent.tools.GitHubService") as MockService:
        instance = MagicMock()
        instance.validate_config.return_value = (True, None)
        instance.list_pull_requests_sync.side_effect = GitHubAuthenticationError("GitHub authentication failed (HTTP 401).")
        MockService.return_value = instance

        result = search_github_repository.invoke({"query": "pull requests"})
        assert "GitHub API error:" in result
        assert "401" in result


# 6. Research subgraph tool calling loop with GitHub tool
def test_research_subgraph_github_tool_calling():
    from agents.multi_agent.research_graph import research_graph

    tool_call_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_github_repository",
                "args": {"query": "Find recent pull requests in my GitHub repository"},
                "id": "call_gh_123",
            }
        ],
    )
    final_ai_message = AIMessage(
        content="Research findings: There are no open or closed pull requests in the repository adarsh745/Production-ready-RAG-."
    )

    with patch("agents.multi_agent.research_graph.model_with_tools") as mock_model:
        mock_model.invoke.side_effect = [tool_call_message, final_ai_message]

        with patch("agents.multi_agent.tools.GitHubService") as MockService:
            instance = MagicMock()
            instance.validate_config.return_value = (True, None)
            instance.owner = "adarsh745"
            instance.repo = "Production-ready-RAG-"
            instance.list_pull_requests_sync.return_value = []
            MockService.return_value = instance

            output = research_graph.invoke({
                "messages": [HumanMessage(content="Find recent pull requests in my GitHub repository.")]
            })

            messages = output["messages"]
            assert len(messages) >= 3
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            assert len(tool_messages) == 1
            assert "No matching pull requests found" in tool_messages[0].content
            assert messages[-1].content == "Research findings: There are no open or closed pull requests in the repository adarsh745/Production-ready-RAG-."


# 7. Complete Supervisor -> Research (GitHub) -> Analysis -> Verification workflow
def test_complete_supervisor_github_workflow():
    from agents.multi_agent.graph import graph

    mock_research_output = {
        "messages": [
            AIMessage(content="Research: No pull requests were found in repository adarsh745/Production-ready-RAG-.")
        ]
    }
    mock_analysis_output = {
        "messages": [
            AIMessage(content="Analysis: The query returned no pull requests. This does not establish that no PRs exist outside the search scope.")
        ]
    }
    mock_verification_response = AIMessage(
        content="The analysis accurately reflects the search result and does not make unsupported claims about overall system quality."
    )

    with patch("agents.multi_agent.agents.research_graph.invoke", return_value=mock_research_output):
        with patch("agents.multi_agent.agents.analysis_graph.invoke", return_value=mock_analysis_output):
            with patch.object(ChatOllama, "invoke", return_value=mock_verification_response):
                state = {
                    "user_request": "Find recent pull requests in my GitHub repository",
                    "research_result": "",
                    "analysis_result": "",
                    "verification_result": "",
                    "next_agent": "",
                    "final_response": "",
                }

                result = graph.invoke(state)

                assert "No pull requests were found" in result["research_result"]
                assert "returned no pull requests" in result["analysis_result"]
                assert "accurately reflects the search result" in result["verification_result"]
                assert "accurately reflects the search result" in result["final_response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
