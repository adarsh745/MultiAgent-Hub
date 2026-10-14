from unittest.mock import MagicMock, call, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.test_cross_source_live_integration import (
    SLACK_SEARCH_CHANNEL,
    build_source_queries,
    format_source_report,
    parse_investigation_query,
    run_cross_source_live_test,
)
from agents.multi_agent.tools import search_slack_messages
from agents.multi_agent.research_graph import research_graph


def test_parse_investigation_query_accepts_command_line_query_with_spaces():
    query = "Investigate the RAG project across Jira, GitHub, and Slack"

    assert parse_investigation_query([query]) == query


def test_parse_investigation_query_prompts_when_query_is_not_provided():
    query = "Find the latest release blockers"
    prompt = MagicMock(return_value=query)

    assert parse_investigation_query([], input_fn=prompt) == query
    prompt.assert_called_once_with("Enter investigation query: ")


def test_build_source_queries_keeps_topic_and_uses_separate_search_terms():
    query = "Investigate the RAG project across Jira, GitHub, and Slack"

    assert build_source_queries(query) == {
        "Jira": "RAG project issues tasks bugs",
        "GitHub commits": "RAG project commits",
        "GitHub pull requests": "RAG project pull requests",
        "Slack": "RAG project discussions",
    }


def test_slack_tool_propagates_explicit_channel_even_if_query_mentions_jira():
    with patch("agents.multi_agent.tools.SlackService") as mock_service:
        service = MagicMock()
        service.validate_config.return_value = (True, None)
        service.search_channel_history_sync.return_value = {
            "count": 0,
            "items": [],
            "channels_searched": [SLACK_SEARCH_CHANNEL],
            "note": "Search performed in specified channel.",
        }
        mock_service.return_value = service

        result = search_slack_messages.invoke(
            {"query": "RAG project in Jira", "channel": SLACK_SEARCH_CHANNEL}
        )

    service.search_channel_history_sync.assert_called_once_with(
        query="RAG project in Jira",
        channel=SLACK_SEARCH_CHANNEL,
        max_messages_per_channel=20,
    )
    assert f"in channel '{SLACK_SEARCH_CHANNEL}'" in result


def test_research_graph_executes_all_planned_source_queries_exactly():
    plan = [
        {"name": "search_jira_issues", "args": {"query": "RAG project issues tasks bugs"}},
        {"name": "search_github_repository", "args": {"query": "RAG project commits"}},
        {"name": "search_github_repository", "args": {"query": "RAG project pull requests"}},
        {
            "name": "search_slack_messages",
            "args": {"query": "RAG project discussions", "channel": SLACK_SEARCH_CHANNEL},
        },
    ]
    with patch("agents.multi_agent.research_graph.model") as mock_model, patch(
        "agents.multi_agent.tools.JiraService"
    ) as mock_jira, patch(
        "agents.multi_agent.tools.GitHubService"
    ) as mock_github, patch(
        "agents.multi_agent.tools.SlackService"
    ) as mock_slack:
        jira_service = MagicMock()
        jira_service.validate_config.return_value = (True, None)
        jira_service.search_issues_sync.return_value = {"issues": []}
        mock_jira.return_value = jira_service

        github_service = MagicMock()
        github_service.validate_config.return_value = (True, None)
        github_service.owner = "example"
        github_service.repo = "rag"
        github_service.list_commits_sync.return_value = [
            {"sha": "abc123", "message": "RAG project commit", "author": "test"}
        ]
        github_service.list_pull_requests_sync.return_value = []
        mock_github.return_value = github_service

        slack_service = MagicMock()
        slack_service.validate_config.return_value = (True, None)
        slack_service.search_channel_history_sync.return_value = {
            "count": 0,
            "items": [],
            "channels_searched": [SLACK_SEARCH_CHANNEL],
            "note": "Limited channel history search.",
        }
        mock_slack.return_value = slack_service
        result = research_graph.invoke(
            {
                "messages": [HumanMessage(content="Investigate RAG")],
                "source_search_plan": plan,
            }
        )

    jira_service.search_issues_sync.assert_called_once()
    assert jira_service.search_issues_sync.call_args.kwargs["jql"] == (
        'text ~ "rag project" ORDER BY updated DESC'
    )
    github_service.list_commits_sync.assert_called_once_with(per_page=10)
    github_service.list_pull_requests_sync.assert_called_once_with(state="open", per_page=10)
    slack_service.search_channel_history_sync.assert_called_once_with(
        query="RAG project discussions",
        channel=SLACK_SEARCH_CHANNEL,
        max_messages_per_channel=20,
    )
    tool_results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_results) == 4
    mock_model.invoke.assert_not_called()
    final_report = result["messages"][-1].content
    assert "Source evidence report" in final_report
    assert "zero matches do not establish absence" in final_report
    assert "Search completed with 1 returned result(s)" in final_report
    assert "GitHub commits" in final_report
    assert "GitHub pull requests" in final_report
    assert f"Specified channel #{SLACK_SEARCH_CHANNEL}" in final_report


def test_format_source_report_includes_actual_findings_scope_and_errors():
    report = format_source_report(
        "Slack",
        "Investigate RAG in Jira",
        "RAG discussions",
        "Slack error: channel_not_found",
        channel=SLACK_SEARCH_CHANNEL,
    )

    assert "User query: Investigate RAG in Jira" in report
    assert "Search query: RAG discussions" in report
    assert "Search scope: At most 20 recent channel-history messages in #new-channel." in report
    assert "Integration: FAILURE: Slack error: channel_not_found" in report
    assert "Actual result:\nSlack error: channel_not_found" in report


def test_live_test_propagates_query_to_each_source_and_research_subgraph(capsys):
    query = "Investigate the RAG project across Jira, GitHub, and Slack"
    jira_tool = MagicMock(name="search_jira_issues")
    jira_tool.name = "search_jira_issues"
    github_tool = MagicMock(name="search_github_repository")
    github_tool.name = "search_github_repository"
    slack_tool = MagicMock(name="search_slack_messages")
    slack_tool.name = "search_slack_messages"
    for tool in (jira_tool, github_tool, slack_tool):
        tool.invoke.return_value = "No matching results found."

    graph = MagicMock()
    graph.invoke.return_value = {
        "messages": [
            HumanMessage(content=query),
            ToolMessage(content="No matching results found.", tool_call_id="call-1"),
            AIMessage(content="No matching results were returned in the searched sources."),
        ]
    }

    with patch(
        "agents.test_cross_source_live_integration.search_jira_issues", jira_tool
    ), patch(
        "agents.test_cross_source_live_integration.search_github_repository", github_tool
    ), patch(
        "agents.test_cross_source_live_integration.search_slack_messages", slack_tool
    ), patch(
        "agents.test_cross_source_live_integration.research_graph", graph
    ):
        assert run_cross_source_live_test(query)

    jira_tool.invoke.assert_called_once_with({"query": "RAG project issues tasks bugs"})
    github_tool.invoke.assert_has_calls(
        [
            call({"query": "RAG project commits"}),
            call({"query": "RAG project pull requests"}),
        ]
    )
    assert github_tool.invoke.call_count == 2
    slack_tool.invoke.assert_called_once_with(
        {"query": "RAG project discussions", "channel": SLACK_SEARCH_CHANNEL}
    )
    graph_input = graph.invoke.call_args.args[0]
    assert graph_input["messages"] == [HumanMessage(content=query)]
    assert graph_input["source_search_plan"] == [
        {"name": "search_jira_issues", "args": {"query": "RAG project issues tasks bugs"}},
        {"name": "search_github_repository", "args": {"query": "RAG project commits"}},
        {
            "name": "search_github_repository",
            "args": {"query": "RAG project pull requests"},
        },
        {
            "name": "search_slack_messages",
            "args": {
                "query": "RAG project discussions",
                "channel": SLACK_SEARCH_CHANNEL,
            },
        },
    ]

    output = capsys.readouterr().out
    assert f"Query: {query}" in output
    assert "Search succeeded with zero matches; this limited search does not establish absence." in output
    assert "[SUCCESS] All source integrations connected" in output
