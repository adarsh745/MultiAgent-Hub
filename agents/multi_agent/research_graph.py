import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agents.multi_agent.research_state import ResearchState
from agents.multi_agent.tools import (
    search_jira_issues,
    search_github_repository,
    search_slack_messages,
    get_recent_slack_messages,
    is_recent_slack_intent,
    search_issues,
)

# Initialize ChatOllama model with qwen2.5:1.5b
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
)

# Register real Jira, GitHub, and Slack tools; search_issues retained for backward compatibility
tools = [
    search_jira_issues,
    search_github_repository,
    search_slack_messages,
    get_recent_slack_messages,
    search_issues,
]
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an expert Research Agent responsible for retrieving engineering information from Jira, GitHub, and Slack.\n\n"
        "Available tools:\n"
        "- search_jira_issues(query: str): Searches real Jira Cloud issues, bugs, ticket statuses, and project tasks.\n"
        "- search_github_repository(query: str): Searches GitHub repository metadata, pull requests, commits, and code changes.\n"
        "- search_slack_messages(query: str, channel: str): Searches Slack channel messages and team discussions. If a channel like 'new-channel' is specified, pass channel='new-channel'. Otherwise leave channel empty.\n"
        "- get_recent_slack_messages(query: str, channel: str): Retrieves recent message history, newest first; use it for latest/recent conversation requests rather than keyword search. If no channel is named, the tool uses SLACK_DEFAULT_CHANNEL or asks the user to specify one.\n"
        "- search_issues(query: str): Legacy mock issue search tool.\n\n"
        "Tool Selection Guidelines:\n"
        "1. For questions about Jira tickets, issues, statuses, priorities, or project tasks: Call search_jira_issues.\n"
        "2. For questions about GitHub repository, pull requests, commits, code changes, or repository issues: Call search_github_repository.\n"
        "3. For questions about Slack messages, channels, discussions, or team chat: Call search_slack_messages.\n"
        "4. For multi-source investigations involving multiple systems (e.g., Jira, GitHub, and Slack): Call each corresponding tool to collect data across all requested systems.\n"
        "5. If the user message includes a source-specific search plan, call each listed source tool using its exact query and channel. "
        "Do not infer a Slack channel from query text when the plan explicitly specifies one. "
        "After receiving results, provide an accurate, factual summary based strictly on returned data. "
        "Distinguish successful searches with zero matches from integration errors. State the searched scope and limits, identify missing evidence, "
        "and never infer that no updates exist, or that a project is inactive or undocumented, from a limited search with no matches. "
        "Correlate items only when supported by issue keys, commit hashes, or explicit references. Do not invent findings or relationships."
    )
)


def _format_planned_search_report(
    source_search_plan: list[dict[str, object]],
    tool_messages: list[ToolMessage],
) -> str:
    report_sections = ["Source evidence report (limited searches; zero matches do not establish absence):"]
    for call, tool_message in zip(source_search_plan, tool_messages):
        tool_name = str(call["name"])
        args = call["args"]
        query = args.get("query", "")
        if tool_name == "search_jira_issues":
            source = "Jira"
            scope = "Bounded Jira text search; up to 25 issues."
        elif tool_name == "search_github_repository":
            source = "GitHub commits" if "commit" in query.lower() else "GitHub pull requests"
            scope = "Recent repository records; up to 10 items."
        else:
            source = "Slack"
            channel = args.get("channel", "")
            scope = f"Specified channel #{channel}; up to 20 recent history messages."

        result = str(tool_message.content)
        lowered_result = result.lower()
        failed = lowered_result.startswith(
            (
                "error:",
                "error searching jira:",
                "unexpected error",
                "jira api error:",
                "github api error:",
                "slack error:",
                "slack permission error:",
                "slack api error:",
            )
        )
        if failed:
            status = "Integration/search failure; findings are unavailable."
        elif "no matching" in lowered_result or "found 0 " in lowered_result:
            status = "Search completed with zero matches within this limited scope."
        else:
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                status = "Search response returned; findings are shown below."
            else:
                if isinstance(parsed_result, list):
                    result_count = len(parsed_result)
                elif isinstance(parsed_result, dict) and isinstance(
                    parsed_result.get("items"), list
                ):
                    result_count = len(parsed_result["items"])
                elif isinstance(parsed_result, dict) and isinstance(
                    parsed_result.get("count"), int
                ):
                    result_count = parsed_result["count"]
                else:
                    result_count = None
                status = (
                    f"Search completed with {result_count} returned result(s) within this limited scope."
                    if result_count is not None
                    else "Search response returned; findings are shown below."
                )

        report_sections.append(
            f"\n{source}\n"
            f"Query: {query}\n"
            f"Scope: {scope}\n"
            f"Status: {status}\n"
            f"Actual result:\n{result}"
        )
    return "\n".join(report_sections)



def research_llm_node(state: ResearchState):
    source_search_plan = state.get("source_search_plan")
    if source_search_plan:
        messages = list(state["messages"])
        if not any(isinstance(message, ToolMessage) for message in messages):
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": call["name"],
                        "args": call["args"],
                        "id": f"planned-search-{index}",
                    }
                    for index, call in enumerate(source_search_plan, 1)
                ],
            )
            return {"messages": [response]}

        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        response = AIMessage(
            content=_format_planned_search_report(source_search_plan, tool_messages)
        )
    else:
        messages = list(state["messages"])
        recent_query = next(
            (
                str(message.content)
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        if recent_query and is_recent_slack_intent(recent_query):
            recent_tool_results = [
                message
                for message in messages
                if isinstance(message, ToolMessage)
                and message.name == get_recent_slack_messages.name
            ]
            if recent_tool_results:
                response = AIMessage(
                    content=(
                        "Recent Slack messages (newest first):\n"
                        f"{recent_tool_results[-1].content}"
                    )
                )
            else:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": get_recent_slack_messages.name,
                            "args": {"query": recent_query},
                            "id": "recent-slack-history",
                        }
                    ],
                )
        else:
            messages = [SYSTEM_PROMPT] + messages
            response = model_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(ResearchState)

# Add nodes
builder.add_node("research_llm", research_llm_node)
builder.add_node("tools", ToolNode(tools))

# Add edges following flow: START -> Research LLM -> tools_condition -> ToolNode/END
builder.add_edge(START, "research_llm")
builder.add_conditional_edges(
    "research_llm",
    tools_condition,
)
builder.add_edge("tools", "research_llm")

research_graph = builder.compile()
