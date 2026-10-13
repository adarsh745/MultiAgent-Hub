from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from agents.multi_agent.research_state import ResearchState
from agents.multi_agent.tools import (
    search_jira_issues,
    search_github_repository,
    search_slack_messages,
    search_issues,
)

# Initialize ChatOllama model with qwen2.5:1.5b
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
)

# Register real Jira, GitHub, and Slack tools; search_issues retained for backward compatibility
tools = [search_jira_issues, search_github_repository, search_slack_messages, search_issues]
model_with_tools = model.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an expert Research Agent responsible for retrieving engineering information from Jira, GitHub, and Slack.\n\n"
        "Available tools:\n"
        "- search_jira_issues(query: str): Searches real Jira Cloud issues, bugs, ticket statuses, and project tasks.\n"
        "- search_github_repository(query: str): Searches GitHub repository metadata, pull requests, commits, and code changes.\n"
        "- search_slack_messages(query: str, channel: str): Searches Slack channel messages and team discussions. If a channel like 'new-channel' is specified, pass channel='new-channel'. Otherwise leave channel empty.\n"
        "- search_issues(query: str): Legacy mock issue search tool.\n\n"
        "Tool Selection Guidelines:\n"
        "1. For questions about Jira tickets, issues, statuses, priorities, or project tasks: Call search_jira_issues.\n"
        "2. For questions about GitHub repository, pull requests, commits, code changes, or repository issues: Call search_github_repository.\n"
        "3. For questions about Slack messages, channels, discussions, or team chat: Call search_slack_messages.\n"
        "4. For multi-source investigations involving multiple systems (e.g., Jira, GitHub, and Slack): Call each corresponding tool to collect data across all requested systems.\n"
        "5. After receiving tool results, provide an accurate, factual summary based strictly on the returned data. "
        "Distinguish clearly between confirmed evidence and searches that returned no matching results. Correlate items only when supported by issue keys, commit hashes, or explicit references. Do not invent relationships or assume unverified connections."
    )
)



def research_llm_node(state: ResearchState):
    messages = [SYSTEM_PROMPT] + list(state["messages"])
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
