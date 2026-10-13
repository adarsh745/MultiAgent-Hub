from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from agents.langgraph_agent.state import AgentState
from agents.langgraph_agent.nodes import llm_node, tool_node


builder = StateGraph(AgentState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")

builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

builder.add_edge("tools", "llm")

graph = builder.compile()