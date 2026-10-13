from langgraph.graph import StateGraph, START, END

from agents.multi_agent.state import MultiAgentState

from agents.multi_agent.agents import (
    supervisor_agent,
    research_agent,
    analysis_agent,
    verification_agent,
)


builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("research", research_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("verification", verification_agent)

builder.add_edge(START, "supervisor")


def route_from_supervisor(state: MultiAgentState):
    return state["next_agent"]


builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "research",
        "analysis": "analysis",
        "verification": "verification",
        "finish": END,
    },
)


builder.add_edge("research", "supervisor")
builder.add_edge("analysis", "supervisor")
builder.add_edge("verification", "supervisor")


graph = builder.compile()