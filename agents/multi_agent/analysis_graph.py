import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_local_dir = str(Path(__file__).resolve().parent)
if _local_dir in sys.path:
    sys.path.remove(_local_dir)

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from agents.multi_agent.analysis_state import AnalysisState
from agents.multi_agent.verification import format_factual_analysis

# Initialize ChatOllama model with qwen2.5:1.5b
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an expert Analysis Agent for software issues.\n\n"
        "Strict Analysis Rules:\n"
        "1. Base your analysis STRICTLY on the provided research findings. State only facts directly supported by the research.\n"
        "2. If the search returned NO matching issues:\n"
        "   - State clearly that the query returned no matching issues.\n"
        "   - Explicitly state that this result is limited to the search performed and does NOT establish that no such bugs exist outside the search scope or across the entire project.\n"
        "   - CRITICAL: NEVER infer or claim that the system has 'high quality', 'high stability', 'is well maintained', 'meets highest standards', or that 'no further action is needed'. An empty search result only reflects the query executed.\n"
        "3. If matching issues ARE present in the research:\n"
        "   - Clearly categorize open vs. resolved issues.\n"
        "   - State the priority/severity and key summary of each identified issue.\n"
        "4. Keep the analysis concise, factual, and free of repetitive text."
    )
)


def analysis_llm_node(state: AnalysisState):
    messages = [SYSTEM_PROMPT] + list(state["messages"])
    response = model.invoke(messages)

    # Get research text from the human message
    research_text = state["messages"][-1].content if state["messages"] else ""
    cleaned_content = format_factual_analysis(research_text, response.content)
    return {"messages": [AIMessage(content=cleaned_content)]}


builder = StateGraph(AnalysisState)

# Add Analysis LLM node and flow: START -> Analysis LLM -> END
builder.add_node("analysis_llm", analysis_llm_node)

builder.add_edge(START, "analysis_llm")
builder.add_edge("analysis_llm", END)

analysis_graph = builder.compile()
