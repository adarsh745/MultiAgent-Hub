from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agents.multi_agent.state import MultiAgentState
from agents.multi_agent.tools import search_issues
from agents.multi_agent.research_graph import research_graph
from agents.multi_agent.analysis_graph import analysis_graph
from agents.multi_agent.verification import verify_analysis_claims

model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
)


def research_agent(state: MultiAgentState):
    request = state["user_request"]

    result = research_graph.invoke({
        "messages": [HumanMessage(content=request)]
    })

    final_message = result["messages"][-1]

    return {
        "research_result": final_message.content
    }


def analysis_agent(state: MultiAgentState):
    research = state["research_result"]

    result = analysis_graph.invoke({
        "messages": [HumanMessage(content=f"Analyze the following research result:\n\n{research}")]
    })

    final_message = result["messages"][-1]

    return {
        "analysis_result": final_message.content
    }


def verification_agent(state: MultiAgentState):
    research = state.get("research_result", "")
    analysis = state.get("analysis_result", "")

    prompt = f"""You are a strict Verification Agent.

Your task is to verify that each claim in the Analysis is strictly supported by the Research result.

Research Result:
{research}

Analysis to Verify:
{analysis}

Rules:
1. Verify that the analysis accurately reflects the research result without making unsupported claims.
2. If the search returned no issues, ensure the analysis does NOT claim the system has 'high quality', 'high stability', or that no bugs exist across the entire project.
3. Remove any unsupported claims, speculation, and repetitive wording.
4. Provide a concise, factual verified conclusion.
"""

    response = model.invoke(prompt)

    # Perform claim-by-claim verification and remove unsupported claims
    is_valid, verified_text = verify_analysis_claims(
        research=research,
        analysis=analysis,
        llm_conclusion=response.content,
    )

    return {
        "verification_result": verified_text,
        "final_response": verified_text,
    }


def supervisor_agent(state: MultiAgentState):
    research = state["research_result"]
    analysis = state["analysis_result"]
    verification = state["verification_result"]

    if not research:
        next_agent = "research"

    elif not analysis:
        next_agent = "analysis"

    elif not verification:
        next_agent = "verification"

    else:
        next_agent = "finish"

    return {
        "next_agent": next_agent
    }