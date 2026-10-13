from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from agents.langgraph_agent.state import AgentState
from agents.langchain_agent.tools import calculator, unit_converter, text_analyzer

model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
)

tools = [calculator,unit_converter,text_analyzer]
model_with_tools = model.bind_tools(tools)


def llm_node(state: AgentState) -> AgentState:
    response = model_with_tools.invoke(state["messages"])

    return {
        "messages": state["messages"] + [response]
    }   


tool_node = ToolNode(tools)