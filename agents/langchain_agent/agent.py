from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from agents.langchain_agent.tools import calculator,unit_converter,text_analyzer


model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
    format="json",
)

tools = [calculator,unit_converter,text_analyzer]

agent = create_agent(
    model=model,
    tools=tools,
)