from typing import Annotated, Sequence
from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchToolCall(TypedDict):
    name: str
    args: dict[str, str]


class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    source_search_plan: NotRequired[list[ResearchToolCall]]
