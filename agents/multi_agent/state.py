from typing import TypedDict


class MultiAgentState(TypedDict):
    user_request: str

    research_result: str
    analysis_result: str
    verification_result: str

    next_agent: str
    final_response: str