import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_local_dir = str(Path(__file__).resolve().parent)
if _local_dir in sys.path:
    sys.path.remove(_local_dir)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.multi_agent.research_graph import research_graph


def main():
    user_query = "Find payment bugs"
    print(f"--- Running Research Subgraph with Query: '{user_query}' ---")

    result = research_graph.invoke({
        "messages": [HumanMessage(content=user_query)]
    })

    print("\n=== All Messages ===")
    for msg in result["messages"]:
        print(f"\n[{msg.type.upper()} MESSAGE]:")
        print(msg.content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Tool Calls: {msg.tool_calls}")

    print("\n=== Tool Calls ===")
    tool_calls_found = False
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_found = True
            for tc in msg.tool_calls:
                print(f"Tool: {tc['name']}")
                print(f"Arguments: {tc['args']}")
    if not tool_calls_found:
        print("No tool calls were made.")

    print("\n=== Tool Result ===")
    tool_results_found = False
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results_found = True
            print(f"Tool Result ({getattr(msg, 'name', 'tool')}):")
            print(msg.content)
    if not tool_results_found:
        print("No tool results received.")

    print("\n=== Final LLM Response ===")
    final_response = result["messages"][-1]
    if isinstance(final_response, AIMessage):
        print(final_response.content)
    else:
        print("No final response generated.")


if __name__ == "__main__":
    main()
