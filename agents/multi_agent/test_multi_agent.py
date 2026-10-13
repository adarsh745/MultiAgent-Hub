import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_local_dir = str(Path(__file__).resolve().parent)
if _local_dir in sys.path:
    sys.path.remove(_local_dir)

from agents.multi_agent.graph import graph

print("=== Graph Mermaid Diagram ===")
print(graph.get_graph().draw_mermaid())

print("\n=== Invoking Multi-Agent Graph ===")
result = graph.invoke({
    "user_request": "Find unresolved payment bugs",
    "research_result": "",
    "analysis_result": "",
    "verification_result": "",
    "next_agent": "",
    "final_response": "",
})

print("\n=== MultiAgentState Results ===")
for key, value in result.items():
    print(f"\n[{key.upper()}]:\n{value}")

# Assertions to verify all required outputs are populated
assert result.get("research_result"), "Error: research_result is empty!"
assert result.get("analysis_result"), "Error: analysis_result is empty!"
assert result.get("verification_result"), "Error: verification_result is empty!"
assert result.get("final_response"), "Error: final_response is empty!"

print("\n[SUCCESS] All required outputs (research_result, analysis_result, verification_result, final_response) are populated.")
