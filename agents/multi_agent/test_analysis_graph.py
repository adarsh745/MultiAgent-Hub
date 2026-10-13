import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_local_dir = str(Path(__file__).resolve().parent)
if _local_dir in sys.path:
    sys.path.remove(_local_dir)

from langchain_core.messages import HumanMessage
from agents.multi_agent.analysis_graph import analysis_graph

sample_research_result = """Here are the payment-related bugs found:

1. **BUG-101**: Payment timeout during checkout - High priority, open issue.
2. **BUG-102**: Payment callback failure - High priority, resolved issue.
3. **BUG-103**: Payment retry creates duplicate transaction - Critical priority, open issue."""


def main():
    print("=== Input Research Result ===")
    print(sample_research_result)

    print("\n--- Invoking Analysis Subgraph ---")
    result = analysis_graph.invoke({
        "messages": [HumanMessage(content=f"Analyze the following research result:\n\n{sample_research_result}")]
    })

    print("\n=== Analysis Response ===")
    final_response = result["messages"][-1]
    print(final_response.content)


if __name__ == "__main__":
    main()
