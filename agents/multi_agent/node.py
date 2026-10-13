from langgraph.prebuilt import ToolNode

from agents.multi_agent.tools import search_issues

tool_node = ToolNode([search_issues])