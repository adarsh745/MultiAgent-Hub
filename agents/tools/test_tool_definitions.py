from agents.tools.calculator import CalculatorTool
from agents.tools.registry import ToolRegistry


registry = ToolRegistry()

calculator = CalculatorTool()

registry.register(calculator)

definitions = registry.get_tool_definitions()

print("Tool definitions:")
print(definitions)