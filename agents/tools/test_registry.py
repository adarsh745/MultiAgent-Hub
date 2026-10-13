from agents.tools.calculator import CalculatorTool
from agents.tools.registry import ToolRegistry


registry = ToolRegistry()

calculator = CalculatorTool()

registry.register(calculator)

print("Available tools:")
print(registry.list_tools())

tool = registry.get("calculator")

input_data = tool.input_schema(
    operation="multiply",
    a=10,
    b=20,
)

result = tool.execute(input_data)

print("Calculator result:")
print(result)

print("\nTool definitions:")
print(registry.get_tool_definitions())