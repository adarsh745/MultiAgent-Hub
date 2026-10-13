import sys
from pathlib import Path

# Ensure workspace root and backend directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "backend"))

import asyncio
from typing import Any, List, Optional, Dict
from agents.assistant_agent import AssistantAgent
from agents.tools.calculator import CalculatorTool
from agents.tools.unit_converter import UnitConverterTool
from agents.tools.text_analyzer import TextAnalyzerTool
from agents.tools.registry import ToolRegistry
from backend.app.services.llm.ollama_service import OllamaLLMService


class MockLLMService:
    """Mock LLM service for simulating specific LLM outputs (e.g. invalid JSON, bad schema, or forced tool selection)."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return '{"action": "final", "response": "Fallback response"}'


async def main():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(UnitConverterTool())
    registry.register(TextAnalyzerTool())

    llm_service = OllamaLLMService()
    agent = AssistantAgent(name="assistant", tool_registry=registry, llm_service=llm_service)

    # Define test suite (25 cases including forced relevance test)
    test_specs = [
        # CALCULATOR (1-4)
        (1, "CALCULATOR", "Calculate 25 × 40", "calculator", "multiply", None),
        (2, "CALCULATOR", "Add 10 and 20", "calculator", "add", None),
        (3, "CALCULATOR", "Divide 100 by 4", "calculator", "divide", None),
        (4, "CALCULATOR", "Subtract 25 from 100", "calculator", "subtract", None),

        # UNIT CONVERTER (5-12)
        (5, "UNIT CONVERTER", "Convert 25 degrees Celsius to Fahrenheit", "unit_converter", "celsius_to_fahrenheit", None),
        (6, "UNIT CONVERTER", "Convert 77 degrees Fahrenheit to Celsius", "unit_converter", "fahrenheit_to_celsius", None),
        (7, "UNIT CONVERTER", "Convert 10 kilometers to miles", "unit_converter", "km_to_miles", None),
        (8, "UNIT CONVERTER", "Convert 15 miles to kilometers", "unit_converter", "miles_to_km", None),
        (9, "UNIT CONVERTER", "Convert 100 degrees Celsius to Fahrenheit", "unit_converter", "celsius_to_fahrenheit", None),
        (10, "UNIT CONVERTER", "Convert 98.6 degrees Fahrenheit to Celsius", "unit_converter", "fahrenheit_to_celsius", None),
        (11, "UNIT CONVERTER", "Convert 42.195 kilometers to miles", "unit_converter", "km_to_miles", None),
        (12, "UNIT CONVERTER", "Convert 15 miles to kilometers", "unit_converter", "miles_to_km", None),

        # # TEXT ANALYZER (13-15)
        # (13, "TEXT ANALYZER", "Analyze this text: Hello World", "text_analyzer", None, None),
        # (14, "TEXT ANALYZER", "Count the words in this sentence: WorkPilot AI empowers enterprise productivity.", "text_analyzer", None, None),
        # (15, "TEXT ANALYZER", "Analyze this paragraph: Python is great. AI agents make development fast.", "text_analyzer", None, None),

        # # NO TOOL (16-19)
        # (16, "NO TOOL", "What is FastAPI?", None, None, None),
        # (17, "NO TOOL", "What is PostgreSQL?", None, None, None),
        # (18, "NO TOOL", "Explain REST API in simple terms", None, None, None),
        # (19, "NO TOOL", "What is JWT?", None, None, None),

        # # EDGE CASES & RECOVERY (20-24)
        # (20, "EDGE CASE", "Run database query using sql_executor tool", None, None, [
        #     '{"action": "tool", "tool": "sql_executor", "arguments": {"query": "SELECT 1"}}',
        #     '{"action": "final", "response": "I do not have access to a database query tool."}'
        # ]),
        # (21, "EDGE CASE", "Calculate 50 divided by 2 using calculator", "calculator", "divide", [
        #     '{"action": "tool", "tool": "calculator", "arguments": {"operation": "modulus", "a": 50, "b": 2}}',
        #     '{"action": "tool", "tool": "calculator", "arguments": {"operation": "divide", "a": 50, "b": 2}}',
        #     'Final answer is 25'
        # ]),
        # (22, "EDGE CASE", "Convert distance", "unit_converter", "km_to_miles", [
        #     '{"action": "tool", "tool": "unit_converter", "arguments": {"conversion": "km_to_miles"}}',
        #     '{"action": "tool", "tool": "unit_converter", "arguments": {"conversion": "km_to_miles", "value": 10}}',
        #     '10 km is 6.21 miles'
        # ]),
        # (23, "EDGE CASE", "What is 5 times 5?", "calculator", "multiply", [
        #     'Not JSON response at all',
        #     '{"action": "tool", "tool": "calculator", "arguments": {"operation": "multiply", "a": 5, "b": 5}}',
        #     '25'
        # ]),
        # (24, "EDGE CASE", "Multiply 10 by 10", "calculator", "multiply", [
        #     '{"action": "calculator", "operation": "multiply", "a": 10, "b": 10}',
        #     '{"action": "tool", "tool": "calculator", "arguments": {"operation": "multiply", "a": 10, "b": 10}}',
        #     '100'
        # ]),

        # # FORCED RELEVANCE GUARD TEST (25)
        # (25, "RELEVANCE GUARD", "What is JWT?", None, None, [
        #     '{"action": "tool", "tool": "unit_converter", "arguments": {"conversion": "celsius_to_fahrenheit", "value": 0}}',
        #     '{"action": "final", "response": "JWT stands for JSON Web Token. It is a compact, URL-safe means of representing claims between two parties."}'
        # ]),
    ]

    total_tests = len(test_specs)
    passed_tests = 0
    failed_tests = 0
    tool_correct_count = 0
    op_correct_count = 0
    arg_validation_failures = 0
    relevance_errors_caught = 0

    print("==================================================")
    print("      WORKPILOT AI TOOL EVALUATION SUITE          ")
    print("==================================================")

    for test_id, category, user_prompt, expected_tool, expected_op, mock_responses in test_specs:
        print(f"\n--------------------------------------------------")
        print(f"Test #{test_id} [{category}]: {user_prompt}")
        print(f"Expected Tool: {expected_tool} | Expected Op: {expected_op}")

        if mock_responses:
            test_agent = AssistantAgent(name="assistant", tool_registry=registry, llm_service=MockLLMService(mock_responses))
        else:
            test_agent = agent

        try:
            final_response = await test_agent.run(user_prompt)
            trace = test_agent.last_execution

            actual_tool = trace.get("tool_name")
            actual_op = trace.get("operation")
            actual_action = trace.get("decision")
            validation_errors = trace.get("validation_errors", [])
            relevance_errors = trace.get("relevance_errors", [])
            retries = trace.get("attempts", 1) - 1

            if len(validation_errors) > 0:
                arg_validation_failures += len(validation_errors)
            if len(relevance_errors) > 0:
                relevance_errors_caught += len(relevance_errors)

            tool_match = (actual_tool == expected_tool) or (expected_tool is None and actual_action == "final")
            op_match = (expected_op is None) or (actual_op == expected_op)

            if tool_match:
                tool_correct_count += 1
            if op_match and tool_match:
                op_correct_count += 1

            is_pass = tool_match and op_match and (trace.get("error") is None)

            if is_pass:
                passed_tests += 1
                status = "PASS"
            else:
                failed_tests += 1
                status = "FAIL"

            print(f"Actual Tool: {actual_tool} | Actual Op: {actual_op} | Action: {actual_action}")
            print(f"Retries Triggered: {retries} | Validation Errors: {len(validation_errors)} | Relevance Errors: {len(relevance_errors)}")
            print(f"Final Output: {final_response[:100]}..." if len(str(final_response)) > 100 else f"Final Output: {final_response}")
            print(f"Status: [{status}]")

        except Exception as e:
            failed_tests += 1
            print(f"EXCEPTION: {e}")
            print("Status: [FAIL]")

    tool_accuracy = (tool_correct_count / total_tests) * 100
    op_accuracy = (op_correct_count / total_tests) * 100
    overall_accuracy = (passed_tests / total_tests) * 100

    print("\n==================================================")
    print("               EVALUATION SUMMARY                 ")
    print("==================================================")
    print(f"Total Tests:                 {total_tests}")
    print(f"Passed:                      {passed_tests}")
    print(f"Failed:                      {failed_tests}")
    print(f"Tool Selection Accuracy:     {tool_accuracy:.2f}%")
    print(f"Operation Selection Accuracy:{op_accuracy:.2f}%")
    print(f"Argument Validation Errors:  {arg_validation_failures}")
    print(f"Relevance Errors Caught:     {relevance_errors_caught}")
    print(f"Overall Test Pass Accuracy:  {overall_accuracy:.2f}%")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())