from typing import Literal

from pydantic import BaseModel, Field

from agents.tools.base import BaseTool


class CalculatorInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class CalculatorTool(BaseTool):

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Performs arithmetic calculations such as addition, subtraction, "
            "multiplication, and division. Use this tool ONLY when the user asks to calculate numbers."
        )

    @property
    def input_schema(self):
        return CalculatorInput

    def execute(self, input_data: CalculatorInput) -> float:

        if input_data.operation == "add":
            return input_data.a + input_data.b

        if input_data.operation == "subtract":
            return input_data.a - input_data.b

        if input_data.operation == "multiply":
            return input_data.a * input_data.b

        if input_data.operation == "divide":
            if input_data.b == 0:
                raise ValueError("Cannot divide by zero.")

            return input_data.a / input_data.b

        raise ValueError(
            f"Unsupported operation: {input_data.operation}"
        )