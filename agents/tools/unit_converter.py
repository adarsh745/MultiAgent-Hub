from typing import Literal

from pydantic import BaseModel, Field

from agents.tools.base import BaseTool


class UnitConverterInput(BaseModel):
    conversion: Literal[
        "celsius_to_fahrenheit",
        "fahrenheit_to_celsius",
        "km_to_miles",
        "miles_to_km",
    ] = Field(
        ...,
        description=(
            "The exact conversion direction: "
            "'celsius_to_fahrenheit' (Celsius -> Fahrenheit), "
            "'fahrenheit_to_celsius' (Fahrenheit -> Celsius), "
            "'km_to_miles' (kilometers -> miles), "
            "'miles_to_km' (miles -> kilometers)."
        ),
    )
    value: float = Field(..., description="The numeric value to convert.")


class UnitConverterTool(BaseTool):

    @property
    def name(self) -> str:
        return "unit_converter"

    @property
    def description(self) -> str:
        return (
            "Converts units. Supports Celsius to Fahrenheit, Fahrenheit to Celsius, "
            "kilometers to miles, and miles to kilometers. Use this tool ONLY when the user explicitly asks to convert units."
        )

    @property
    def input_schema(self):
        return UnitConverterInput

    def execute(self, input_data: UnitConverterInput) -> float:

        if input_data.conversion == "celsius_to_fahrenheit":
            return (input_data.value * 9 / 5) + 32

        if input_data.conversion == "fahrenheit_to_celsius":
            return (input_data.value - 32) * 5 / 9

        if input_data.conversion == "km_to_miles":
            return input_data.value * 0.621371

        if input_data.conversion == "miles_to_km":
            return input_data.value * 1.60934

        raise ValueError("Unsupported conversion.")