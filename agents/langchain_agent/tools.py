from langchain_core.tools import tool


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Performs basic arithmetic: add, subtract, multiply, divide."""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError(f"Unsupported operation: {operation}")


@tool
def unit_converter(conversion: str, value: float) -> float:
    """
    Converts temperature and distance.
    
    Supported conversions:
    celsius_to_fahrenheit
    fahrenheit_to_celsius
    km_to_miles
    miles_to_km
    """

    if conversion == "celsius_to_fahrenheit":
        return (value * 9 / 5) + 32

    if conversion == "fahrenheit_to_celsius":
        return (value - 32) * 5 / 9

    if conversion == "km_to_miles":
        return value * 0.621371

    if conversion == "miles_to_km":
        return value * 1.60934

    raise ValueError(f"Unsupported conversion: {conversion}")


@tool
def text_analyzer(text: str) -> dict:
    """Analyzes text and returns word, character, and sentence counts."""

    words = text.split()
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    return {
        "word_count": len(words),
        "character_count": len(text),
        "sentence_count": len(sentences),
    }