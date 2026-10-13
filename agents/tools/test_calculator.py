from calculator import CalculatorInput, CalculatorTool


calculator = CalculatorTool()


# Addition
result = calculator.execute(
    CalculatorInput(
        operation="add",
        a=10,
        b=5
    )
)

print("10 + 5 =", result)


# Subtraction
result = calculator.execute(
    CalculatorInput(
        operation="subtract",
        a=10,
        b=5
    )
)

print("10 - 5 =", result)


# Multiplication
result = calculator.execute(
    CalculatorInput(
        operation="multiply",
        a=125,
        b=48
    )
)

print("125 × 48 =", result)


# Division
result = calculator.execute(
    CalculatorInput(
        operation="divide",
        a=100,
        b=4
    )
)

print("100 / 4 =", result)