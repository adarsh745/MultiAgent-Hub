from agents.langchain_agent.agent import agent


def run_test(message: str):
    print("\n" + "=" * 60)
    print(f"USER: {message}")
    print("=" * 60)

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        }
    )

    for msg in response["messages"]:
        print(f"\nTYPE: {type(msg).__name__}")

        if hasattr(msg, "content") and msg.content:
            print("CONTENT:", msg.content)

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("TOOL CALLS:", msg.tool_calls)


def main():
    run_test("Calculate 25 multiplied by 40")
    run_test("Convert 15 miles to kilometers")
    run_test(
        "Analyze this text: "
        "Artificial intelligence is changing software development. "
        "Developers are using AI agents."
    )


if __name__ == "__main__":
    main()