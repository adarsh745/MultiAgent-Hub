from agents.langgraph_agent.graph import graph


def run_test(question: str):
    print("\n" + "=" * 60)
    print(f"USER: {question}")
    print("=" * 60)

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    for message in result["messages"]:
        print(f"\n{type(message).__name__}:")

        if isinstance(message, dict):
            print(message.get("content"))
        else:
            if message.content:
                print(message.content)

            if getattr(message, "tool_calls", None):
                print("TOOL CALLS:", message.tool_calls)


def main():
    run_test("Calculate 25 multiplied by 40")

    run_test("Convert 15 miles to kilometers")

    run_test(
        "Analyze this text: "
        "Artificial intelligence is changing software development. "
        "Developers are using AI agents."
    )

    run_test("What is an API?")


if __name__ == "__main__":
    main()