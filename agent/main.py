from .agent import build_agent

def main():
    agent = build_agent()

    print("MediCare Follow-Up Agent")
    print("Type 'exit' or 'quit' or 'q'to stop.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]}
            )

            messages = response.get("messages", [])

            if messages:
                print(f"\nAgent:\n{messages[-1].content}")
            else:
                print("\nAgent:\nNo response generated.")

        except Exception as exc:
            print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    main()