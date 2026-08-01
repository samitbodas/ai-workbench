"""
Lab 1C: AI Workbench CLI v1
A menu-driven CLI application with 4 AI tasks.
Demonstrates: separation of prompts from logic, function reuse, user interaction loop.

Architecture:
  User input → Task selection → Prompt template → LLM API → Formatted response

This same architecture becomes a REST API in Week 2.
"""

"""
Cloud computing is the delivery of computer services - including data storage, servers, databases, networking, and software - over the internet rather than a local hard drive. Instead of buying and maintaining physical computers or data centers, people and companies rent access to these tools from remove providers.
This setup lets users reach their files and apps from any device with an internet connection. IT saves money because you only pay for what you use. It also helps teams grow their systems quickly and work together easily from anywhere in the world.
"""


from anthropic import Anthropic, AuthenticationError, APIConnectionError
from dotenv import load_dotenv
import time
import os
import sys

# Ensure box-drawing characters render on Windows terminals (cp1252 default).
# Harmless on macOS/Linux, which are already UTF-8. Guarded for older Pythons.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- PROMPT TEMPLATES ---
# Each task = a different system prompt. Adding a new AI capability = adding an entry here.
TASKS = {
    "1": {
        "name": "Summarize",
        "prompt": "You are a concise summarizer. Summarize the user's text in 3-5 clear bullet points. Focus on the most important information.",
    },
    "2": {
        "name": "Rewrite",
        "prompt": "You are a professional editor. Rewrite the user's text in a clear, professional tone. Maintain the original meaning but improve clarity and readability.",
    },
    "3": {
        "name": "Key Points",
        "prompt": "You are an analyst. Extract the key points from the user's text as a numbered list. Each point should be one clear sentence.",
    },
    "4": {
        "name": "Explain",
        "prompt": "You are a patient teacher. Explain the user's text in simple terms that a non-expert can understand. Use analogies where helpful.",
    },
}


def call_llm(system_prompt: str, user_text: str) -> dict:
    """
    Call the LLM with a system prompt and user text.
    Returns dict with 'content', 'tokens', and 'model' keys.
    Handles errors gracefully with specific messages.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            system=system_prompt,

            messages=[
                #                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return {
            "content": response.content[0].text,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "model": response.model,
        }
    except AuthenticationError:
        return {"content": "Error: Invalid API key. Check your .env file.", "tokens": 0, "model": "N/A"}
    except RateLimitError:
        return {"content": "Error: Rate limit hit. Wait a moment and try again.", "tokens": 0, "model": "N/A"}
    except APIConnectionError:
        return {"content": "Error: Cannot connect to OpenAI. Check your internet.", "tokens": 0, "model": "N/A"}
    except Exception as e:
        return {"content": f"Error: {e}", "tokens": 0, "model": "N/A"}


def display_menu():
    print("\n┌──────────────────────────────────────┐")
    print("│        AI WORKBENCH CLI v1           │")
    print("├──────────────────────────────────────┤")
    for key, task in TASKS.items():
        print(f"│  {key}. {task['name']:<32}│")
    print("├──────────────────────────────────────┤")
    print("│  q. Quit                             │")
    print("└──────────────────────────────────────┘")


def get_user_text() -> str:
    print("\n  Paste your text below.")
    print("  (Press Enter on an empty line to submit)")
    print("  " + "─" * 36)
    lines = []
    while True:
        try:
            line = input("  │ ")
        except EOFError:
            break
        if line == "":
            if lines:
                break
        else:
            lines.append(line)
    return "\n".join(lines)


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("┌──────────────────────────────────────┐")
        print("│  ERROR: ANTHROPIC_API_KEY not set        │")
        print("│                                       │")
        print("│  1. Copy .env.example → .env          │")
        print("│  2. Add your API key                  │")
        print("│  3. Run again                         │")
        print("└──────────────────────────────────────┘")
        sys.exit(1)

    print("\n  Welcome to AI Workbench!")
    print(f"  Model: {MODEL}")
    print(f"  Tasks available: {len(TASKS)}")

    session_tokens = 0

    while True:
        display_menu()
        choice = input("\n  Select a task (1-4 or q): ").strip()

        if choice.lower() == "q":
            print(f"\n  Session total: {session_tokens} tokens used")
            print("  Goodbye!\n")
            break

        if choice not in TASKS:
            print("  Invalid choice. Enter 1-4 or q.")
            continue

        task = TASKS[choice]
        print(f"\n  → Task: {task['name']}")
        user_text = get_user_text()

        if not user_text.strip():
            print("  No text provided. Try again.")
            continue

        print(f"\n  Processing ({task['name']})...")
        result = call_llm(task["prompt"], user_text)

        print(f"\n  {'━' * 38}")
        print(f"  Result ({task['name']}):")
        print(f"  {'━' * 38}")
        print()
        for line in result["content"].split("\n"):
            print(f"  {line}")
        print()
        print(f"  {'━' * 38}")
        print(f"  Tokens: {result['tokens']} | Model: {result['model']}")
        print(f"  {'━' * 38}")

        session_tokens += result["tokens"]


if __name__ == "__main__":
    main()
