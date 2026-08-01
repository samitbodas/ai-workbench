import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic, AuthenticationError, APIConnectionError

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found!")
    print("Copy .env.example to .env and add your key.")
    sys.exit()

client = Anthropic(api_key=api_key)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
print(MODEL)

# -- the call --

print("Sending prompt: 'What is generative AI in one sentence?'")

try:
    response = client.messages.create(
        model=MODEL,
        system="You are a helpful assistant.",

        messages=[
            #            {"role": "system", "content": "You are a helpful assistant. Be concise"},
            {"role": "user", "content": "What is generative AI in once sentence?"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

# response + errors

#    print(response.content[0].text)
    print(f"Response: {response.content[0].text}")
#    print(response.usage.input_tokens)
    print(
        f"Tokens Used: {response.usage.input_tokens + response.usage.output_tokens}")

except AuthenticationError:
    print("Error: Invalid API key. Check your .env file.")
except APIConnectionError:
    print("Error: Cannot connect. Check your internet.")
except Exception as e:
    print("Unexpected Error: {e}")
