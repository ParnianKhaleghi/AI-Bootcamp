from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# --- Load environment variables ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_MODEL = os.getenv("API_MODEL")

# --- Connect to OpenRouter ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Conversation history ---
conversation_history = [
    {"role": "system", "content": "You are a friendly chatbot that remembers the conversation."}
]

# JSON file to save history
HISTORY_FILE = "chat_history.json"

def save_history():
    """Save conversation history to a JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2, ensure_ascii=False)

def chat_with_memory(user_input):
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Get model response
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=conversation_history,
        max_tokens=200
    )

    # Extract assistant reply
    reply = response.choices[0].message.content.strip()

    # Add assistant reply to history
    conversation_history.append({"role": "assistant", "content": reply})

    # Save the full history to JSON
    save_history()

    return reply

# ---------- Chat Loop ----------
print("Chatbot with Memory (type 'quit' to exit)\n")
if __name__ == "__main__":
    while True:
        user_message = input("You: ")
        if user_message.lower() == "quit":
            print("Chat ended.")
            break

        bot_reply = chat_with_memory(user_message)
        print("Bot:", bot_reply)
