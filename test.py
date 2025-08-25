from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# --- Load .env file ---
load_dotenv()

# --- Access the variables ---
API_MODEL = os.getenv("API_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# connect to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initialize conversation history (will also be saved to JSON)
conversation_history = []

def persian_chat(user_text):
    # Append user message to history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Create chat completion
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds only in Persian."},
            *conversation_history  # pass previous messages for context
        ],
        max_tokens=150
    )

    # Extract assistant reply
    reply = response.choices[0].message.content.strip()

    # Append assistant reply to history
    conversation_history.append({"role": "assistant", "content": reply})

    # Save conversation history to JSON
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, ensure_ascii=False, indent=2)

    return reply


# Example usage
article = """زبان فارسی یکی از زیباترین زبان های دنیاست."""
print("Summary:", persian_chat(article))


# ---------- Chat Loop ----------
if __name__ == "__main__":
    print("Persian Chatbot (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chat ended.")
            break
        bot_reply = persian_chat(user_input)
        print("Bot:", bot_reply)
