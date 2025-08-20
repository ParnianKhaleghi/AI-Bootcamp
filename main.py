from openai import OpenAI
import os
from dotenv import load_dotenv
import os

# --- Load .env file ---
load_dotenv()

# --- Access the variable ---
API_MODEL = os.getenv("API_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# connect to OpenRouter
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

def summarize_text(text):
    response = client.chat.completions.create(
        model= API_MODEL,  
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize this:\n\n{text}"}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Example usage
article = """Artificial Intelligence is transforming industries...
It can automate tasks, improve decision making,
and create personalized user experiences."""
print(summarize_text(article))


if __name__ == "__main__":
  while True:
      user_input = input("\nPaste text (or 'quit'): ")
      if user_input.lower() == "quit":
          break
      print("\nSummary:")
      print(summarize_text(user_input))
