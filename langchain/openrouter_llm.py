from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from typing import Optional, List
import requests
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

secret_key = os.getenv("OPENROUTER_API_KEY")
api_model = os.getenv("API_MODEL", "gpt-4o-mini")  # fallback model

print("API key:", secret_key)

class OpenrouterChatLLM(LLM):
    """Custom LangChain-compatible LLM for OpenRouter.ai's chat endpoint."""

    api_key: str
    model: str = api_model
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"

    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
    # Handle both string and list[BaseMessage]
      if isinstance(prompt, list):
          # Convert messages to OpenRouter format
          messages = [{"role": m.type, "content": m.content} for m in prompt]
      else:
          messages = [{"role": "user", "content": str(prompt)}]

      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self.api_key}",
      }

      payload = {
          "model": self.model,
          "messages": messages,
      }

      response = requests.post(self.endpoint, headers=headers, json=payload)

      if response.status_code != 200:
          raise ValueError(f"OpenRouter API returned {response.status_code}: {response.text}")

      result = response.json()

      try:
          return result["choices"][0]["message"]["content"].strip()
      except (KeyError, IndexError):
          raise ValueError(f"Unexpected API response format: {result}")


    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"


if __name__ == "__main__":
    llm = OpenrouterChatLLM(api_key=secret_key)

    prompt = PromptTemplate.from_template(
        "Write a one-sentence bedtime story about a {animal}."
    )
    formatted_prompt = prompt.format(animal="unicorn")

    result = llm.invoke(formatted_prompt)
    print(result)
