from lc.openrouter_llm import OpenrouterChatLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# --- Initialize model ---
model = OpenrouterChatLLM(api_key=api_key)

# --- Use message list ---
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

response = model.invoke(messages)
print("Response from messages:")
print(response)

# Use ChatPromptTemplate ---
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print("\nGenerated prompt:")
print(prompt.to_messages())

response = model.invoke(prompt)
print("\nResponse from prompt:")
print(response)
