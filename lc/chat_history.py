from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("AVALAI_API_KEY")

# Step 1: Initialize model
llm = ChatOpenAI(
  base_url="https://api.avalai.ir/v1",
  api_key=api_key,
  model="gpt-4o-mini")

# Step 2: Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Step 3: Build conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# Step 4: Continuous chat loop
print("Chat started! Type 'end' to finish.\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "end":
        print("👋 Chat ended.")
        break
    response = conversation.invoke(user_input)["response"]
    print("Assistant:", response)
