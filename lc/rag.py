# --- Imports ---
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("AVALAI_API_KEY")

# --- Website URLs to Scrape ---
URLS_DICTIONARY = {
    "w3schools":"https://www.w3schools.com/js/js_objects.asp"
}
COLLECTION_NAME = "w3schools"

# --- Step 1: Load Documents from Web ---
documents = []
for url in list(URLS_DICTIONARY.values()):
    loader = WebBaseLoader(url)
    data = loader.load()
    documents += data

# --- Step 2: Clean up whitespace ---
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split())

# --- Step 3: Split text into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# --- Step 4: Create OpenAI embeddings ---
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    base_url="https://api.avalai.ir/v1",
    model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory="./chroma_db"
)
vectorstore.persist()

# --- Step 5: Create Chroma Vector Store ---
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name=COLLECTION_NAME)
retriever = vectorstore.as_retriever()

# --- Step 6: Create LLM (ChatGPT model) ---
llm = ChatOpenAI(
    base_url="https://api.avalai.ir/v1",
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.7)

# --- Step 7: Prompt Template ---
template = """Generate a summary of the context that answers the question. 
Explain the answer in multiple steps if possible. 
Answer style should match the context. Ideal Answer Length 2-3 sentences.
if you don't know the answer or it's not in the context, just say that you don't know.

{context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Step 8: Define RAG Chain ---
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Step 9: Ask a question ---
result = rag_chain.invoke("tell me about js objectss")
print(result)
