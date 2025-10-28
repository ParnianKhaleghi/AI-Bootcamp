from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("AVALAI_API_KEY")

def show_all_chroma_data(persist_directory: str, collection_name: str):
    """
    Shows all data stored in a Chroma collection.
    
    Args:
        persist_directory (str): Path where Chroma DB is saved.
        collection_name (str): Name of the Chroma collection.
    """
    # Initialize Chroma client (reconnect to existing DB)
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=None  # not needed for reading
    )

    # Access the underlying raw collection
    collection = db._collection  # private, but fine for debugging/inspection

    # Get all stored data (ids, documents, metadatas, embeddings)
    data = collection.get()

    print(f"📦 Collection name: {collection_name}")
    print(f"📁 Persist directory: {persist_directory}")
    print(f"🧮 Number of records: {len(data['ids'])}")
    print("=" * 60)

    for i, doc_id in enumerate(data["ids"]):
        print(f"🆔 ID: {doc_id}")
        print(f"📄 Document: {data['documents'][i]}")
        print(f"🧾 Metadata: {data['metadatas'][i]}")
        print("-" * 60)

    return data  # Optional: return dictionary for programmatic use


# Initialize embeddings and vectorstore
embedding_function = OpenAIEmbeddings(
    api_key=api_key,
    base_url="https://api.avalai.ir/v1",
    model="text-embedding-3-small")

# Create a local persistent Chroma database (or set persist_directory=None for in-memory)
vectorstore = Chroma(
    collection_name="sample_collection",
    embedding_function=embedding_function,
    persist_directory="./chroma_langchain_db"
)

# Insert some sample data
texts = [
    "LangChain is a framework for developing applications powered by large language models.",
    "Chroma is a vector database used for storing and retrieving embeddings.",
    "OpenAI provides models for text generation and embeddings."
]

metadatas = [
    {"source": "langchain_docs"},
    {"source": "chroma_docs"},
    {"source": "openai_docs"}
]

ids = ["doc1", "doc2", "doc3"]

vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
print("Added 3 documents.")

show_all_chroma_data(
    persist_directory="./chroma_langchain_db",
    collection_name="sample_collection"
)

# Update one document (delete old + reinsert with same ID)
vectorstore.delete(ids=["doc2"])
vectorstore.add_texts(
    texts=["Chroma is an open-source embedding database that integrates with LangChain."],
    metadatas=[{"source": "chroma_updated"}],
    ids=["doc2"]
)
print("Updated doc2.")

show_all_chroma_data(
    persist_directory="./chroma_langchain_db",
    collection_name="sample_collection"
)

# Delete one document
vectorstore.delete(ids=["doc3"])
print("Deleted doc3.")

show_all_chroma_data(
    persist_directory="./chroma_langchain_db",
    collection_name="sample_collection"
)

# Query to check the current state
query = "What is Chroma?"
results = vectorstore.similarity_search(query, k=5)
print("\nSearch results:")
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)
