import os
from dotenv import load_dotenv
from langchain_core.tools import retriever
load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── CONFIG ──────────────────────────────────────────────────────────
PERSIST_DIR    = "./chroma_db"
COLLECTION     = "knowledge_base"
CHUNK_SIZE     = 300
CHUNK_OVERLAP  = 60



embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

if os.path.exists(PERSIST_DIR):
    print("Loading existing vector store...")
    vectorstore = Chroma(
        collection_name    = COLLECTION,
        embedding_function = embeddings,
        persist_directory  = PERSIST_DIR
    )
    print("Loaded existing vector store from disk")
    print(f"Total documents stored: {vectorstore._collection.count()}")
else:
    # First time — embed and save
    print("Building vector store from scratch...")
     
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )

    chunks =splitter.split_documents(documents)
    print(f"  Chunks   : {len(chunks)}")

    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        collection_name   = COLLECTION,
        persist_directory = PERSIST_DIR
    )
    print(f"  Stored   : {vectorstore._collection.count()} vectors")

print("\nVector store ready!\n")

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {
        "k": 2
    }
)

test_queries = [
    "How do I reset my device?",
    "What is the battery life?",
    "Is water damage covered?",
    "What screen technology does it use?"
]

for query in test_queries:
    print(f"Q: {query}")
    results = retriever.invoke(query)
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content[:100]}...")
    print()

print("RAG pipeline complete!")