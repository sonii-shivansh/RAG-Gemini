import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PERSIST_DIR = "./chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

documents = [
    Document(
        page_content="To reset your device, hold the power button for 10 seconds.",
        metadata={"source": "manual.pdf", "page": 1, "topic": "reset"}
    ),
    Document(
        page_content="Battery life is approximately 12 hours under normal usage.",
        metadata={"source": "manual.pdf", "page": 3, "topic": "battery"}
    ),
    Document(
        page_content="The warranty covers defects in materials for 1 year.",
        metadata={"source": "manual.pdf", "page": 5, "topic": "warranty"}
    ),
    Document(
        page_content="Fast charging supports up to 65 watts with the included adapter.",
        metadata={"source": "manual.pdf", "page": 3, "topic": "battery"}
    ),
    Document(
        page_content="Factory reset will permanently erase all data on the device.",
        metadata={"source": "manual.pdf", "page": 1, "topic": "reset"}
    ),
    Document(
        page_content="Water damage and physical damage are not covered under warranty.",
        metadata={"source": "manual.pdf", "page": 5, "topic": "warranty"}
    ),
]



if os.path.exists(PERSIST_DIR):
    vectorstore = Chroma(
        collection_name    = "device_manual",
        embedding_function = embeddings,
        persist_directory  = PERSIST_DIR
    )
    print("Loaded existing vector store from disk")
    print(f"Total documents stored: {vectorstore._collection.count()}")
else:
    # First time — embed and save
    vectorstore = Chroma.from_documents(
        documents         = documents,
        embedding         = embeddings,
        collection_name   = "device_manual",
        persist_directory = PERSIST_DIR
    )
    print("Created new vector store and saved to disk")


query = "How do I reset my device?"
results = vectorstore.similarity_search(
    query=query,
    k = 3
)

print("\nQUERY: ", query)
print(f"Found {len(results)} results\n")

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Content : {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

