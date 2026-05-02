from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

print("Building FAISS vector store with Gemini embeddings...")
faiss_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

print(f"FAISS store built successfully")

results = faiss_store.similarity_search(
    query="How do I reset my device?",
    k=2
)

for doc in results:
    print(f" -> {doc.page_content}")