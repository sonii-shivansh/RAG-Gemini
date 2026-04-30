from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document


embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

documents = [
    Document(
        page_content = "To perform a soft reset, hold the power button for 10 seconds until the screen goes dark.",
        metadata     = {"source": "techpro_manual.pdf", "page": 1, "topic": "reset"}
    ),
    Document(
        page_content = "Factory reset erases all data permanently. Go to Settings > System > Factory Reset.",
        metadata     = {"source": "techpro_manual.pdf", "page": 2, "topic": "reset"}
    ),
    Document(
        page_content = "Battery life is approximately 12 hours under normal usage conditions.",
        metadata     = {"source": "techpro_manual.pdf", "page": 3, "topic": "battery"}
    ),
    Document(
        page_content = "Fast charging supports up to 65 watts. A full charge takes approximately 45 minutes.",
        metadata     = {"source": "techpro_manual.pdf", "page": 3, "topic": "battery"}
    ),
    Document(
        page_content = "The TechPro X500 warranty is valid for 1 year from the original date of purchase.",
        metadata     = {"source": "techpro_manual.pdf", "page": 5, "topic": "warranty"}
    ),
    Document(
        page_content = "Water damage and physical damage void the warranty completely.",
        metadata     = {"source": "techpro_manual.pdf", "page": 5, "topic": "warranty"}
    ),
    Document(
        page_content = "Bluetooth 5.3 supports connections up to 10 meters in open environments.",
        metadata     = {"source": "techpro_manual.pdf", "page": 7, "topic": "connectivity"}
    ),
    Document(
        page_content = "Wi-Fi 6E support enables speeds up to 9.6 Gbps on compatible networks.",
        metadata     = {"source": "techpro_manual.pdf", "page": 7, "topic": "connectivity"}
    ),
]

print("Creating Chroma vector store with Gemini embeddings...")
print("(This embeds all documents — may take a few seconds)")


# Create vector store — this embeds all documents automatically
# You do NOT call embed_documents() yourself anymore
# The vector store handles it internally

# ONE LINE creates the entire vector store
# Internally: calls gemini_embeddings.embed_documents() on all chunks
# Stores vectors + text + metadata in ChromaDB

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    collection_name = "techpro_gemini"
    
)

print(f"\nVector store created!")
print(f"Documents stored: {vectorstore._collection.count()}")

# ── Basic similarity search ──────────────────────────────────────────
# Internally: calls gemini_embeddings.embed_query() on your question
# Then finds closest vectors using cosine similarity

# query = "How do I reset my device?"

# results = vectorstore.similarity_search(
#     query=query,
#     k=2
# )

# print(f"QUERY: '{query}'")
# print(f"Found {len(results)} results\n")

# for i, doc in enumerate(results, 1):
#     print(f"--- Result {i} ---")
#     print(f"Content : {doc.page_content}")
#     print(f"Topic   : {doc.metadata['topic']}")
#     print(f"Page    : {doc.metadata['page']}")
#     print()


# -------------------------------------------------------------------------
# Similarity Search WITH Scores

results_with_scores = vectorstore.similarity_search_with_score(
    query = "How long does the battery last?",
    k     = 3
)

print("QUERY: 'How long does the battery last?'\n")
for doc, score in results_with_scores:
    print(f"Score   : {score:.4f}  ← lower is MORE similar in ChromaDB")
    print(f"Content : {doc.page_content}")
    print(f"Topic   : {doc.metadata['topic']}")
    print()