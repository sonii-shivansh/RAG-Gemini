from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Step 1: Load----------------
loader    = TextLoader("knowledge_base.txt")
documents = loader.load()
print(f"Documents loaded : {len(documents)}")


# ── STEP 2: Split ───────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 200,
    chunk_overlap = 40
)
chunks = splitter.split_documents(documents)
print(f"Chunks created   : {len(chunks)}")

# ── STEP 3: Embed ───────────────────────────────────────────────────
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

# Embed all chunks
chunk_texts   = [chunk.page_content for chunk in chunks]
chunk_vectors = embeddings_model.embed_documents(chunk_texts)

print(f"Vectors created  : {len(chunk_vectors)}")
print(f"Each vector size : {len(chunk_vectors[0])} dimensions")

# ── STEP 4: Manual Retrieval (to see what's happening under the hood) ──

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve(query, top_k=2):
    print(f"\nQUERY: '{query}'")
    print("-" * 50)

    query_vector = embeddings_model.embed_query(query)

    # Score every chunk
    scored_chunks = []
    for i, chunk_vector in enumerate(chunk_vectors):
        score = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((score, chunks[i].page_content))

    # Sort by score descending
    scored_chunks.sort(reverse=True)

    print(f"TOP {top_k} MOST RELEVANT CHUNKS:")
    for rank, (score, content) in enumerate(scored_chunks[:top_k], 1):
        print(f"\n  [{rank}] Score: {score:.4f}")
        print(f"  Content: {content[:120]}...")


# Test retrieval
retrieve("How do I reset my device?")
retrieve("How long does the battery last?")
retrieve("Is water damage covered?")