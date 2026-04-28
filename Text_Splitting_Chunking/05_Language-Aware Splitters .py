from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python-aware splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language      = Language.PYTHON,
    chunk_size    = 1000,
    chunk_overlap = 100
)

python_code = """
def calculate_embeddings(text: str) -> list:
    \"\"\"Convert text to vector embeddings.\"\"\"
    model = load_model()
    embedding = model.encode(text)
    return embedding.tolist()

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []

    def add(self, vector: list):
        self.vectors.append(vector)

    def search(self, query_vector: list, top_k: int = 5):
        similarities = []
        for vec in self.vectors:
            score = cosine_similarity(query_vector, vec)
            similarities.append(score)
        return sorted(similarities, reverse=True)[:top_k]
"""

chunks = python_splitter.split_text(python_code)

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)