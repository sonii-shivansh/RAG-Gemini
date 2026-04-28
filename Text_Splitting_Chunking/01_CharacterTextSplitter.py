from langchain_text_splitters import CharacterTextSplitter

# Splits purely based on a single character separator (default: \n\n)

text = """Artificial Intelligence is transforming industries worldwide.
Companies are investing billions in AI research.

Machine learning is a subset of AI that uses data.
Neural networks are inspired by the human brain.

RAG systems combine retrieval with generation.
They ground LLMs in real, factual data."""

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
