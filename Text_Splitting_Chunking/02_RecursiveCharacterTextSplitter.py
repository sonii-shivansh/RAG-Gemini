from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Introduction to Machine Learning

Machine learning is a branch of artificial intelligence. It focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed.

Types of Machine Learning

There are three main types of machine learning. The first is supervised learning, where the model learns from labeled data. The second is unsupervised learning, where the model finds patterns in unlabeled data. The third is reinforcement learning, where an agent learns by interacting with an environment.

Applications

Machine learning is used in many fields. In healthcare, it helps diagnose diseases. In finance, it detects fraud. In transportation, it powers self-driving cars."""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 40
)

chunks = text_splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)