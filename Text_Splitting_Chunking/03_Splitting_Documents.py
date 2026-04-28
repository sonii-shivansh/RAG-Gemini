from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_rag_doc.pdf")
documents = loader.load()

print(f"Pages loaded: {len(documents)}")

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 1000,
    chunk_overlap = 200
)

# split_documents preserves AND enriches metadata automatically
chunks = splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")
print(f"\nSample chunk content:\n{chunks[0].page_content}")
print(f"\nSample chunk metadata:\n{chunks[0].metadata}")