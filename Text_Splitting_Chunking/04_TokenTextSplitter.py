from langchain_text_splitters import TokenTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_rag_doc.pdf")
documents = loader.load()

print(f"Pages loaded: {len(documents)}")

splitter = TokenTextSplitter(
    chunk_size = 256,
    chunk_overlap = 30
)

chunks = splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")
