from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_rag_document.pdf")
documents = loader.load()

print(f"Total pages loaded: {len(documents)}")

# PyPDFLoader creates ONE Document PER PAGE
for i, doc in enumerate(documents):
    print(f"\n--- Page {i+1} ---")
    print(f"Content preview : {doc.page_content[:200]}...")
    print(f"Metadata        : {doc.metadata}")