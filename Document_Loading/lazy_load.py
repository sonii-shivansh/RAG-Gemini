from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample_rag_document.pdf")

# lazy_load returns a generator - memory efficient
for document in loader.lazy_load():
    print(document.metadata)
    # Process one page at a time
    # Only ONE page is in memory at any moment