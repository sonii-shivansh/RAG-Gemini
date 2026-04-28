from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader
)

def load_all_documents():
    all_documents = []

    # Load text files
    txt_loader = TextLoader("sample.txt")
    all_documents.extend(txt_loader.load())

    # Load PDFs
    pdf_loader = PyPDFLoader("sample_rag_document.pdf")
    all_documents.extend(pdf_loader.load())

    # Load CSVs
    csv_loader = CSVLoader("products.csv")
    all_documents.extend(csv_loader.load())

    return all_documents

docs = load_all_documents()

print(f"Total documents loaded: {len(docs)}")

# Each document knows where it came from
for doc in docs:
    print(doc.metadata['source'])