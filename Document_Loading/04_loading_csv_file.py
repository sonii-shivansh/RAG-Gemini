from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("products.csv")
documents = loader.load()

print(f"Total rows loaded: {len(documents)}")

for doc in documents:
    print("\n--- Row as Document ---")
    print(f"Content :\n{doc.page_content}")
    print(f"Metadata: {doc.metadata}")