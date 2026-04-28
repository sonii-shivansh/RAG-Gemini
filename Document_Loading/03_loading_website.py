from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load()

print(f"Number of documents: {len(documents)}")
print(f"Content length     : {len(documents[0].page_content)} characters")
print(f"Metadata           : {documents[0].metadata}")
print(f"\nContent preview:\n{documents[0].page_content[:300]}")