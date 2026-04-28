from langchain_community.document_loaders import TextLoader

# Create the loader
loader = TextLoader("sample.txt")

# Load returns a LIST of Document objects
documents = loader.load()

print(type(documents))
print(len(documents))
print(type(documents[0]))

print("--- CONTENT ---")
print(documents[0].page_content)

print("--- METADATA ---")
print(documents[0].metadata)