from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load ALL .txt files in a folder
loader = DirectoryLoader(
    path       = "./documents/",    # folder path
    glob       = "**/*.txt",        # pattern: all txt files, all subfolders
    loader_cls = TextLoader         # which loader to use for each file
)

documents = loader.load()
print(f"Loaded {len(documents)} documents from folder")