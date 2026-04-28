from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

chunks = [
    "To reset your device, hold the power button for 10 seconds.",
    "The device warranty covers manufacturing defects for 1 year.",
    "Contact support at support@company.com for assistance."
]

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

# vectors = embeddings.embed_documents(chunks) 
vectors = [embeddings.embed_query(chunk) for chunk in chunks]
print(vectors)
print(f"Number of vectors : {len(vectors)}")        
print(f"Each vector size  : {len(vectors[0])}")     # 1536