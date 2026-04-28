from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

# Embed a single piece of text
text = "How do I reset my device?"

vector = embeddings.embed_query(text)

print(f"Type: {type(vector)}")
print(f"Dimensions     : {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")
print(f"Last 5 numbers : {vector[-5:]}")