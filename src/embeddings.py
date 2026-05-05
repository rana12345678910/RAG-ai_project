from sentence_transformers import SentenceTransformer
from embeddings import chunks
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

collection = client.create_collection("juridique")

for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(documents=[chunk], embeddings=[emb], ids=[f"chunk_{i}"])
