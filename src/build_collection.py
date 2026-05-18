from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

print("Working directory:", os.getcwd())

# Load PDF
reader = PdfReader("../corpus_juridique/Code civil.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + " "

# Smaller chunks with overlap
def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

chunks = chunk_text(text)

# Embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
client = chromadb.PersistentClient(path="db/")
collection = client.get_or_create_collection("juridique")

# Store chunks
for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[emb],
        ids=[f"chunk_{i}"]
    )

print(f"Database created successfully with {len(chunks)} chunks in db/")