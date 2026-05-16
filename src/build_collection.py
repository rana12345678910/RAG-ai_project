from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os
print("Dossier courant :", os.getcwd())


# Charger le PDF
reader = PdfReader("corpus_juridique/Code civil.pdf")
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:  
        text += page_text

# Chunking
def chunk_text(text, size=500):
    words = text.split()
    # avancer par pas de "size" pour éviter les chevauchements
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(text)

# Embeddings + stockage persistant
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db/")  # stockage automatique dans le dossier db/
collection = client.get_or_create_collection("juridique")

# Ajouter les chunks
for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(documents=[chunk], embeddings=[emb], ids=[f"chunk_{i}"])

print("Collection créée et sauvegardée dans 'db/'")
