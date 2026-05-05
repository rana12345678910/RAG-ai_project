from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import pickle
import subprocess

# Charger un PDF 
reader = PdfReader("corpus_juridique/AI_Act.pdf")
text = "".join([page.extract_text() for page in reader.pages])

# Chunking
def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words, size))]

chunks = chunk_text(text)

# Embeddings + stockage
model = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.Client()
collection = client_db.create_collection("juridique")

for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    collection.add(documents=[chunk], embeddings=[emb], ids=[f"chunk_{i}"])

# Sauvegarder la collection pour réutilisation
with open("src/collection.pkl", "wb") as f:
    pickle.dump(collection, f)

# Exemple de recherche
query = "Quelles obligations pour les fournisseurs d'IA ?"
query_emb = model.encode(query).tolist()
results = collection.query(query_embeddings=[query_emb], n_results=3)

# Construire le contexte
context = "\n".join(sum(results["documents"], []))

# Préparer le prompt pour Ollama
prompt = f"Tu es un assistant juridique. Utilise uniquement ce contexte:\n{context}\n\nQuestion: {query}"

# Appeler Ollama (modèle mistral)
result = subprocess.run(
    ["ollama", "run", "mistral"],
    input=prompt.encode(),
    capture_output=True
)

# Afficher la réponse
print("Réponse du modèle :")
print(result.stdout.decode())

# Afficher aussi les passages utilisés
print("\nPassages juridiques utilisés :")
for doc in results["documents"]:
    print(doc)
