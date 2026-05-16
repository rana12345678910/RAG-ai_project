import subprocess
from sentence_transformers import SentenceTransformer
import chromadb

# Charger le modèle et la collection persistante
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="db/")
collection = client.get_or_create_collection("juridique")

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
for i, doc in enumerate(results["documents"][0], 1):
    print(f"{i}. {doc[:300]}...")
