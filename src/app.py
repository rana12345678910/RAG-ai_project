import streamlit as st
import subprocess
import pickle
from sentence_transformers import SentenceTransformer

# Charger la collection sauvegardée (créée dans demo.py)
with open("src/collection.pkl", "rb") as f:
    collection = pickle.load(f)

# Charger le modèle d'embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Assistant Juridique RAG (Ollama)")

query = st.text_input("Posez votre question :")

if query:
    # Encoder la requête
    query_emb = model.encode(query).tolist()

    # Chercher les 3 passages les plus pertinents
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    # Construire le contexte à partir des documents
    context = "\n".join(sum(results["documents"], []))  

    # Préparer le prompt
    prompt = f"Tu es un assistant juridique. Utilise uniquement ce contexte:\n{context}\n\nQuestion: {query}"

    # Appeler Ollama 
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        capture_output=True
    )

    # Afficher la réponse
    st.subheader("Réponse du modèle :")
    st.write(result.stdout.decode())

    # Afficher aussi les passages utilisés
    st.subheader("Passages juridiques utilisés :")
    for doc in results["documents"]:
        st.write(doc)
