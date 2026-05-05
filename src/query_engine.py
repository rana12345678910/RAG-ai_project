from embeddings import collection, model

#recherche et reponse
query = "Quelles obligations pour les fournisseurs d'IA ?"
query_emb = model.encode(query).tolist()

results = collection.query(query_embeddings=[query_emb], n_results=3)
print(results["documents"])
