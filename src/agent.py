from langchain_ollama import OllamaLLM
from tools import search_tool, calculator_tool, llm_tool

# 🤖 LLM
llm = OllamaLLM(model="mistral")

# 🧰 Tools map
tools = {
    "LegalSearch": search_tool,
    "Calculator": calculator_tool,
    "LLM": llm_tool
}

tools_description = """
- LegalSearch: Pour chercher dans les documents juridiques français (Code Civil)
- Calculator: Pour faire des calculs mathématiques. Exemple: '2 + 2'
- LLM: Pour les questions générales qui ne nécessitent pas de recherche juridique
"""

def run_agent(question: str):
    scratchpad = ""
    max_steps = 5

    for step in range(max_steps):
        prompt = f"""Tu es un assistant juridique strict. Tu dois choisir le bon outil pour répondre.

Outils disponibles:
{tools_description}

RÈGLES:
- Pour toute question juridique → utilise LegalSearch
- Pour tout calcul → utilise Calculator  
- Pour questions générales → utilise LLM
- Ta réponse finale doit être en maximum 5 phrases
- Basée UNIQUEMENT sur ce que l'outil retourne
- JAMAIS inventer ou ajouter des informations
- JAMAIS mentionner des dates ou articles non trouvés dans l'Observation
- Si l'information n'est pas dans l'Observation, ne la mentionne pas
- JAMAIS inventer ou ajouter des informations
- Basée UNIQUEMENT sur ce que l'outil retourne

Format OBLIGATOIRE:
Thought: réfléchis à quel outil utiliser
Action: nom de l'outil (LegalSearch, Calculator, ou LLM)
Action Input: ta requête pour l'outil
Observation: résultat de l'outil
Thought: j'ai maintenant la réponse
Final Answer: ta réponse finale en maximum 3 phrases

Question: {question}
{scratchpad}"""

        response = llm.invoke(prompt)
        scratchpad += response

        # ✅ Final Answer found
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
            return answer

        # ✅ Action found
        if "Action:" in response and "Action Input:" in response:
            try:
                action = response.split("Action:")[-1].split("\n")[0].strip()
                action_input = response.split("Action Input:")[-1].split("\n")[0].strip()

                print(f"[Étape {step+1}] Outil: {action} | Requête: {action_input}")

                if action in tools:
                    observation = tools[action](action_input)
                    # Truncate observation to avoid too long prompts
                    observation = observation[:500]
                else:
                    observation = f"Outil inconnu: {action}. Utilise: LegalSearch, Calculator, ou LLM"

                scratchpad += f"\nObservation: {observation}\nThought:"

            except Exception as e:
                scratchpad += f"\nObservation: Erreur - {e}\nThought:"

    return "Je n'ai pas pu trouver une réponse."


# 💬 CLI chat
if __name__ == "__main__":
    print("=== Agent Juridique ===")
    print("Outils: LegalSearch | Calculator | LLM")
    print("Tapez 'exit' pour quitter\n")

    while True:
        q = input("Vous: ")

        if q.lower() in ["exit", "quit"]:
            break

        print("\n[Réflexion en cours...]\n")
        result = run_agent(q)
        print("\nAI:", result, "\n")