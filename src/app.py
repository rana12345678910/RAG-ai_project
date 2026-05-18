import streamlit as st
from agent import run_agent

st.set_page_config(page_title="Agent Juridique", page_icon="⚖️")

st.title("⚖️ Agent Juridique")
st.caption("Posez votre question — l'agent choisit automatiquement le bon outil.")

query = st.text_area("Votre question", height=100, placeholder="Ex : Qu'est-ce que la responsabilité civile ?")

if st.button("Envoyer", type="primary"):
    if not query.strip():
        st.warning("Veuillez entrer une question.")
    else:
        with st.spinner("L'agent réfléchit..."):
            result = run_agent(query)

        st.subheader("Réponse")
        st.write(result)