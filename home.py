import streamlit as st
import subprocess
import os
import time
import sys

from utils.rag import (
    build_rag_chain,
    embed_profil,
    expliquer_matching,
    rechercher_offres_similaires
)

# Mode test pour CI/CD
if "--test" in sys.argv:
    print("Home.py OK")
    sys.exit(0)

# -----------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(
    page_title="RAG France Travail",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS PERSONNALISÉ
# -----------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .section-header {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<h1 class="main-header">💼 RAG France Travail</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Trouvez les offres d\'emploi qui vous correspondent grâce à l\'IA</p>', unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### Mise à jour de la base")

    if st.button("Actualiser la base", use_container_width=True):
        with st.spinner("Ingestion en cours..."):
            progress = st.progress(0)
            status = st.empty()

            try:
                for i in range(100):
                    progress.progress(i + 1)
                    status.text(f"Progression : {i+1}%")
                    time.sleep(0.01)

                result = subprocess.run(
                    [sys.executable, "utils/ingest.py"],
                    capture_output=True,
                    text=True
                )

                progress.empty()
                status.empty()

                if result.returncode == 0:
                    st.success("✅ Ingestion terminée !")
                    st.balloons()
                    st.session_state.clear()
                    st.info("La base a été recréée, rechargement de l'application...")
                    st.rerun()
                else:
                    st.error("❌ Erreur lors de l'ingestion")
                    st.code(result.stderr)

            except Exception as e:
                st.error(f"Erreur : {e}")

    st.markdown("---")
    st.markdown("### Exemples de questions")

    examples = [
        "Quelles offres mentionnent DevOps ?",
        "Offres d'emploi pour développeur Python",
        "Quelles offres parlent de machine learning ?",
        "Offres avec télétravail"
    ]

    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state.example_question = q
    
    # Ne pas charger le RAG si on est en mode test
    if "--test" in sys.argv:
        st.stop()

# -----------------------------
# CHARGEMENT DU RAG
# -----------------------------
if "--test" not in sys.argv and "rag_chain" not in st.session_state:
    with st.spinner("Chargement du modèle IA..."):
        try:
            rag = build_rag_chain()
        except Exception as e:
            st.error(
                "Impossible de charger la base RAG. "
                "Cliquez sur 'Actualiser la base' dans le menu de gauche."
            )
            st.exception(e)   # affiche le détail de l'erreur pour le debug
            st.stop()

        st.session_state.rag_chain = rag["chain"]
        st.session_state.embedder = rag["embedder"]
        st.session_state.collection = rag["collection"]
        st.session_state.llm = rag["llm"]

    st.success("Modèle IA chargé !")

embedder   = st.session_state.embedder
collection = st.session_state.collection
llm        = st.session_state.llm

# Vérification si la base RAG est vide ou cassée
try:
    nb_docs = collection._collection.count()
except Exception:
    st.error("Base RAG corrompue ou illisible — veuillez cliquer sur 'Actualiser la base'.")
    st.stop()

if nb_docs == 0:
    st.error("Base RAG vide — veuillez cliquer sur 'Actualiser la base'.")
    st.stop()

# -----------------------------
# SECTION : MATCHING PROFIL
# -----------------------------
st.markdown('<div class="section-header"><h2>Matching Profil ↔ Offres</h2></div>', unsafe_allow_html=True)

competences  = st.text_area("Vos compétences principales", placeholder="Python, gestion de projet...")
experience   = st.slider("Années d'expérience", 0, 40, 3)
niveau       = st.selectbox("Niveau d'étude", ["Aucun", "CAP / BEP", "Bac", "Bac+2", "Bac+3", "Master", "Doctorat"])
metier_vise  = st.text_input("Métier visé", placeholder="Développeur Python...")
localisation = st.text_input("Localisation (optionnel)", placeholder="Paris, Lyon...")

colA, colB = st.columns(2)

with colA:
    if st.button("Enregistrer mon profil", use_container_width=True):
        st.session_state.profil = {
            "competences":  competences,
            "experience":   experience,
            "niveau":       niveau,
            "metier_vise":  metier_vise,
            "localisation": localisation,
        }
        st.success("Profil enregistré !")

with colB:
    if st.button("Analyser les offres compatibles", use_container_width=True):
        profil = st.session_state.get("profil")

        if not profil:
            st.error("Veuillez d'abord enregistrer votre profil.")
        else:
            try:
                embedding  = embed_profil(profil, embedder)
                offres     = rechercher_offres_similaires(embedding, collection)
                explication = expliquer_matching(profil, offres, llm)
            except Exception as e:
                st.error(
                    "❌ Erreur lors de la recherche dans la base RAG. "
                    "Cliquez sur 'Actualiser la base' puis réessayez."
                )
                st.exception(e)
                st.stop()

            st.subheader("Offres les plus compatibles")
            for i, doc in enumerate(offres):
                st.markdown(f"### Offre {i+1}")
                st.write(doc.page_content)

            # CORRECTION : affichage de l'analyse de compatibilité
            st.subheader("Analyse de compatibilité")
            st.write(explication)

# -----------------------------
# SECTION : CHAT IA
# -----------------------------
st.markdown('<div class="section-header"><h2>Assistant IA</h2></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Gestion question exemple
question = st.session_state.pop("example_question", None)

# Input utilisateur
prompt = st.chat_input("Posez votre question...") or question

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                answer = st.session_state.rag_chain.invoke(prompt)
            except Exception as e:
                st.error(
                    "❌ Erreur lors de l'accès à la base RAG. "
                    "Cliquez sur 'Actualiser la base' puis réessayez."
                )
                st.exception(e)
                st.stop()
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})