import streamlit as st
import subprocess
import os
from utils.rag import build_rag_chain
import time
import sys

if "--test" in sys.argv:
    print("Home.py OK")
    sys.exit(0)

# Page configuration
st.set_page_config(
    page_title="RAG France Travail",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
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

    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💼 RAG France Travail</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Trouvez les offres d\'emploi qui vous correspondent grâce à l\'IA</p>', unsafe_allow_html=True)

# Sidebar for controls and info
with st.sidebar:
    # Database update section
    st.markdown("### Base de données")

    if st.button("Actualiser la base", type="primary", use_container_width=True):
        with st.spinner("Ingestion en cours..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Simulate progress (you can replace with actual progress tracking)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f'Progression: {i+1}%')
                    time.sleep(0.01)  # Remove this in production

                result = subprocess.run(
                    [sys.executable, "ingest.py"],
                    capture_output=True,
                    text=True
                )


                progress_bar.empty()
                status_text.empty()

                if result.returncode == 0:
                    st.success("✅ Ingestion terminée avec succès !")
                    st.balloons()
                else:
                    st.error("❌ Erreur lors de l'ingestion")
                    with st.expander("Voir les détails de l'erreur"):
                        st.code(result.stderr)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Erreur : {e}")

    st.markdown("---")

    st.markdown("### Exemples de questions")
    example_questions = [
        "Quelles offres mentionnent DevOps ?",
        "Quelles offres parlent de machine learning ?",
        "Offres d'emploi pour développeur Python",
        "Quelles offres mentionnent la possibilité de télétravail ?",
    ]

    for question in example_questions:
        if st.button(f"{question}", key=f"example_{hash(question)}", use_container_width=True):
            st.session_state.example_question = question

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header"><h2>Assistant IA pour l\'emploi</h2></div>', unsafe_allow_html=True)

    # Initialize RAG chain
    if 'rag_chain' not in st.session_state:
        try:
            with st.spinner("Chargement du modèle IA..."):
                st.session_state.rag_chain = build_rag_chain()
            st.success("✅ Modèle IA chargé avec succès !")
        except Exception as e:
            st.error(f"❌ Impossible de charger la chaîne RAG : {e}")
            st.stop()

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle example question from sidebar
    if 'example_question' in st.session_state:
        question = st.session_state.example_question
        del st.session_state.example_question
    else:
        question = None

    # Chat input
    if prompt := st.chat_input("Posez votre question sur les offres d'emploi...") or question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                try:
                    # Stream the response for better UX
                    response_placeholder = st.empty()

                    answer = st.session_state.rag_chain.invoke(prompt)

                    # Display the answer
                    response_placeholder.write(answer)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"❌ Erreur lors de l'appel au modèle : {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with col2:

    # Recent activity
    st.markdown("### 🕒 Activité récente")
    if st.session_state.get('messages'):
        recent_messages = st.session_state.messages[-3:]  # Last 3 messages
        for msg in recent_messages:
            if msg['role'] == 'user':
                st.markdown(f"**Vous:** {msg['content'][:50]}...")
    else:
        st.info("Aucune activité récente")

    # Clear chat button
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Propulsé par l'IA | 💼 France Travail RAG Assistant</p>
</div>
""", unsafe_allow_html=True)