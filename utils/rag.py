"""
rag.py — RAG France Travail
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
# ───────────────────────────────────────────────
# FONCTIONS MATCHING PROFIL ↔ OFFRES
# ───────────────────────────────────────────────

def embed_profil(profil: dict, embedder) -> list[float]:
    texte = (
        f"Compétences : {profil['competences']}\n"
        f"Expérience : {profil['experience']} ans\n"
        f"Niveau d'étude : {profil['niveau']}\n"
        f"Métier visé : {profil['metier_vise']}\n"
        f"Localisation : {profil['localisation']}"
    )
    return embedder.embed_query(texte)


def rechercher_offres_similaires(profil_embedding: list[float], collection, n: int = 5):
    docs = collection.similarity_search_by_vector(
        embedding=profil_embedding,
        k=n
    )
    return docs


def expliquer_matching(profil: dict, offres: list, llm) -> str:
    """
    Demande au LLM d'expliquer la correspondance profil ↔ offres.
    CORRECTION : on extrait .content pour retourner une str propre.
    """
    offres_texte = "\n\n".join(
        f"Offre {i+1} :\n{doc.page_content}" for i, doc in enumerate(offres)
    )

    prompt = (
        "Voici un profil candidat :\n"
        f"{profil}\n\n"
        "Voici des offres d'emploi trouvées :\n"
        f"{offres_texte}\n\n"
        "Explique en quoi ces offres correspondent au profil.\n"
        "Donne :\n"
        "- un score de compatibilité (0 à 100)\n"
        "- les compétences correspondantes\n"
        "- les compétences manquantes\n"
        "- pourquoi l'offre est pertinente"
    )

    # CORRECTION : .invoke() retourne un AIMessage — on extrait .content
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ───────────────────────────────────────────────
# CONSTRUCTION DE LA CHAÎNE RAG
# ───────────────────────────────────────────────

def build_rag_chain(persist_dir: str = "./chroma_db") -> dict:
    """
    Construit et retourne la chaîne RAG complète.

    Returns:
        dict avec les clés : chain, embedder, collection, llm
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "Variable d'environnement GROQ_API_KEY manquante. "
            "Vérifiez votre fichier .env à la racine du projet."
        )

    # 1 — Embeddings
    print("Chargement du modèle d'embeddings…")
    embedder = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 2 — Chargement ChromaDB
    try:
        collection = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedder
        )
        nb_docs = collection._collection.count()
        print(f"Base ChromaDB chargée ({nb_docs} documents).")
    except Exception as e:
        raise RuntimeError(f"Impossible de charger ChromaDB : {e}")

    # 3 — Retriever
    retriever = collection.as_retriever(search_kwargs={"k": 4})

    # 4 — LLM Groq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    # 5 — Prompt
    prompt = ChatPromptTemplate.from_template("""
    Tu es un assistant spécialisé dans l'analyse du marché de l'emploi en France.
    Utilise uniquement les offres d'emploi ci-dessous pour répondre.

    Offres d'emploi :
    {context}

    Question : {question}

    Réponse :
    """)

    # 6 — Formatage des documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7 — Assemblage LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 8 — Retour complet pour Streamlit
    return {
        "chain":      chain,
        "embedder":   embedder,
        "collection": collection,
        "llm":        llm,
    }


# ───────────────────────────────────────────────
# MODE TEST
# ───────────────────────────────────────────────

if __name__ == "__main__":
    rag = build_rag_chain()
    chain = rag["chain"]
    print(chain.invoke("Quelles offres mentionnent Python ?"))