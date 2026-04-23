"""
rag.py — RAG France Travail · Formation Docker J2
Retrieval + génération avec Groq (Llama 3.3-70b).
À lancer après ingest.py.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# TODO : remplacer par votre clé API Groq
# Créer un compte sur https://console.groq.com → API Keys → Create API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ── Construction de la chain RAG ──────────────────────────────────────────────

def build_rag_chain(persist_dir: str = "./chroma_db"):
    """
    Construit la chaine RAG complète avec LCEL.

    Une chain LCEL se lit de gauche à droite avec | :
        retriever → prompt → llm → output_parser

    Args:
        persist_dir : dossier ChromaDB créé par ingest.py

    Returns:
        chain LCEL prête à invoquer avec chain.invoke(question)

    TODO (dans l'ordre) :
        1. Instancier HuggingFaceEmbeddings avec le même modèle que ingest.py
        2. Charger la base ChromaDB depuis persist_dir
        3. Créer un retriever à partir du vectorstore (chercher dans la doc : as_retriever)
        4. Instancier le LLM ChatGroq — modèle : llama-3.3-70b-versatile, temperature : 0
        5. Écrire le prompt — il doit contenir deux variables : {context} et {question}
           context = les chunks retrouvés / question = la question de l'utilisateur
        6. Écrire format_docs — les chunks sont des Documents, extraire page_content
           et les assembler en une seule chaîne
        7. Assembler la chain — elle se lit de gauche à droite avec |
           entrée → retriever → prompt → llm → parser → sortie
        8. Retourner la chain
    """

    # Étape 1 — Embeddings
    # Utiliser exactement le même modèle que dans ingest.py
    # --- votre code ici ---
    print("Chargement du modèle d'embeddings (1-2 min la première fois)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Étape 2 — Chargement ChromaDB
    # La base a déjà été créée par ingest.py — on la charge depuis le disque
    # --- votre code ici ---
    try:
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        nb_docs = vectorstore._collection.count()
        if nb_docs == 0:
            raise ValueError("La base ChromaDB est vide.")
        print(f"Base ChromaDB chargée avec succès ({nb_docs} documents).")
    except Exception as e:
        print("Erreur : impossible de charger la base documentaire ChromaDB.")
        print(f"Détail : {e}")
        raise SystemExit("Arrêt du programme : la base documentaire est introuvable ou corrompue.")


    # Étape 3 — Retriever
    # Le retriever est l'objet qui fait la recherche dans ChromaDB
    # Il prend une question en entrée et retourne les k chunks les plus proches
    # --- votre code ici ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Étape 4 — LLM
    # ChatGroq est le client pour appeler Llama via l'API Groq
    # La clé API est dans le .env — load_dotenv() s'en charge automatiquement
    # --- votre code ici ---
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)

    # Étape 5 — Prompt
    # Le prompt est le message envoyé au LLM
    # Il doit contenir {context} (les chunks) et {question} (la question utilisateur)
    # --- votre code ici ---
    prompt = ChatPromptTemplate.from_template("""
    Tu es un assistant spécialisé dans l'analyse du marché de l'emploi data en France.
    Utilise uniquement les offres d'emploi ci-dessous pour répondre à la question.
    Si tu ne trouves pas la réponse dans les offres, dis-le clairement. 

    Offres d'emploi :
    {context}

    Question : {question}

    Réponse :""")

    # Étape 6 — Formatage des chunks
    # Les chunks sont des objets Document avec un attribut page_content
    # Cette fonction les assemble en une seule chaîne de texte
    # --- votre code ici ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Étape 7 — Assemblage de la chain LCEL
    # On connecte tous les éléments avec | dans l'ordre logique du pipeline
    # --- votre code ici ---
    assemblage = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    return assemblage

if __name__ == "__main__":
    assemblage = build_rag_chain()

    questions = [
        # Categorie 1
        "Je cherche un poste où je peux apprendre rapidement sur des projets variés.",
        "Quelles offres correspondent à quelqu'un qui sort d'un master IA ?",
        "Je veux un poste avec de la technique mais aussi du contact avec des clients.",

        # Categ 2
        "Quelles offres mentionnent la possibilité de travailler à distance ?",
        "Y a-t-il des postes qui parlent d'un environnement startup ou scale-up ?",

        # Categ 3 
        "Quelles offres mentionnent Python ?",
        "Quelles offres parlent de machine learning ?",

        # Categ 4
        "Quel outil MLOps est le plus demandé dans les offres ?",
        "Combien d'offres mentionnent Docker ?",

        # Categ 5
        "Montre-moi toutes les offres où il y a le mot Docker.",
        "Montre-moi toutes les offres où il y a le mot Spark.",
        "Montre-moi toutes les offres où il y a le mot Kubernetes.",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question : {q}")
        print(f"{'='*60}")
        print(assemblage.invoke(q))