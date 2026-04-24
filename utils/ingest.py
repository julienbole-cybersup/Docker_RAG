"""
ingest.py — RAG France Travail
Charge les offres depuis offres.json, construit des chunks,
et les stocke dans ChromaDB.
"""

import json
import os
import shutil
import time
import gc
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ── Étape 1 : Chargement des offres ──────────────────────────────────────────

def load_offers(filepath: str = "data/offres.json") -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Étape 2 : Construction des Documents LangChain ───────────────────────────

def build_documents(offers: list[dict]) -> list[Document]:
    docs = []

    for offer in offers:
        title       = offer.get("intitule", "")        or "N/A"
        description = offer.get("description", "")     or "N/A"
        company     = offer.get("entreprise", {}).get("nom", "")        or "N/A"
        location    = offer.get("lieuTravail", {}).get("libelle", "")   or "N/A"
        contract    = offer.get("typeContratLibelle", "")               or "N/A"
        salary      = offer.get("salaire", {}).get("libelle", "")       or "N/A"
        experience  = offer.get("experienceLibelle", "")                or "N/A"
        published   = (offer.get("dateCreation", "") or "")[:10]        or "N/A"
        offer_id    = offer.get("id", "")                               or "N/A"

        text = (
            f"Titre : {title}\n"
            f"Entreprise : {company}\n"
            f"Lieu : {location}\n"
            f"Contrat : {contract}\n"
            f"Salaire : {salary}\n"
            f"Expérience : {experience}\n"
            f"Date : {published}\n\n"
            f"Description :\n{description}"
        )

        metadata = {
            "id":         offer_id,
            "title":      title,
            "company":    company,
            "location":   location,
            "contract":   contract,
            "salary":     salary,
            "experience": experience,
            "published":  published,
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


# ── Utilitaire : suppression robuste sur Windows ─────────────────────────────

def _remove_dir_windows_safe(path: str, retries: int = 5, delay: float = 1.0) -> bool:
    """
    Sur Windows, SQLite pose un verrou sur chroma.sqlite3 tant qu'un autre
    processus (ex: Streamlit) le tient ouvert. On réessaie plusieurs fois
    avant d'abandonner et de vider la collection à la place.
    """
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            print(f"Ancien dossier supprimé : {path}")
            return True
        except PermissionError:
            print(f"Fichier verrouillé, tentative {attempt + 1}/{retries} dans {delay}s…")
            gc.collect()
            time.sleep(delay)

    print(
        f"Impossible de supprimer {path} (verrou Windows actif).\n"
        "La collection existante va être vidée puis réindexée."
    )
    return False


# ── Étape 3 : Pipeline d'ingestion complet ───────────────────────────────────

def ingest(filepath: str = "data/offres.json", persist_dir: str = "./chroma_db"):

    # Étape 1 — Chargement
    offers = load_offers(filepath)
    print(f"{len(offers)} offres chargées")

    # Étape 2 — Construction des Documents
    documents = build_documents(offers)
    print(f"{len(documents)} documents construits")

    # Étape 3 — Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks créés")

    # Étape 4 — Embeddings
    print("Chargement du modèle d'embeddings (1-2 min la première fois)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Étape 5 — Stockage ChromaDB
    if os.path.exists(persist_dir):
        deleted = _remove_dir_windows_safe(persist_dir)

        if not deleted:
            # Fallback Windows : vider la collection sans supprimer les fichiers
            import chromadb
            client = chromadb.PersistentClient(path=persist_dir)
            collection_name = "langchain"  # nom par défaut LangChain/Chroma
            try:
                client.delete_collection(collection_name)
                print(f"Collection '{collection_name}' vidée.")
            except Exception:
                pass  # collection inexistante, pas de problème

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Base vectorielle sauvegardée dans {persist_dir}")
    print(f"{vectorstore._collection.count()} vecteurs indexés")


if __name__ == "__main__":
    ingest()