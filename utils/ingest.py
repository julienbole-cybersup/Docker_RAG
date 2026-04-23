"""
ingest.py — RAG France Travail · Formation Docker J2
Charge les offres depuis offres.json, construit des chunks,
et les stocke dans ChromaDB.
"""

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import chromadb


# ── Étape 1 : Chargement des offres ──────────────────────────────────────────

def load_offers(filepath: str = "offres.json") -> list[dict]:
    """
    Charge les offres depuis un fichier JSON.

    Args:
        filepath : chemin vers le fichier JSON

    Returns:
        liste de dictionnaires, une offre par dictionnaire

    TODO :
        - Ouvrir le fichier JSON en lecture avec le bon encodage
        - Retourner son contenu
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Étape 2 : Construction des Documents LangChain ───────────────────────────

def build_documents(offers: list[dict]) -> list[Document]:
    """
    Convertit chaque offre en Document LangChain.

    Un Document LangChain a deux attributs :
        - page_content : le texte qui sera embedé
        - metadata     : des champs structurés pour filtrer les résultats (ex = titre, salaire, type de contrat)

    Args:
        offers : liste d'offres brutes (dictionnaires)

    Returns:
        liste de Documents LangChain

    TODO :
        Pour chaque offre, extraire les champs suivants :
            - intitule
            - description
            - entreprise → nom        ⚠️ champ imbriqué, peut être absent
            - lieuTravail → libelle   ⚠️ champ imbriqué
            - typeContratLibelle
            - salaire → libelle       ⚠️ peut être absent sur certaines offres
            - experienceLibelle
            - dateCreation            ⚠️ garder seulement les 10 premiers caractères
            - id

        Construire page_content : texte lisible avec tous les champs sauf description,
        puis ajouter la description en dessous.

        Construire metadata : dictionnaire avec tous les champs sauf description.

        Créer un Document(page_content=..., metadata=...) par offre.

    Indice pour les champs imbriqués :
        offer.get("entreprise", {}).get("nom", "")
    """

    docs = []

    for offer in offers:

        # Extraction des champs
        title       = offer.get("intitule", "")
        description = offer.get("description", "")
        company     = offer.get("entreprise", {}).get("nom", "")
        location    = offer.get("lieuTravail", {}).get("libelle", "")
        contract    = offer.get("typeContratLibelle", "")
        salary      = offer.get("salaire", {}).get("libelle", "")
        experience  = offer.get("experienceLibelle", "")
        published   = offer.get("dateCreation", "")[:10]
        offer_id    = offer.get("id", "")

        # Construction du texte principal (lisible pour embedding)
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

        # Construction des métadonnées
        metadata = {
            "id": offer_id,
            "title": title,
            "company": company,
            "location": location,
            "contract": contract,
            "salary": salary,
            "experience": experience,
            "published": published,
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


# ── Étape 3 : Pipeline d'ingestion complet ───────────────────────────────────

def ingest(filepath: str = "offres.json", persist_dir: str = "./chroma_db"):
    
    """
    Pipeline complet :
        1. Chargement des offres depuis le JSON
        2. Construction des Documents LangChain
        3. Découpage en chunks
        4. Calcul des embeddings
        5. Stockage dans ChromaDB

    Args:
        filepath    : chemin vers offres.json
        persist_dir : dossier de persistance ChromaDB

    TODO :
        - Appeler load_offers() et afficher le nombre d'offres chargées
        - Appeler build_documents() et afficher le nombre de documents construits
        - Appeler splitter.split_documents() et afficher le nombre de chunks
        - Appeler Chroma.from_documents() avec chunks, embeddings, persist_directory
        - Afficher le nombre de vecteurs indexés : vectorstore._collection.count()
    """

    # Étape 1 — Chargement
    # --- votre code ici ---
    offers = load_offers(filepath)
    print(f"{len(offers)} offres chargées")

    # Étape 2 — Construction des Documents
    # --- votre code ici ---
    documents = build_documents(offers)
    print(f" {len(documents)} documents construits")

    # Étape 3 — Chunking
    # Ne pas modifier ces paramètres
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )

    # --- votre code ici ---
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks créés")

    # Étape 4 — Chargement du modèle d'embeddings
    # Ne pas modifier — modèle calibré pour le français
    print("Chargement du modèle d'embeddings (1-2 min la première fois)…")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Étape 5 — Stockage dans ChromaDB
    # --- votre code ici ---
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Base vectorielle sauvegardée dans {persist_dir}")
    print(f"{vectorstore._collection.count()} vecteurs indexés")


if __name__ == "__main__":
    ingest()
    # Aperçu de la base vectorielle
    # TODO : importer chromadb
    # TODO : créer un client PersistentClient avec le chemin vers chroma_db
    # TODO : récupérer la collection "langchain"
    # TODO : afficher le nombre de vecteurs avec collection.count()
    # TODO : récupérer 3 documents avec collection.get(limit=3, include=["documents", "metadatas"])
    # TODO : pour chaque document, afficher :
    #        - le titre (dans metadatas)
    #        - le lieu (dans metadatas)
    #        - le contrat (dans metadatas)
    #        - les 200 premiers caractères du texte (dans documents)

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("langchain")

    print(f"\n{collection.count()} vecteurs dans la base\n")

    results = collection.get(
        limit=3,
        include=["documents", "metadatas"]
    )

    for doc, meta in zip(results["documents"], results["metadatas"]):
        print("----")
        print(f"Titre      : {meta.get('title')}")
        print(f"Lieu       : {meta.get('location')}")
        print(f"Contrat    : {meta.get('contract')}")
        print(f"Texte      : {doc[:200]}...")