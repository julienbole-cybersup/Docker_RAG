# 📚 RAG France Travail — Documentation du projet

Ce projet implémente un **système RAG (Retrieval-Augmented Generation)** permettant d’interroger une base documentaire locale via un modèle LLM.  
L’application propose :

- une interface **Streamlit**
- un bouton pour **mettre à jour la base ChromaDB** via `ingest.py`
- un champ de question pour interroger le **RAG**
- une architecture **Docker** simple et reproductible

---

## 🚀 Fonctionnalités

- **Ingestion des documents** : extraction, nettoyage et vectorisation des données dans ChromaDB  
- **RAG** : recherche des passages pertinents + génération de réponse via un LLM  
- **Interface utilisateur** : Streamlit pour interagir facilement avec le système  
- **Docker** : déploiement reproductible avec un seul service  

---

## 🧱 Architecture du projet

