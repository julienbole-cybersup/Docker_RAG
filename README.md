# Chatbot RAG – Recherche d’offres d’emploi (Streamlit)

Ce projet est une application **Streamlit** qui permet de rechercher des offres d’emploi via un **chatbot intelligent** basé sur une approche **RAG (Retrieval-Augmented Generation)**.

Les données proviennent de l’API **France Travail**, exportées en **JSON**, puis utilisées comme base documentaire pour interroger les offres via un modèle de langage (LLM).

---

## 🚀 Fonctionnalités

* 💬 Chatbot interactif pour poser des questions sur les offres d’emploi
* 📄 Recherche sémantique dans une base d’offres (RAG)
* 🔎 Filtrage intelligent (métier, localisation, compétences, etc.)
* ⚡ Utilisation d’un LLM via l’API **Groq**
* 🐳 Conteneurisation avec Docker
* 🔁 Pipeline CI/CD avec GitHub Actions

---

## 🧱 Architecture

```
Utilisateur → Streamlit UI → RAG → Base JSON → LLM (Groq)
```

* **Frontend** : Streamlit
* **Backend** : Python
* **Données** : fichiers JSON (offres France Travail)
* **LLM** : API Groq
* **CI/CD** : GitHub Actions
* **Déploiement** : Docker

---

## 📦 Installation

### 1. Cloner le projet

```bash
git clone <repo-url>
cd <repo>
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

### 1. Clé API Groq

Créer une clé API sur Groq, puis ajouter une variable d’environnement :

```bash
export GROQ_API_KEY=your_api_key_here
```

Sous Windows :

```bash
set GROQ_API_KEY=your_api_key_here
```

---

### 2. Ajouter / modifier les données

Les offres d’emploi sont stockées dans :

```
/data/
```

👉 Tu peux :

* ajouter un nouveau fichier JSON
* remplacer les données existantes
* enrichir les offres

⚠️ Assure-toi que le format JSON reste cohérent avec celui attendu par l’application.

---

## ▶️ Lancer l’application

```bash
streamlit run app.py
```

Puis ouvrir dans le navigateur :

```
http://localhost:8501
```

---

## 🐳 Utilisation avec Docker

### Build de l’image

```bash
docker build -t streamlit-rag .
```

### Lancer le conteneur

```bash
docker run -p 8501:8501 -e GROQ_API_KEY=your_api_key streamlit-rag
```

---

## 🔁 CI/CD

Le projet inclut un pipeline GitHub Actions qui :

1. 🧪 Lance les tests avec `pytest`
2. 🐳 Build l’image Docker
3. 📤 Push l’image sur Docker Hub

Déclenchement automatique à chaque push sur `main`.

---

## 🧪 Tests

```bash
pytest
```

---

## 📁 Structure du projet

```
.
├── app.py
├── data/
│   └── jobs.json
├── tests/
├── requirements.txt
├── Dockerfile
└── .github/workflows/
```

---

## 🛠️ Personnalisation

Tu peux facilement :

* changer la source de données (autre API que France Travail)
* modifier la logique RAG
* adapter le prompt du chatbot
* améliorer l’interface Streamlit

---

## ⚠️ Prérequis

* Python 3.10+
* Compte Groq (clé API requise)
* Docker (optionnel)

---

## 💡 Exemple d’utilisation

Tu peux poser des questions comme :

* *"Trouve-moi des jobs en data à Paris"*
* *"Quelles offres demandent Python ?"*
* *"Y a-t-il des postes en télétravail ?"*

---

## 📌 Notes

* Les performances dépendent de la qualité des données JSON
* Le modèle LLM dépend de Groq (latence et quotas)
* Le projet est facilement extensible vers d'autres cas d’usage RAG

---

## 👨‍💻 Contribution

Les contributions sont les bienvenues :

1. Fork du repo
2. Création d’une branche
3. Pull Request

---

## 📄 Licence

MIT
