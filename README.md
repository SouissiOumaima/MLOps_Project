# 🚢 MLOps Titanic : Pipeline Complet

> Pipeline MLOps de bout en bout pour la prédiction de survie sur le Titanic.  
> Modularisation · Makefile CI · FastAPI · MLflow · Docker · Elasticsearch · Kibana · React

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![MLflow](https://img.shields.io/badge/MLflow-2.14-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Tests](https://img.shields.io/badge/Tests-13%2F13%20✔-brightgreen)

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Architecture](#-architecture)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Lancement](#-lancement)
- [Endpoints API](#-endpoints-api)
- [Commandes Makefile](#-commandes-makefile)
- [Interface React](#-interface-react)
- [Ateliers réalisés](#-ateliers-réalisés)

---

## 📖 À propos

Ce projet implémente un pipeline **MLOps complet** pour prédire la survie des passagers du Titanic. Il couvre l'ensemble du cycle de vie d'un modèle ML :

- **Modélisation** : Logistic Regression, Random Forest, SVM
- **Tracking** : MLflow pour suivre les expériences
- **Déploiement** : FastAPI + Docker
- **Monitoring** : Elasticsearch + Kibana
- **CI/CD** : Makefile + tests automatisés
- **Interface** : React avec ajout de données en temps réel

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface React :3000                    │
│         Saisie manuelle | CSV | Excel | Comparaison         │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────────┐
│                   FastAPI :8000                             │
│  /health /predict /retrain /add-data /upload-file           │
└──────┬────────────────┬─────────────────────────────────────┘
       │                │
┌──────▼──────┐  ┌──────▼───────────────────────────────────┐
│  MLflow     │  │         model_pipeline.py                │
│  :5000      │  │  prepare → train → evaluate → save       │
└─────────────┘  └──────────────────┬───────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────┐
│              Elasticsearch :9200  ←→  Kibana :5601         │
│              Logs MLflow + Métriques système               │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure du projet

```
MLOps_Project/
│
├── src/
│   └── model_pipeline.py      # Pipeline ML + MLflow + Elasticsearch
│
├── tests/
│   └── test_model.py          # 13 tests unitaires
│
├── frontend/
│   └── src/
│       └── App.js             # Interface React
│
├── data/
│   └── titanic.csv            # Dataset
│
├── models/
│   └── titanic_model.pkl      # Modèle sauvegardé
│
├── logs/
│   └── pipeline.log           # Logs d'exécution
│
├── app.py                     # API FastAPI (8 endpoints)
├── main.py                    # Point d'entrée CLI
├── Makefile                   # Automatisation des tâches
├── Dockerfile                 # Conteneurisation
├── docker-compose.yml         # Elasticsearch + Kibana
├── conftest.py                # Configuration pytest
├── .env                       # Variables d'environnement
├── .dockerignore              # Fichiers exclus de Docker
└── requirements.txt           # Dépendances Python
```

---

## ✅ Prérequis

- Python 3.11+
- Node.js 18+ (pour React)
- Docker Desktop
- Git

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd MLOps_Project
```

### 2. Créer et activer l'environnement virtuel

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances Python

```bash
make install
# ou
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créez un fichier `.env` à la racine :

```env
# API
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/titanic_model.pkl
DATA_PATH=data/titanic.csv

# MLflow
MLFLOW_PORT=5000
MLFLOW_BACKEND_URI=sqlite:///mlflow.db

# Elasticsearch
ES_HOST=http://localhost:9200
ES_INDEX_METRICS=mlflow-metrics
ES_INDEX_SYSTEM=system-metrics

# Docker
IMAGE_NAME=oumaima_souissi_ds2_mlops
DOCKER_USER=oumaima_dockerhub
```

### 5. Installer les dépendances React

```bash
cd frontend
npm install
cd ..
```
---

## 🚀 Lancement

### Pipeline ML uniquement

```bash
# Entraîner le modèle
make train

# Pipeline complet (prepare + train + evaluate + save)
make all

# Comparer les 3 modèles
python main.py --compare
```

### Lancement complet (tout le stack)

Ouvrez **4 terminaux** :

```bash
# Terminal 1 — Elasticsearch + Kibana
make monitoring-up

# Terminal 2 — MLflow
make mlflow-server

# Terminal 3 — API FastAPI
make run-api

# Terminal 4 — React
cd frontend && npm start
```

### URLs disponibles

| Service | URL | Description |
|---|---|---|
| Interface React | http://localhost:3000 | Interface utilisateur |
| API FastAPI | http://localhost:8000 | REST API |
| Swagger Docs | http://localhost:8000/docs | Documentation interactive |
| MLflow UI | http://localhost:5000 | Suivi des expériences |
| Kibana | http://localhost:5601 | Dashboard monitoring |
| Elasticsearch | http://localhost:9200 | Stockage des logs |

---

## 🔌 Endpoints API

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie que l'API et le modèle sont en ligne |
| `/predict` | POST | Prédit la survie d'un passager |
| `/retrain` | POST | Réentraîne avec nouveaux hyperparamètres |
| `/add-data` | POST | Ajoute 1 passager + relance le pipeline |
| `/add-multiple-rows` | POST | Ajoute plusieurs passagers + relance |
| `/upload-file` | POST | Importe CSV ou Excel + relance le pipeline |
| `/run-compare` | POST | Compare LogisticRegression / RandomForest / SVM |
| `/dataset-info` | GET | Statistiques du dataset actuel |

### Exemple — Prédiction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1, "Age": 29.0, "SibSp": 0, "Parch": 0,
    "Fare": 211.3, "Sex_male": false,
    "Embarked_Q": false, "Embarked_S": true
  }'
```

Réponse :
```json
{
  "prediction": 1,
  "survived": "Yes",
  "probability": 0.8742
}
```

---

## 🛠️ Commandes Makefile

```bash
# ── Installation ─────────────────────────────
make install          # Installe les dépendances

# ── Pipeline ML ──────────────────────────────
make prepare          # Prépare les données
make train            # Entraîne le modèle
make evaluate         # Évalue le modèle
make all              # Pipeline complet

# ── CI (Intégration Continue) ────────────────
make format           # Formate le code (black)
make lint             # Vérifie la qualité (flake8)
make security         # Analyse de sécurité (bandit)
make test             # Lance les tests (pytest)
make coverage         # Tests + rapport de couverture
make ci               # CI complète (format+lint+security+coverage)
make clean            # Supprime les fichiers temporaires

# ── API & Services ───────────────────────────
make run-api          # Lance FastAPI (port 8000)
make mlflow-ui        # Lance MLflow UI (port 5000)
make mlflow-server    # Lance MLflow en mode serveur
make monitoring-up    # Lance Elasticsearch + Kibana
make monitoring-down  # Arrête Elasticsearch + Kibana

# ── Docker ───────────────────────────────────
make docker-build     # Construit l'image Docker
make docker-run       # Lance le conteneur
make docker-push      # Publie sur Docker Hub
```

---

## 🖥️ Interface React

L'interface React propose **3 méthodes** pour ajouter des données et déclencher automatiquement le pipeline :

### Méthode 1 — Saisie manuelle
- Formulaire complet avec tous les champs du dataset
- Ajout de plusieurs passagers simultanément (➕)
- Aperçu CSV en temps réel sous chaque ligne

### Méthode 2 — Import CSV
- Glisser-déposer ou sélectionner un fichier `.csv`
- Validation automatique des colonnes requises
- Fusion avec le dataset existant

### Méthode 3 — Import Excel
- Support des fichiers `.xlsx` et `.xls`
- Un fichier de test `titanic_test_data.xlsx` est fourni (15 passagers)

### Résultats affichés
- Accuracy du modèle en temps réel
- Classification Report complet
- Graphique de comparaison des 3 modèles

---

## 📚 Ateliers réalisés

| Atelier | Description | Excellence |
|---|---|---|
| **Atelier 2** | Modularisation du code en fonctions Python | Tests unitaires 13/13 + Logging |
| **Atelier 3** | Création du Makefile CI/CD | `make ci` (format+lint+security+tests) |
| **Atelier 4** | API REST avec FastAPI | `/health` + Validation Pydantic + `/retrain` |
| **Atelier 5** | Suivi MLflow des expériences | Comparaison RandomForest / LR / SVM |
| **Atelier 6** | Conteneurisation Docker | `.env` + `.dockerignore` |
| **Atelier 7** | Monitoring Elasticsearch + Kibana | Dashboard temps réel + métriques système |
| **Bonus** | Interface React CI/CD automatique | 3 méthodes d'ajout + résultats en temps réel |

---

## 👩‍💻 Auteur

**Oumaima Souissi** — DS2 — MLOps  
Esprit School of Engineering — 2025/2026

---

## 📄 Licence

Ce projet est réalisé dans le cadre académique du cours MLOps — Esprit School of Engineering.