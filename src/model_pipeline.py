"""
Pipeline MLOps - Fonctions modulaires pour le projet Titanic
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch
from datetime import datetime
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Lire les variables
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX_ML = os.getenv("ES_INDEX_METRICS", "mlflow-metrics")
ES_INDEX_SYS = os.getenv("ES_INDEX_SYSTEM", "system-metrics")
MODEL_PATH = os.getenv("MODEL_PATH", "models/titanic_model.pkl")

# ── Configuration du logger ───────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Affiche dans le terminal
        logging.FileHandler("logs/pipeline.log"),  # Sauvegarde dans un fichier
    ],
)
logger = logging.getLogger("model_pipeline")


# ── Elasticsearch ─────────────────────────────────────────────────────────────


def get_es_client():
    """Retourne un client Elasticsearch connecté via variable d'environnement."""
    return Elasticsearch(ES_HOST)


def log_to_elasticsearch(run_id: str, params: dict, metrics: dict) -> None:
    try:
        es = get_es_client()
        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
        }
        es.index(index=ES_INDEX_ML, document=doc)
        logger.info("Logs MLflow envoyés vers Elasticsearch.")
    except Exception as e:
        logger.warning("Elasticsearch indisponible : %s", e)


def log_system_metrics() -> None:
    try:
        es = get_es_client()
        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "system_metrics",
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
            "disk_percent": psutil.disk_usage("/").percent,
        }
        es.index(index=ES_INDEX_SYS, document=doc)
        logger.info("Métriques système envoyées vers Elasticsearch.")
    except Exception as e:
        logger.warning("Elasticsearch indisponible : %s", e)


# ── Pipeline ML ───────────────────────────────────────────────────────────────


def prepare_data(data_path: str = "data/titanic.csv"):
    """Charge et prétraite les données Titanic de façon robuste."""
    logger.info("Chargement des données depuis '%s'...", data_path)

    df = pd.read_csv(data_path)

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.strip().str.capitalize()
    logger.info("Colonnes disponibles : %s", df.columns.tolist())

    # Suppression sécurisée des colonnes inutiles
    cols_to_drop = ["Unnamed: 0", "Passengerid", "Name", "Ticket", "Cabin"]
    df = df.drop(
        columns=[col for col in cols_to_drop if col in df.columns], errors="ignore"
    )

    # Remplir les valeurs manquantes
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encodage des variables catégorielles
    if "Sex" in df.columns:
        df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Vérification de la cible
    if "Survived" not in df.columns:
        raise ValueError(
            "La colonne 'Survived' est absente du fichier ! Vérifie ton CSV."
        )

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    logger.info("Données prêtes : %d lignes, %d features.", len(X), X.shape[1])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, X_test=None, y_test=None):
    """Entraîne le modèle LogisticRegression avec suivi MLflow + envoi Elasticsearch."""
    logger.info("Démarrage de l'entraînement du modèle...")

    mlflow.set_experiment("Titanic_Survival")

    with mlflow.start_run(run_name="LogisticRegression_Training") as run:

        params = {
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42,
        }
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Modèle entraîné avec succès.")

        metrics = {}
        if X_test is not None and y_test is not None:
            accuracy = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", accuracy)
            metrics["accuracy"] = round(accuracy, 4)
            logger.info("Accuracy sur le jeu de test : %.4f", accuracy)

        mlflow.sklearn.log_model(model, "model")
        logger.info("Modèle enregistré dans MLflow.")

        # Envoi vers Elasticsearch
        log_to_elasticsearch(run.info.run_id, params, metrics)

        # Métriques système au moment de l'entraînement
        log_system_metrics()

        return model


def retrain_model(
    X_train, y_train, max_iter: int = 1000, C: float = 1.0, solver: str = "lbfgs"
):
    """Réentraîne le modèle avec nouveaux hyperparamètres et suivi MLflow."""
    logger.info("Réentraînement — max_iter=%d, C=%.2f, solver=%s", max_iter, C, solver)

    mlflow.set_experiment("Titanic_Survival")

    with mlflow.start_run(run_name=f"LogisticRegression_Retrain_C={C}") as run:

        params = {
            "model_type": "LogisticRegression",
            "max_iter": max_iter,
            "C": C,
            "solver": solver,
            "random_state": 42,
        }
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = LogisticRegression(
            max_iter=max_iter, C=C, solver=solver, random_state=42
        )
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "model")
        logger.info("Modèle réentraîné et enregistré dans MLflow.")

        # Envoi vers Elasticsearch
        log_to_elasticsearch(run.info.run_id, params, {})

        # Métriques système
        log_system_metrics()

        return model


def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle et logue l'accuracy dans MLflow."""
    logger.info("Évaluation du modèle...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with mlflow.start_run(run_name="Evaluation", nested=True):
        mlflow.log_metric("accuracy", accuracy)

    logger.info("Accuracy : %.4f", accuracy)
    logger.info("Rapport de classification :\n%s", report)

    return accuracy


def save_model(model, filepath: str = "models/titanic_model.pkl") -> None:
    """Sauvegarde le modèle entraîné avec joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Modèle sauvegardé : %s", filepath)


def load_model(filepath: str = "models/titanic_model.pkl"):
    """Charge un modèle sauvegardé."""
    logger.info("Chargement du modèle depuis '%s'...", filepath)
    return joblib.load(filepath)


def train_random_forest(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    n_estimators: int = 100,
    max_depth: int = None,
):
    """Entraîne un RandomForest avec suivi MLflow."""
    logger.info("Entraînement RandomForest — n_estimators=%d", n_estimators)
    mlflow.set_experiment("Titanic_Survival")

    with mlflow.start_run(run_name="RandomForest") as run:
        params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "random_state": 42,
        }
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        metrics = {}
        if X_test is not None and y_test is not None:
            accuracy = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", accuracy)
            metrics["accuracy"] = round(accuracy, 4)
            logger.info("RandomForest — Accuracy : %.4f", accuracy)

        mlflow.sklearn.log_model(model, "model")
        log_to_elasticsearch(run.info.run_id, params, metrics)
        log_system_metrics()

        return model


def train_svm(
    X_train, y_train, X_test=None, y_test=None, C: float = 1.0, kernel: str = "rbf"
):
    """Entraîne un SVM avec suivi MLflow."""
    logger.info("Entraînement SVM — C=%.2f, kernel=%s", C, kernel)
    mlflow.set_experiment("Titanic_Survival")

    with mlflow.start_run(run_name=f"SVM_kernel={kernel}") as run:
        params = {"model_type": "SVC", "C": C, "kernel": kernel}
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        model.fit(X_train, y_train)

        metrics = {}
        if X_test is not None and y_test is not None:
            accuracy = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", accuracy)
            metrics["accuracy"] = round(accuracy, 4)
            logger.info("SVM — Accuracy : %.4f", accuracy)

        mlflow.sklearn.log_model(model, "model")
        log_to_elasticsearch(run.info.run_id, params, metrics)
        log_system_metrics()

        return model


def compare_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Entraîne et compare LogisticRegression, RandomForest et SVM.
    Retourne un dictionnaire avec les résultats triés par accuracy.
    """
    logger.info("=== Comparaison des modèles ===")

    results = {}

    # LogisticRegression
    model_lr = train_model(X_train, y_train, X_test, y_test)
    results["LogisticRegression"] = {
        "model": model_lr,
        "accuracy": round(accuracy_score(y_test, model_lr.predict(X_test)), 4),
    }

    # RandomForest
    model_rf = train_random_forest(
        X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5
    )
    results["RandomForest"] = {
        "model": model_rf,
        "accuracy": round(accuracy_score(y_test, model_rf.predict(X_test)), 4),
    }

    # SVM
    model_svm = train_svm(X_train, y_train, X_test, y_test, C=1.0, kernel="rbf")
    results["SVM"] = {
        "model": model_svm,
        "accuracy": round(accuracy_score(y_test, model_svm.predict(X_test)), 4),
    }

    # Trier par accuracy décroissante
    results_sorted = dict(
        sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    )

    # Afficher le classement
    logger.info("=== Classement des modèles ===")
    for rank, (name, info) in enumerate(results_sorted.items(), 1):
        logger.info("%d. %s — Accuracy : %.4f", rank, name, info["accuracy"])

    # Identifier le meilleur modèle
    best_name = list(results_sorted.keys())[0]
    best_model = results_sorted[best_name]["model"]
    logger.info(
        "Meilleur modèle : %s (%.4f)", best_name, results_sorted[best_name]["accuracy"]
    )

    # Sauvegarder le meilleur modèle
    save_model(best_model)
    logger.info("Meilleur modèle sauvegardé.")

    return results_sorted
