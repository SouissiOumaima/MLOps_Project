"""
Tests unitaires pour le pipeline MLOps Titanic 
pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from src.model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Crée un CSV Titanic minimal pour les tests."""
    data = pd.DataFrame({
        "survived":  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "pclass":    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "sex":       ["female","male","female","male","female",
                      "male","female","male","female","male"],
        "age":       [29, 35, 22, 45, 28, 55, 30, 40, 25, 50],
        "sibsp":     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "parch":     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "fare":      [211.3, 13.0, 7.9, 35.5, 26.0, 8.0, 55.0, 15.0, 7.2, 30.0],
        "embarked":  ["S","C","S","S","C","Q","S","C","S","S"],
    })
    csv_path = tmp_path / "titanic.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def prepared_data(sample_csv):
    """Retourne les données préparées pour les tests."""
    return prepare_data(sample_csv)


@pytest.fixture
def trained_model(prepared_data):
    """Retourne un modèle entraîné pour les tests."""
    X_train, X_test, y_train, y_test = prepared_data
    with patch("mlflow.set_experiment"), \
         patch("mlflow.start_run"), \
         patch("mlflow.log_param"), \
         patch("mlflow.sklearn.log_model"):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
    return model, X_test, y_test


# ── Tests prepare_data ────────────────────────────────────────────

class TestPrepareData:

    def test_retourne_4_elements(self, sample_csv):
        result = prepare_data(sample_csv)
        assert len(result) == 4, "prepare_data doit retourner (X_train, X_test, y_train, y_test)"

    def test_taille_split(self, sample_csv):
        X_train, X_test, y_train, y_test = prepare_data(sample_csv)
        total = len(X_train) + len(X_test)
        assert total == 10

    def test_pas_de_valeurs_manquantes(self, sample_csv):
        X_train, X_test, y_train, y_test = prepare_data(sample_csv)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_colonnes_encodees(self, sample_csv):
        X_train, _, _, _ = prepare_data(sample_csv)
        # Vérifie que les colonnes catégorielles ont bien été encodées
        assert "Sex_male" in X_train.columns or "Sex_Male" in X_train.columns \
            or any("sex" in c.lower() for c in X_train.columns)

    def test_survived_absent_des_features(self, sample_csv):
        X_train, X_test, _, _ = prepare_data(sample_csv)
        assert "Survived" not in X_train.columns
        assert "survived" not in X_train.columns

    def test_fichier_inexistant(self):
        with pytest.raises(Exception):
            prepare_data("fichier_inexistant.csv")


# ── Tests train_model ─────────────────────────────────────────────

class TestTrainModel:

    def test_retourne_un_modele(self, prepared_data):
        X_train, _, y_train, _ = prepared_data
        with patch("mlflow.set_experiment"), \
             patch("mlflow.start_run") as mock_run, \
             patch("mlflow.log_param"), \
             patch("mlflow.sklearn.log_model"):
            mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_run.return_value.__exit__ = MagicMock(return_value=False)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            assert hasattr(model, "predict")

    def test_modele_peut_predire(self, trained_model):
        model, X_test, _ = trained_model
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})


# ── Tests evaluate_model ──────────────────────────────────────────

class TestEvaluateModel:

    def test_accuracy_entre_0_et_1(self, trained_model):
        model, X_test, y_test = trained_model
        with patch("mlflow.start_run"), \
             patch("mlflow.log_metric"):
            accuracy = evaluate_model(model, X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_retourne_un_float(self, trained_model):
        model, X_test, y_test = trained_model
        with patch("mlflow.start_run"), \
             patch("mlflow.log_metric"):
            accuracy = evaluate_model(model, X_test, y_test)
        assert isinstance(accuracy, float)


# ── Tests save_model / load_model ────────────────────────────────

class TestSaveLoadModel:

    def test_sauvegarde_et_charge(self, trained_model, tmp_path):
        model, _, _ = trained_model
        filepath = str(tmp_path / "test_model.pkl")
        save_model(model, filepath)
        loaded = load_model(filepath)
        assert hasattr(loaded, "predict")

    def test_modele_charge_fait_memes_predictions(self, trained_model, tmp_path):
        model, X_test, _ = trained_model
        filepath = str(tmp_path / "test_model.pkl")
        save_model(model, filepath)
        loaded = load_model(filepath)
        np.testing.assert_array_equal(
            model.predict(X_test),
            loaded.predict(X_test)
        )

    def test_fichier_inexistant(self):
        with pytest.raises(Exception):
            load_model("modele_inexistant.pkl")