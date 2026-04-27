"""
API REST FastAPI — Projet MLOps Titanic
Endpoints : /health, /predict, /retrain, /add-data, /run-compare, /dataset-info
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import UploadFile, File
import io
import openpyxl
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

from src.model_pipeline import (
    retrain_model, prepare_data, train_model,
    save_model, compare_models, logger,
)

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/titanic_model.pkl")
DATA_PATH  = os.getenv("DATA_PATH",  "data/titanic.csv")

api_logger = logging.getLogger("fastapi_app")
model      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(MODEL_PATH)
        api_logger.info("Modèle chargé depuis '%s'.", MODEL_PATH)
    except FileNotFoundError:
        api_logger.error("Modèle introuvable : '%s'. Lancez 'make train'.", MODEL_PATH)
    yield
    api_logger.info("Arrêt de l'API.")


app = FastAPI(
    title="MLOps Titanic API",
    description="API REST pour la prédiction de survie sur le Titanic.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class PassengerData(BaseModel):
    Pclass:     int   = Field(..., ge=1,   le=3)
    Age:        float = Field(..., ge=0.0, le=120.0)
    SibSp:      int   = Field(..., ge=0,   le=10)
    Parch:      int   = Field(..., ge=0,   le=10)
    Fare:       float = Field(..., ge=0.0)
    Sex_male:   bool  = Field(...)
    Embarked_Q: bool  = Field(...)
    Embarked_S: bool  = Field(...)

    @field_validator("Age")
    @classmethod
    def age_valide(cls, v):
        if v < 0:
            raise ValueError("L'âge ne peut pas être négatif.")
        return round(v, 1)

    @field_validator("Fare")
    @classmethod
    def fare_valide(cls, v):
        if v < 0:
            raise ValueError("Le prix du billet ne peut pas être négatif.")
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Pclass": 1, "Age": 29.0, "SibSp": 0, "Parch": 0,
                "Fare": 211.3, "Sex_male": False,
                "Embarked_Q": False, "Embarked_S": True
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction:  int
    survived:    str
    probability: float


class RetrainRequest(BaseModel):
    max_iter: int   = Field(default=1000, ge=100,  le=5000)
    C:        float = Field(default=1.0,  ge=0.01, le=100.0)
    solver:   str   = Field(default="lbfgs")

    @field_validator("solver")
    @classmethod
    def solver_valide(cls, v):
        valides = ["lbfgs", "liblinear", "saga", "newton-cg", "sag"]
        if v not in valides:
            raise ValueError(f"Solver invalide. Choisir parmi : {valides}")
        return v


class RetrainResponse(BaseModel):
    message:  str
    max_iter: int
    C:        float
    solver:   str


class NewPassenger(BaseModel):
    pclass:   int   = Field(..., ge=1, le=3)
    survived: int   = Field(..., ge=0, le=1)
    sex:      str   = Field(...)
    age:      float = Field(..., ge=0, le=120)
    sibsp:    int   = Field(..., ge=0, le=10)
    parch:    int   = Field(..., ge=0, le=10)
    fare:     float = Field(..., ge=0)
    embarked: str   = Field(...)

    @field_validator("sex")
    @classmethod
    def sex_valide(cls, v):
        if v.lower() not in ["male", "female"]:
            raise ValueError("Sex doit être 'male' ou 'female'")
        return v.lower()

    @field_validator("embarked")
    @classmethod
    def embarked_valide(cls, v):
        if v.upper() not in ["S", "C", "Q"]:
            raise ValueError("Embarked doit être S, C ou Q")
        return v.upper()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Vérifie que l'API est en ligne et que le modèle est bien chargé."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {
        "status":       "ok",
        "model_loaded": True,
        "model_path":   MODEL_PATH,
        "api_version":  "1.0.0"
    }


@app.get("/dataset-info", tags=["Pipeline"])
def dataset_info():
    """Retourne des informations sur le dataset actuel."""
    try:
        df = pd.read_csv(DATA_PATH)
        survived_col = next(
            (c for c in ["survived", "Survived"] if c in df.columns), None
        )
        return {
            "total_rows":   len(df),
            "survived":     int(df[survived_col].sum())        if survived_col else 0,
            "not_survived": int((df[survived_col] == 0).sum()) if survived_col else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(data: PassengerData):
    """Prédit la survie d'un passager du Titanic."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    try:
        input_df     = pd.DataFrame([data.model_dump()])
        prediction   = int(model.predict(input_df)[0])
        probability  = round(float(model.predict_proba(input_df)[0][prediction]), 4)
        survived_str = "Yes" if prediction == 1 else "No"
        api_logger.info("Prédiction : %s (%.2f%%)", survived_str, probability * 100)
        return PredictionResponse(
            prediction=prediction, survived=survived_str, probability=probability
        )
    except Exception as e:
        api_logger.error("Erreur prédiction : %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse, tags=["Entraînement"])
def retrain(params: RetrainRequest):
    """Réentraîne le modèle avec de nouveaux hyperparamètres."""
    global model
    try:
        api_logger.info(
            "Réentraînement — max_iter=%d, C=%.2f, solver=%s",
            params.max_iter, params.C, params.solver
        )
        X_train, X_test, y_train, y_test = prepare_data()
        model = retrain_model(
            X_train, y_train,
            max_iter=params.max_iter,
            C=params.C,
            solver=params.solver,
        )
        joblib.dump(model, MODEL_PATH)
        return RetrainResponse(
            message="Modèle réentraîné et sauvegardé avec succès.",
            max_iter=params.max_iter, C=params.C, solver=params.solver,
        )
    except Exception as e:
        api_logger.error("Erreur réentraînement : %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-data", tags=["Pipeline"])
def add_data(passenger: NewPassenger):
    """
    Ajoute une ligne au dataset et relance automatiquement le pipeline complet.
    Retourne accuracy + classification report + infos dataset.
    """
    global model
    try:
        # 1. Lire le CSV et ajouter la nouvelle ligne
        df = pd.read_csv(DATA_PATH)
        new_row = {
            "pclass": passenger.pclass, "survived": passenger.survived,
            "name": "New Passenger",    "sex": passenger.sex,
            "age": passenger.age,       "sibsp": passenger.sibsp,
            "parch": passenger.parch,   "ticket": "NEW",
            "fare": passenger.fare,     "cabin": None,
            "embarked": passenger.embarked,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        api_logger.info("Ligne ajoutée. Total : %d lignes.", len(df))

        # 2. Relancer le pipeline complet
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train, X_test, y_test)

        # 3. Calculer les métriques
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report   = classification_report(y_test, y_pred, output_dict=True)

        # 4. Sauvegarder
        save_model(model, MODEL_PATH)
        api_logger.info("Pipeline terminé. Accuracy : %.4f", accuracy)

        return {
            "message":    "Pipeline relancé avec succès !",
            "total_rows": len(df),
            "accuracy":   round(accuracy, 4),
            "report":     report,
            "model_type": "LogisticRegression",
        }
    except Exception as e:
        api_logger.error("Erreur pipeline : %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-compare", tags=["Pipeline"])
def run_compare():
    """Compare LogisticRegression, RandomForest et SVM."""
    global model
    try:
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        results    = compare_models(X_train, y_train, X_test, y_test)
        comparison = []
        for name, info in results.items():
            y_pred = info["model"].predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            comparison.append({
                "model":    name,
                "accuracy": info["accuracy"],
                "report":   report,
            })
        api_logger.info("Comparaison terminée — %d modèles.", len(comparison))
        return {"comparison": comparison}
    except Exception as e:
        api_logger.error("Erreur comparaison : %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add-multiple-rows", tags=["Pipeline"])
def add_multiple_rows(passengers: list[NewPassenger]):
    """
    Ajoute plusieurs lignes manuellement et relance le pipeline.
    """
    global model
    try:
        df = pd.read_csv(DATA_PATH)

        new_rows = []
        for p in passengers:
            new_rows.append({
                "pclass": p.pclass, "survived": p.survived,
                "name": "New Passenger", "sex": p.sex,
                "age": p.age, "sibsp": p.sibsp,
                "parch": p.parch, "ticket": "NEW",
                "fare": p.fare, "cabin": None,
                "embarked": p.embarked,
            })

        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        api_logger.info("%d lignes ajoutées. Total : %d.", len(new_rows), len(df))

        # Relancer le pipeline
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train, X_test, y_test)
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report   = classification_report(y_test, y_pred, output_dict=True)
        save_model(model, MODEL_PATH)

        return {
            "message":      f"{len(new_rows)} lignes ajoutées avec succès !",
            "rows_added":   len(new_rows),
            "total_rows":   len(df),
            "accuracy":     round(accuracy, 4),
            "report":       report,
            "model_type":   "LogisticRegression",
        }
    except Exception as e:
        api_logger.error("Erreur add-multiple : %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-file", tags=["Pipeline"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload un fichier CSV ou Excel, fusionne avec le dataset
    et relance le pipeline automatiquement.
    """
    global model
    try:
        contents = await file.read()
        filename = file.filename.lower()

        # Lire le fichier selon son extension
        if filename.endswith(".csv"):
            new_df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            new_df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Format non supporté. Utilisez CSV ou Excel (.xlsx, .xls)"
            )

        api_logger.info(
            "Fichier '%s' reçu — %d lignes.", file.filename, len(new_df)
        )

        # Normaliser les colonnes
        new_df.columns = new_df.columns.str.strip().str.lower()

        # Vérifier les colonnes requises
        required = ["pclass", "survived", "sex", "age", "fare"]
        missing  = [c for c in required if c not in new_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes : {missing}"
            )

        # Fusionner avec le dataset existant
        existing_df = pd.read_csv(DATA_PATH)
        merged_df   = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df.to_csv(DATA_PATH, index=False)
        api_logger.info(
            "Fusion terminée. Total : %d lignes.", len(merged_df)
        )

        # Relancer le pipeline
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model    = train_model(X_train, y_train, X_test, y_test)
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report   = classification_report(y_test, y_pred, output_dict=True)
        save_model(model, MODEL_PATH)

        return {
            "message":      f"Fichier '{file.filename}' importé avec succès !",
            "rows_added":   len(new_df),
            "total_rows":   len(merged_df),
            "accuracy":     round(accuracy, 4),
            "report":       report,
            "model_type":   "LogisticRegression",
        }

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Erreur upload : %s", e)
        raise HTTPException(status_code=500, detail=str(e))