"""
API REST FastAPI — Projet MLOps Titanic
Endpoints : /health, /predict, /retrain, /add-data, /run-compare, /dataset-info
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import UploadFile, File, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import joblib
import pandas as pd
from pydantic import BaseModel, Field, field_validator
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


# ✅ UNE SEULE INSTANCE APP
app = FastAPI(
    title="MLOps Titanic API",
    description="API REST pour la prédiction de survie sur le Titanic.",
    version="1.0.0",
    lifespan=lifespan,
)

# ✅ CORS CONFIG CORRECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS
# ══════════════════════════════════════════════════════════════════════════════

class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Age: float = Field(..., ge=0.0, le=120.0)
    SibSp: int = Field(..., ge=0, le=10)
    Parch: int = Field(..., ge=0, le=10)
    Fare: float = Field(..., ge=0.0)
    Sex_male: bool
    Embarked_Q: bool
    Embarked_S: bool


class PredictionResponse(BaseModel):
    prediction: int
    survived: str
    probability: float


class RetrainRequest(BaseModel):
    max_iter: int = Field(default=1000)
    C: float = Field(default=1.0)
    solver: str = Field(default="lbfgs")


class RetrainResponse(BaseModel):
    message: str
    max_iter: int
    C: float
    solver: str


class NewPassenger(BaseModel):
    pclass: int
    survived: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {"status": "ok"}


@app.get("/dataset-info")
def dataset_info():
    df = pd.read_csv(DATA_PATH)
    return {
        "total_rows": len(df)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PassengerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    input_df = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][prediction])

    return {
        "prediction": prediction,
        "survived": "Yes" if prediction == 1 else "No",
        "probability": round(probability, 4)
    }


@app.post("/retrain", response_model=RetrainResponse)
def retrain(params: RetrainRequest):
    global model

    X_train, X_test, y_train, y_test = prepare_data()

    model = retrain_model(
        X_train, y_train,
        max_iter=params.max_iter,
        C=params.C,
        solver=params.solver,
    )

    joblib.dump(model, MODEL_PATH)

    return {
        "message": "Modèle réentraîné",
        "max_iter": params.max_iter,
        "C": params.C,
        "solver": params.solver,
    }


@app.post("/run-compare")
def run_compare():
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    results = compare_models(X_train, y_train, X_test, y_test)

    return {"comparison": results}


@app.post("/add-data")
def add_data(passenger: NewPassenger):
    global model

    df = pd.read_csv(DATA_PATH)

    new_row = {
        "pclass": passenger.pclass,
        "survived": passenger.survived,
        "sex": passenger.sex,
        "age": passenger.age,
        "sibsp": passenger.sibsp,
        "parch": passenger.parch,
        "fare": passenger.fare,
        "embarked": passenger.embarked,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

    return {"message": "Ligne ajoutée"}