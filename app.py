"""
API REST FastAPI — Projet MLOps Titanic
"""

import os
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

from src.model_pipeline import (
    retrain_model, prepare_data, train_model,
    save_model, compare_models,
)

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
        api_logger.error("Modèle introuvable : '%s'.", MODEL_PATH)
    yield


app = FastAPI(title="MLOps Titanic API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PassengerData(BaseModel):
    Pclass:     int   = Field(..., ge=1, le=3)
    Age:        float = Field(..., ge=0.0, le=120.0)
    SibSp:      int   = Field(..., ge=0, le=10)
    Parch:      int   = Field(..., ge=0, le=10)
    Fare:       float = Field(..., ge=0.0)
    Sex_male:   bool  = Field(...)
    Embarked_Q: bool  = Field(...)
    Embarked_S: bool  = Field(...)


class PredictionResponse(BaseModel):
    prediction:  int
    survived:    str
    probability: float


class RetrainRequest(BaseModel):
    max_iter: int   = Field(default=1000, ge=100, le=5000)
    C:        float = Field(default=1.0, ge=0.01, le=100.0)
    solver:   str   = Field(default="lbfgs")

    @field_validator("solver")
    @classmethod
    def solver_valide(cls, v):
        if v not in ["lbfgs", "liblinear", "saga", "newton-cg", "sag"]:
            raise ValueError("Solver invalide.")
        return v


class RetrainResponse(BaseModel):
    message:  str
    max_iter: int
    C:        float
    solver:   str


class NewPassenger(BaseModel):
    pclass:   int   = Field(..., ge=1, le=3)
    survived: int   = Field(..., ge=0, le=1)
    name:     str   = Field(default="Unknown")
    sex:      str   = Field(...)
    age:      float = Field(default=None)
    sibsp:    int   = Field(default=0, ge=0, le=10)
    parch:    int   = Field(default=0, ge=0, le=10)
    ticket:   str   = Field(default="UNKNOWN")
    fare:     float = Field(default=0.0, ge=0)
    cabin:    str   = Field(default=None)
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


@app.get("/health", tags=["Monitoring"])
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {"status": "ok", "model_loaded": True,
            "model_path": MODEL_PATH, "api_version": "1.0.0"}


@app.get("/model-features", tags=["Debug"])
def model_features():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {"features": list(model.feature_names_in_)}


@app.get("/dataset-info", tags=["Pipeline"])
def dataset_info():
    try:
        df  = pd.read_csv(DATA_PATH)
        col = next((c for c in ["survived", "Survived"] if c in df.columns), None)
        return {
            "total_rows":   len(df),
            "survived":     int(df[col].sum())        if col else 0,
            "not_survived": int((df[col] == 0).sum()) if col else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(data: PassengerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    try:
        input_df     = pd.DataFrame([data.model_dump()])
        input_df     = input_df.rename(columns={"SibSp": "Sibsp"})
        prediction   = int(model.predict(input_df)[0])
        probability  = round(float(model.predict_proba(input_df)[0][prediction]), 4)
        survived_str = "Yes" if prediction == 1 else "No"
        return PredictionResponse(prediction=prediction,
                                  survived=survived_str,
                                  probability=probability)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse, tags=["Entraînement"])
def retrain(params: RetrainRequest):
    global model
    try:
        X_train, X_test, y_train, y_test = prepare_data()
        model = retrain_model(X_train, y_train, max_iter=params.max_iter,
                              C=params.C, solver=params.solver)
        joblib.dump(model, MODEL_PATH)
        return RetrainResponse(message="Modèle réentraîné avec succès.",
                               max_iter=params.max_iter, C=params.C,
                               solver=params.solver)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-data", tags=["Pipeline"])
def add_data(passenger: NewPassenger):
    global model
    try:
        df = pd.read_csv(DATA_PATH)
        df = pd.concat([df, pd.DataFrame([{
            "pclass": passenger.pclass, "survived": passenger.survived,
            "name": passenger.name,     "sex": passenger.sex,
            "age": passenger.age,       "sibsp": passenger.sibsp,
            "parch": passenger.parch,   "ticket": passenger.ticket,
            "fare": passenger.fare,     "cabin": passenger.cabin,
            "embarked": passenger.embarked,
        }])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)

        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model    = train_model(X_train, y_train, X_test, y_test)
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report   = classification_report(y_test, y_pred, output_dict=True)
        save_model(model, MODEL_PATH)

        return {"message": "Pipeline relancé avec succès !", "total_rows": len(df),
                "accuracy": round(accuracy, 4), "report": report,
                "model_type": "LogisticRegression"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-multiple-rows", tags=["Pipeline"])
def add_multiple_rows(passengers: list[NewPassenger]):
    global model
    try:
        df = pd.read_csv(DATA_PATH)
        new_rows = [{"pclass": p.pclass, "survived": p.survived, "name": p.name,
                     "sex": p.sex, "age": p.age, "sibsp": p.sibsp, "parch": p.parch,
                     "ticket": p.ticket, "fare": p.fare, "cabin": p.cabin,
                     "embarked": p.embarked} for p in passengers]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)

        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model    = train_model(X_train, y_train, X_test, y_test)
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report   = classification_report(y_test, y_pred, output_dict=True)
        save_model(model, MODEL_PATH)

        return {"message": f"{len(new_rows)} lignes ajoutées avec succès !",
                "rows_added": len(new_rows), "total_rows": len(df),
                "accuracy": round(accuracy, 4), "report": report,
                "model_type": "LogisticRegression"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-file", tags=["Pipeline"])
async def upload_file(file: UploadFile = File(...)):
    global model
    try:
        contents = await file.read()
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            new_df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            new_df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400,
                                detail="Format non supporté. CSV ou Excel uniquement.")

        new_df.columns = new_df.columns.str.strip().str.lower()
        missing = [c for c in ["pclass", "survived", "sex", "fare"]
                   if c not in new_df.columns]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"Colonnes manquantes : {missing}")

        existing_df = pd.read_csv(DATA_PATH)
        merged_df   = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df.to_csv(DATA_PATH, index=False)

        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model    = train_model(X_train, y_train, X_test, y_test)
        y_pred   = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report   = classification_report(y_test, y_pred, output_dict=True)
        save_model(model, MODEL_PATH)

        return {"message": f"Fichier '{file.filename}' importé avec succès !",
                "rows_added": len(new_df), "total_rows": len(merged_df),
                "accuracy": round(accuracy, 4), "report": report,
                "model_type": "LogisticRegression"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-compare", tags=["Pipeline"])
def run_compare():
    try:
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        results    = compare_models(X_train, y_train, X_test, y_test)
        comparison = []
        for name, info in results.items():
            y_pred = info["model"].predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            comparison.append({"model": name, "accuracy": info["accuracy"],
                                "report": report})
        return {"comparison": comparison}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))