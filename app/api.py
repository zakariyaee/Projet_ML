from contextlib import asynccontextmanager
import json
import io
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.responses import HTMLResponse, StreamingResponse

# Variables globales pour le modele et preprocessor
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model, preprocessor and metadata
    try:
        ml_models["model"] = joblib.load("models/final_model.joblib")
        ml_models["preprocessor"] = joblib.load("models/preprocessor.joblib")
        with open("models/final_model_metadata.json", "r") as f:
            ml_models["metadata"] = json.load(f)
        with open("models/tuning_metadata.json", "r") as f:
            ml_models["tuning_metadata"] = json.load(f)
    except Exception as e:
        print(f"Error loading models or metadata: {e}")
        # On ne throw pas ici pour permettre a /health de repondre 500,
        # mais dans un environnement de prod on pourrait throw
    yield
    # Cleanup si necessaire
    ml_models.clear()

app = FastAPI(
    title="API Classification PHAs",
    description="API REST pour la prédiction des astéroïdes potentiellement dangereux (PHAs).",
    version="1.0.0",
    lifespan=lifespan
)

# Schema Pydantic pour la prediction unitaire
class AsteroidFeatures(BaseModel):
    absolute_magnitude_h: float
    is_sentry_object: int
    relative_velocity_km_per_second: float
    miss_distance_astronomical: float
    orbiting_body: str
    n_approaches: int
    min_miss_distance_au: float
    max_velocity_km_s: float
    semi_major_axis: float
    inclination: float
    perihelion_distance: float
    perihelion_argument: float
    orbit_uncertainty: float
    minimum_orbit_intersection: float
    data_arc_in_days: float
    orbit_class_type: str
    diameter_mean_km: float
    diameter_uncertainty: float
    perihelion_to_aphelion_ratio: float
    threat_ratio: float
    velocity_distance_ratio: float
    observation_reliability: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "absolute_magnitude_h": 16.53,
                "is_sentry_object": 0,
                "relative_velocity_km_per_second": 26.25,
                "miss_distance_astronomical": 0.106,
                "orbiting_body": "EARTH",
                "n_approaches": 18,
                "min_miss_distance_au": 0.039,
                "max_velocity_km_s": 34.13,
                "semi_major_axis": 1.078,
                "inclination": 22.8,
                "perihelion_distance": 0.186,
                "perihelion_argument": 31.43,
                "orbit_uncertainty": 0.0,
                "minimum_orbit_intersection": 0.033,
                "data_arc_in_days": 27807.0,
                "orbit_class_type": "APO",
                "diameter_mean_km": 2.12,
                "diameter_uncertainty": 1.62,
                "perihelion_to_aphelion_ratio": 0.094,
                "threat_ratio": 0.015,
                "velocity_distance_ratio": 245.43,
                "observation_reliability": 0.0
            }
        }
    }

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float
    confidence: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>API PHAs</title>
        </head>
        <body>
            <h1>Bienvenue sur l'API de Classification des Astéroïdes Potentiellement Dangereux (PHAs)</h1>
            <p>Accédez à la <a href="/docs">Documentation Swagger (Interface de test)</a></p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    if "model" not in ml_models or ml_models["model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok", "message": "API and model are operational"}

@app.get("/model/info")
async def model_info():
    if "metadata" not in ml_models:
        raise HTTPException(status_code=500, detail="Metadata not loaded")
    
    metadata = ml_models["metadata"]
    return {
        "model_key": metadata.get("model_key"),
        "threshold": metadata.get("threshold"),
        "strategy": metadata.get("strategy"),
        "test_report": metadata.get("test_report")
    }

def process_prediction(df: pd.DataFrame) -> tuple[int, float, float, str]:
    model_dict = ml_models["model"]
    pipeline = model_dict["pipeline"]
    threshold = ml_models["metadata"]["threshold"]
    
    # Prediction (pipeline gère le preprocessing automatiquement)
    prob = pipeline.predict_proba(df)[0][1]
    pred_class = 1 if prob >= threshold else 0
    
    # Confidence level
    distance = abs(prob - threshold)
    if distance > 0.3:
        confidence = "high"
    elif distance > 0.1:
        confidence = "medium"
    else:
        confidence = "low"
        
    return pred_class, prob, threshold, confidence

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: AsteroidFeatures):
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    # Convert input to DataFrame
    df = pd.DataFrame([features.dict()])
    
    try:
        pred_class, prob, threshold, confidence = process_prediction(df)
        
        prediction_label = "PHA (Risque élevé)" if pred_class == 1 else "Non-PHA (Risque faible)"
        
        return PredictionResponse(
            prediction=prediction_label,
            probability=float(prob),
            threshold=float(threshold),
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Copie pour la reponse
        result_df = df.copy()
        
        # Supprimer is_potentially_hazardous si elle est la (on veut predire)
        features_df = df.copy()
        if "is_potentially_hazardous" in features_df.columns:
            features_df = features_df.drop("is_potentially_hazardous", axis=1)
            
        model_dict = ml_models["model"]
        pipeline = model_dict["pipeline"]
        threshold = ml_models["metadata"]["threshold"]
        
        probs = pipeline.predict_proba(features_df)[:, 1]
        
        preds = [1 if p >= threshold else 0 for p in probs]
        labels = ["PHA" if p == 1 else "Non-PHA" for p in preds]
        
        result_df["predicted_probability"] = probs
        result_df["predicted_class"] = labels
        result_df["applied_threshold"] = threshold
        
        # Convert back to CSV
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
