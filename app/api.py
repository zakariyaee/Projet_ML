from contextlib import asynccontextmanager
import json
import io
import logging
from datetime import datetime

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.responses import HTMLResponse, StreamingResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pha_api")

# ── Variables globales pour le modèle et preprocessor ────────────────────────
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et les métadonnées UNE SEULE FOIS au démarrage."""
    try:
        ml_models["model"] = joblib.load("models/final_model.joblib")
        ml_models["preprocessor"] = joblib.load("models/preprocessor.joblib")
        with open("models/final_model_metadata.json", "r", encoding="utf-8") as f:
            ml_models["metadata"] = json.load(f)
        with open("models/tuning_metadata.json", "r", encoding="utf-8") as f:
            ml_models["tuning_metadata"] = json.load(f)
        logger.info("Modèle et métadonnées chargés avec succès.")
    except Exception as e:
        logger.error(f"Erreur au chargement du modèle : {e}")
    yield
    ml_models.clear()

app = FastAPI(
    title="API Classification PHAs",
    description=(
        "API REST pour la prédiction des astéroïdes potentiellement dangereux (PHAs).\n\n"
        "**Projet ML** — ENSA Tétouan — Pr. Y. EL YOUNOUSSI — 2025-2026\n\n"
        "Endpoints :\n"
        "- `GET /` : Page d'accueil\n"
        "- `GET /health` : Vérification de l'état de l'API\n"
        "- `GET /model/info` : Métadonnées du modèle\n"
        "- `POST /predict` : Prédiction unitaire\n"
        "- `POST /predict/batch` : Prédiction par lot (CSV)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schémas Pydantic ─────────────────────────────────────────────────────────

class AsteroidFeatures(BaseModel):
    """Caractéristiques d'un astéroïde pour la prédiction."""
    absolute_magnitude_h: float = Field(..., description="Magnitude absolue (H)")
    is_sentry_object: int = Field(..., ge=0, le=1, description="Objet Sentry (0 ou 1)")
    relative_velocity_km_per_second: float = Field(..., gt=0, description="Vitesse relative (km/s)")
    miss_distance_astronomical: float = Field(..., gt=0, description="Distance de passage (UA)")
    orbiting_body: str = Field(..., description="Corps orbité (EARTH, MARS, JUPTR, VENUS, UNKNOWN)")
    n_approaches: int = Field(..., ge=0, description="Nombre d'approches enregistrées")
    min_miss_distance_au: float = Field(..., ge=0, description="Distance de passage minimale (UA)")
    max_velocity_km_s: float = Field(..., gt=0, description="Vitesse maximale historique (km/s)")
    semi_major_axis: float = Field(..., gt=0, description="Demi-grand axe (UA)")
    inclination: float = Field(..., ge=0, description="Inclinaison orbitale (degrés)")
    perihelion_distance: float = Field(..., ge=0, description="Distance au périhélie (UA)")
    perihelion_argument: float = Field(..., description="Argument du périhélie (degrés)")
    orbit_uncertainty: float = Field(..., ge=0, description="Incertitude orbitale")
    minimum_orbit_intersection: float = Field(..., ge=0, description="MOID (UA)")
    data_arc_in_days: float = Field(..., ge=0, description="Arc de données (jours)")
    orbit_class_type: str = Field(..., description="Classe d'orbite MPC (APO, ATE, AMO, IEO, UNKNOWN)")
    diameter_mean_km: float = Field(..., ge=0, description="Diamètre moyen (km)")
    diameter_uncertainty: float = Field(..., ge=0, description="Incertitude sur le diamètre (km)")
    perihelion_to_aphelion_ratio: float = Field(..., ge=0, le=1, description="Ratio Périhélie/Aphélie")
    threat_ratio: float = Field(..., ge=0, description="Ratio de menace (MOID / diamètre)")
    velocity_distance_ratio: float = Field(..., ge=0, description="Ratio vitesse / distance")
    observation_reliability: float = Field(..., ge=0, description="Fiabilité de l'observation")

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
    """Réponse de prédiction unitaire."""
    prediction: str = Field(..., description="Classe prédite (PHA ou Non-PHA)")
    probability: float = Field(..., description="Probabilité d'être un PHA")
    threshold: float = Field(..., description="Seuil de décision appliqué")
    confidence: str = Field(..., description="Niveau de confiance (high, medium, low)")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Page d'accueil", tags=["Général"])
async def root():
    """Page d'accueil avec informations sur l'API et lien vers la documentation Swagger."""
    return """
    <html>
        <head><title>API Classification PHAs</title></head>
        <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto;">
            <h1>🌍 API de Classification des Astéroïdes Potentiellement Dangereux (PHAs)</h1>
            <p>Cette API permet de prédire si un astéroïde est potentiellement dangereux
            en se basant sur ses caractéristiques orbitales et physiques.</p>
            <h2>Liens utiles</h2>
            <ul>
                <li><a href="/docs">📖 Documentation Swagger (Interface de test)</a></li>
                <li><a href="/redoc">📄 Documentation ReDoc</a></li>
                <li><a href="/health">❤️ État de santé de l'API</a></li>
                <li><a href="/model/info">🧠 Informations sur le modèle</a></li>
            </ul>
            <p><em>Projet ML — ENSA Tétouan — Pr. Y. EL YOUNOUSSI — 2025-2026</em></p>
        </body>
    </html>
    """


@app.get("/health", summary="Vérification de l'état", tags=["Général"])
async def health_check():
    """Vérifie que l'API et le modèle sont opérationnels (retourne 200 OK)."""
    if "model" not in ml_models or ml_models["model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok", "message": "API and model are operational"}


@app.get("/model/info", summary="Métadonnées du modèle", tags=["Modèle"])
async def model_info():
    """Retourne les métadonnées du modèle : type, seuil, stratégie, métriques."""
    if "metadata" not in ml_models:
        raise HTTPException(status_code=500, detail="Metadata not loaded")

    metadata = ml_models["metadata"]
    return {
        "model_key": metadata.get("model_key"),
        "threshold": metadata.get("threshold"),
        "strategy": metadata.get("strategy"),
        "target": metadata.get("target"),
        "test_report": metadata.get("test_report"),
    }


def process_prediction(df: pd.DataFrame) -> tuple:
    """Effectue la prédiction et retourne la classe, la probabilité, le seuil et la confiance."""
    model_dict = ml_models["model"]
    pipeline = model_dict["pipeline"]
    threshold = model_dict["optimal_threshold"]

    prob = pipeline.predict_proba(df)[0][1]
    pred_class = 1 if prob >= threshold else 0

    distance = abs(prob - threshold)
    if distance > 0.3:
        confidence = "high"
    elif distance > 0.1:
        confidence = "medium"
    else:
        confidence = "low"

    return pred_class, prob, threshold, confidence


@app.post("/predict", response_model=PredictionResponse, summary="Prédiction unitaire", tags=["Prédiction"])
async def predict_single(request: Request, features: AsteroidFeatures):
    """Reçoit les caractéristiques d'un astéroïde et retourne la classe prédite + probabilité."""
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])

    try:
        pred_class, prob, threshold, confidence = process_prediction(df)
        prediction_label = "PHA (Risque élevé)" if pred_class == 1 else "Non-PHA (Risque faible)"

        logger.info(
            f"POST /predict | client={request.client.host} | "
            f"prediction={prediction_label} | prob={prob:.4f} | threshold={threshold}"
        )

        return PredictionResponse(
            prediction=prediction_label,
            probability=float(prob),
            threshold=float(threshold),
            confidence=confidence,
        )
    except Exception as e:
        logger.error(f"POST /predict | Erreur : {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", summary="Prédiction par lot (CSV)", tags=["Prédiction"])
async def predict_batch(request: Request, file: UploadFile = File(...)):
    """Reçoit un fichier CSV, retourne un CSV enrichi des prédictions et probabilités."""
    if "model" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        result_df = df.copy()

        features_df = df.copy()
        if "is_potentially_hazardous" in features_df.columns:
            features_df = features_df.drop("is_potentially_hazardous", axis=1)

        model_dict = ml_models["model"]
        pipeline = model_dict["pipeline"]
        threshold = model_dict["optimal_threshold"]

        probs = pipeline.predict_proba(features_df)[:, 1]
        preds = [1 if p >= threshold else 0 for p in probs]
        labels = ["PHA" if p == 1 else "Non-PHA" for p in preds]

        result_df["predicted_probability"] = probs
        result_df["predicted_class"] = labels
        result_df["applied_threshold"] = threshold

        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        logger.info(
            f"POST /predict/batch | client={request.client.host} | "
            f"file={file.filename} | rows={len(df)} | PHAs={sum(preds)}"
        )

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"},
        )

    except Exception as e:
        logger.error(f"POST /predict/batch | Erreur : {e}")
        raise HTTPException(status_code=400, detail=str(e))
