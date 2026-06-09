# Classification des Astéroïdes Potentiellement Dangereux (PHAs)

Système de Machine Learning pour identifier les astéroïdes géocroiseurs potentiellement dangereux (PHAs) à partir de leurs caractéristiques orbitales et physiques. Le modèle est exposé via une API REST (FastAPI) et une interface utilisateur (Streamlit).

**Projet de fin de module Machine Learning** - ENSA Tétouan - Pr. Yacine EL YOUNOUSSI - 2025-2026

**Auteurs** : Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae

---

## Captures d'écran

### Interface de prédiction unitaire
L'utilisateur saisit les caractéristiques d'un astéroïde et obtient une prédiction claire ("ALERTE ROUGE" ou "SÉCURISÉ") avec la probabilité et le niveau de confiance.

### Documentation Swagger automatique
L'API expose une documentation interactive accessible à `http://localhost:8000/docs`, permettant de tester tous les endpoints directement depuis le navigateur.

---

## Architecture du dépôt

```
Projet_ML/
├── app/
│   ├── api.py                  # API REST FastAPI (5 endpoints)
│   └── ui.py                   # Interface Streamlit
├── data/
│   ├── dataset.csv             # Dataset complet (20 000 lignes)
│   └── processed/              # Train / Val / Test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── final_model.joblib      # Pipeline complet (preprocessing + XGBoost tuné + seuil)
│   ├── preprocessor.joblib     # ColumnTransformer sérialisé
│   ├── tuned_model.joblib      # Modèle après tuning
│   └── *.json / *.csv          # Métadonnées et résultats
├── notebooks/
│   ├── 01_discovery.ipynb      # Phase 1 : Cadrage et collecte
│   ├── 02_eda.ipynb            # Phase 2 : Analyse exploratoire
│   ├── 03_preprocessing.ipynb  # Phase 2 : Nettoyage et feature engineering
│   ├── 04_modeling.ipynb       # Phase 3 : Modélisation (12 configurations)
│   ├── 05_tuning.ipynb         # Phase 3 : Optimisation hyperparamètres
│   └── 06_evaluation.ipynb     # Phase 3 : Évaluation finale + seuil
├── src/
│   └── data_collection.py      # Script de collecte API NASA NeoWs
├── dockerfile                  # Image Docker (python:3.11-slim)
├── docker-compose.yml          # Orchestration API + UI
├── .dockerignore               # Exclusions Docker
├── requirements.txt            # Dépendances ML (versions figées)
├── requirements-docker.txt     # Dépendances Docker (ML + API + UI)
├── preprocessing_decisions.md  # Tableau de décisions preprocessing
├── cadrage.md                  # Cadrage métier Phase 1
├── DATASET.md                  # Documentation du dataset
└── README.md                   # Ce fichier
```

---

## Installation et lancement

### Option 1 : Sans Docker (développement local)

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-repo/Projet_ML.git
cd Projet_ML

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Installer les dépendances
pip install -r requirements-docker.txt

# 4. Lancer l'API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# 5. Dans un autre terminal, lancer l'interface Streamlit
streamlit run app/ui.py
```

L'API sera accessible à `http://localhost:8000` et l'interface à `http://localhost:8501`.

### Option 2 : Avec Docker (recommandé)

```bash
# 1. Construire et lancer les conteneurs
docker-compose up --build

# 2. Accéder aux services
# API :        http://localhost:8000
# Swagger :    http://localhost:8000/docs
# Interface :  http://localhost:8501
```

---

## Exemple d'utilisation

### Test avec curl

```bash
# Vérifier que l'API fonctionne
curl http://localhost:8000/health

# Prédiction unitaire
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Réponse attendue

```json
{
  "prediction": "PHA (Risque élevé)",
  "probability": 0.98,
  "threshold": 0.13,
  "confidence": "high"
}
```

### Prédiction par lot

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@test_api.csv" \
  --output predictions.csv
```

---

## Documentation Swagger

Une fois l'API lancée, la documentation interactive est accessible à :
- **Swagger UI** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc** : [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Modèle

| Élément | Détail |
|---------|--------|
| Algorithme | XGBoost (Gradient Boosting) |
| Features | 22 variables (orbitales, physiques, ingéniées) |
| Seuil de décision | Optimisé via matrice de coût asymétrique (FN=1000, FP=20) |
| Métrique principale | F1-score |
| Stratégie déséquilibre | Baseline avec class_weight + SMOTE + RandomUnderSampler comparés |

---

## Limites connues du modèle

1. **Biais temporel** : Le modèle est entraîné sur des données collectées à un instant donné via l'API NASA NeoWs. Les nouveaux astéroïdes découverts après cette date ne sont pas pris en compte.
2. **Dépendance aux features ingéniées** : Les variables `threat_ratio`, `velocity_distance_ratio`, etc. doivent être recalculées correctement pour de nouvelles données.
3. **Classe UNKNOWN** : Les astéroïdes avec `orbiting_body` ou `orbit_class_type` inconnus peuvent recevoir des prédictions moins fiables.
4. **Pas de mise à jour automatique** : Le modèle n'est pas ré-entraîné automatiquement ; un pipeline MLOps serait nécessaire en production.

---

## Licence

Ce projet est réalisé dans un cadre académique (ENSA Tétouan, 2025-2026). Toute réutilisation doit mentionner les auteurs et le contexte d'origine.
