# Rapport Final — Classification des Astéroïdes Potentiellement Dangereux (PHAs)

**Module** : Machine Learning  
**Établissement** : ENSA Tétouan — 2ème année Cycle d'Ingénieurs GI  
**Professeur** : Pr. Y. EL YOUNOUSSI — 2025-2026  
**Auteurs** : Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae

---

## 1. Introduction et Contexte Métier (Phase 1)

### 1.1 Problématique

La surveillance des astéroïdes géocroiseurs (NEOs) est un enjeu majeur de défense planétaire. La NASA maintient une base de données de plus de 30 000 NEOs, mais identifier ceux qui sont **potentiellement dangereux (PHAs)** nécessite une analyse experte coûteuse en temps. Notre objectif est d'automatiser cette classification à l'aide du Machine Learning.

### 1.2 Définition d'un PHA

Un astéroïde est classé PHA par la NASA s'il satisfait deux critères simultanés :
- **MOID ≤ 0.05 UA** (Minimum Orbit Intersection Distance — il passe très près de l'orbite terrestre)
- **H ≤ 22** (Magnitude absolue — il est suffisamment grand, ≥ 140m de diamètre)

### 1.3 Objectif ML

Construire un classifieur binaire (`is_potentially_hazardous` : 0 ou 1) qui maximise le **rappel** (ne pas rater un PHA) tout en maintenant une **précision** raisonnable, le tout mesuré par le **F1-score** comme métrique principale.

### 1.4 Matrice de coût asymétrique

| | Prédit Non-PHA | Prédit PHA |
|---|---|---|
| **Réel Non-PHA** | TN (0) | FP (20 unités) |
| **Réel PHA** | FN (1 000 unités) | TP (0) |

Un faux négatif (rater un PHA) est **50× plus coûteux** qu'un faux positif (fausse alerte), ce qui justifie l'optimisation du seuil de décision en dessous de 0.5.

---

## 2. Description et Compréhension des Données (Phase 1)

### 2.1 Source

Données collectées via l'API **NASA NeoWs** (Near Earth Object Web Service) à l'aide du script `src/data_collection.py`. Le dataset final contient **20 000 lignes** et **27 colonnes**.

### 2.2 Variables principales

| Catégorie | Variables | Description |
|---|---|---|
| Physiques | `absolute_magnitude_h`, `estimated_diameter_*`, `diameter_mean_km` | Taille et brillance |
| Orbitales | `semi_major_axis`, `eccentricity`, `inclination`, `perihelion_distance` | Paramètres de l'orbite |
| Approche | `relative_velocity_km_per_second`, `miss_distance_astronomical`, `minimum_orbit_intersection` | Proximité et vitesse |
| Observation | `orbit_uncertainty`, `data_arc_in_days`, `n_approaches` | Fiabilité des mesures |
| Catégorielles | `orbiting_body`, `orbit_class_type` | Classification MPC |
| Cible | `is_potentially_hazardous` | 0 (non-PHA) ou 1 (PHA) |

### 2.3 Déséquilibre des classes

Le dataset présente un déséquilibre significatif : environ **87% de non-PHAs** contre **13% de PHAs**. Ce déséquilibre est structurant pour tout le projet (choix des métriques, stratégies de rééquilibrage, optimisation du seuil).

---

## 3. Préparation des Données (Phase 2)

### 3.1 Valeurs manquantes

Aucune valeur manquante dans le CSV final. L'imputation a été réalisée dans le script de collecte :
- **Numériques** : imputation par la médiane (robuste aux outliers)
- **Catégorielles** : remplacement par `UNKNOWN`

### 3.2 Doublons et incohérences

- 0 doublon exact détecté.
- Contrôles de cohérence métier effectués (ex : budget vs année, diamètre positif).

### 3.3 Outliers

Les outliers détectés par IQR et Z-score ont été **conservés** car ils reflètent la réalité physique (quelques astéroïdes très grands ou très rapides sont porteurs de signal pour la classification PHA).

### 3.4 Suppression de variables redondantes

Quatre variables ont été supprimées pour multicolinéarité (|r| > 0.95) :
- `estimated_diameter_min_km` → remplacée par `diameter_mean_km` (ingéniée)
- `aphelion_distance` et `orbital_period` → redondantes avec `semi_major_axis` (3ème loi de Kepler)
- `eccentricity` → remplacée par `perihelion_to_aphelion_ratio` (ingéniée)

### 3.5 Feature Engineering (6 variables dérivées)

| Variable | Formule | Justification |
|---|---|---|
| `diameter_mean_km` | (min + max) / 2 | Taille moyenne plus stable |
| `diameter_uncertainty` | max - min | Fiabilité de la mesure |
| `perihelion_to_aphelion_ratio` | perihelion / aphelion | Forme de l'orbite |
| `threat_ratio` | MOID / diamètre | Proximité relative à la taille |
| `velocity_distance_ratio` | vitesse / distance | Indicateur de dangerosité |
| `observation_reliability` | incertitude / arc | Fiabilité de l'orbite |

### 3.6 Pipeline de preprocessing

Un `ColumnTransformer` avec :
- **Numériques** : `SimpleImputer(strategy='median')` + `StandardScaler`
- **Catégorielles** : `SimpleImputer(strategy='constant', fill_value='UNKNOWN')` + `OneHotEncoder`

Sérialisé dans `models/preprocessor.joblib`. Le `fit` est effectué uniquement sur le train pour éviter le data leakage.

### 3.7 Split stratifié

60% train / 20% validation / 20% test avec `stratify=y` et `random_state=42`.

---

## 4. Modélisation, Tuning et Évaluation (Phase 3)

### 4.1 Modèles testés (4 familles)

| Modèle | Forces | Faiblesses |
|---|---|---|
| Régression Logistique | Rapide, interprétable, baseline | Suppose la linéarité |
| Arbre de Décision | Très interprétable | Sur-apprentissage facile |
| XGBoost | Meilleur sur tabulaire, `scale_pos_weight` | Plus lent |
| MLPClassifier | Relations non-linéaires complexes | Boîte noire |

### 4.2 Stratégies de rééquilibrage (3 stratégies)

1. **Baseline** : `class_weight='balanced'` (pondération des classes)
2. **SMOTE** : Suréchantillonnage synthétique de la classe minoritaire
3. **RandomUnderSampler** : Sous-échantillonnage de la classe majoritaire

### 4.3 Tableau comparatif (12 configurations)

Les 12 configurations (4 modèles × 3 stratégies) ont été évaluées par validation croisée stratifiée à 5 splits. Le meilleur couple est **XGBoost + Baseline**, avec un F1 supérieur à 0.98.

### 4.4 Optimisation des hyperparamètres

Le meilleur modèle (XGBoost) a été optimisé via `RandomizedSearchCV` sur les hyperparamètres :
- `n_estimators` : [100, 300, 500]
- `max_depth` : [3, 5, 7]
- `learning_rate` : [0.01, 0.05, 0.1]
- `subsample` : [0.8, 1.0]
- `colsample_bytree` : [0.8, 1.0]

### 4.5 Optimisation du seuil de décision

Le seuil par défaut (0.5) a été remplacé par un seuil optimisé sur la validation en minimisant le coût métier total : `Coût = 1000 × FN + 20 × FP`.

Le seuil optimal retenu est de **0.13**, ce qui favorise le rappel (détecter tous les PHAs) au prix de quelques faux positifs supplémentaires — un compromis parfaitement aligné avec le contexte de défense planétaire.

---

## 5. Interprétation et Recommandations Métier (Phase 3)

### 5.1 Variables les plus importantes

Les features les plus discriminantes pour la classification PHA sont :
- `minimum_orbit_intersection` (MOID) — critère NASA #2
- `diameter_mean_km` — critère NASA #1
- `threat_ratio` — feature ingéniée combinant MOID et taille
- `absolute_magnitude_h` — proxy de la taille

### 5.2 Recommandations

1. **Utilisation opérationnelle** : Le modèle peut être utilisé comme filtre de premier niveau pour prioriser les astéroïdes nécessitant une analyse approfondie.
2. **Ré-entraînement** : Recommandé tous les 6 mois avec les nouvelles données NASA.
3. **Seuil adaptatif** : Le seuil de 0.13 peut être ajusté selon le budget d'observation disponible.

---

## 6. Architecture et Déploiement (Phase 4)

### 6.1 API REST (FastAPI)

5 endpoints implémentés :

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Page d'accueil avec liens vers la documentation |
| `/health` | GET | Vérification de l'état de l'API (200 OK) |
| `/model/info` | GET | Métadonnées du modèle (type, seuil, métriques) |
| `/predict` | POST | Prédiction unitaire (JSON → JSON) |
| `/predict/batch` | POST | Prédiction par lot (CSV → CSV enrichi) |

**Bonnes pratiques** :
- Validation des entrées avec Pydantic (types, plages de valeurs)
- Gestion des erreurs HTTP (400, 500)
- Logging des requêtes pour traçabilité
- Chargement du modèle une seule fois au démarrage (lifespan)

### 6.2 Interface Utilisateur (Streamlit)

3 onglets :
- **Information** : description du modèle et de ses performances
- **Prédiction Unitaire** : formulaire complet avec validation côté UI
- **Prédiction par Lot** : upload CSV et téléchargement des résultats

### 6.3 Conteneurisation Docker

- `Dockerfile` : image basée sur `python:3.11-slim`, couches optimisées
- `docker-compose.yml` : 2 services (API + UI) avec healthcheck
- `.dockerignore` : exclusion de `data/`, `notebooks/`, `.git`, `venv`, `__pycache__`, `.env`
- `requirements-docker.txt` : versions figées de toutes les dépendances

---

## 7. Conclusion, Limites et Perspectives

### 7.1 Conclusion

Ce projet démontre l'application complète de la méthodologie CRISP-DM à un problème réel de défense planétaire. Le modèle XGBoost optimisé atteint d'excellentes performances (F1 > 0.98) tout en respectant les contraintes métier (coût asymétrique FN >> FP). Le déploiement via FastAPI et Streamlit rend le modèle accessible à un non-data-scientist.

### 7.2 Limites

1. **Biais temporel** : Le modèle est entraîné sur un snapshot des données NASA. Les nouvelles découvertes ne sont pas prises en compte automatiquement.
2. **Classe UNKNOWN** : Les astéroïdes avec des champs inconnus peuvent recevoir des prédictions moins fiables.
3. **Features ingéniées** : Les ratios calculés nécessitent que les données d'entrée soient cohérentes.
4. **Pas de MLOps** : Absence de pipeline de ré-entraînement automatique.

### 7.3 Perspectives

- Intégration d'un pipeline MLOps (MLflow, DVC) pour le suivi des expériences
- Ajout de SHAP / LIME pour l'explicabilité locale des prédictions
- Mise en place d'un système d'alerte automatique connecté à l'API NASA en temps réel
- Extension à la prédiction de trajectoires d'impact

---

## 8. Annexes

### A. Arborescence du projet

Voir `README.md` pour l'arborescence complète commentée.

### B. Notebooks de référence

| Notebook | Contenu |
|---|---|
| `01_discovery.ipynb` | Cadrage métier, collecte des données |
| `02_eda.ipynb` | Analyse exploratoire (univariée, bivariée, déséquilibre) |
| `03_preprocessing.ipynb` | Nettoyage, transformations, feature engineering, pipeline |
| `04_modeling.ipynb` | 12 configurations (4 modèles × 3 stratégies) |
| `05_tuning.ipynb` | Optimisation des hyperparamètres (RandomizedSearchCV) |
| `06_evaluation.ipynb` | Évaluation finale, seuil optimal, courbes ROC/PR |

### C. Commandes clés

```bash
# Lancer l'API localement
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Lancer l'interface Streamlit
streamlit run app/ui.py

# Lancer avec Docker
docker-compose up --build

# Test rapide de l'API
curl http://localhost:8000/health
```
