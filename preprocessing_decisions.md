# Tableau de décisions — Preprocessing (Phase 2)

**Projet** : Classification des astéroïdes potentiellement dangereux (PHAs)  
**Module** : Machine Learning — ENSA Tétouan — Pr. Yacine EL YOUNOUSSI — 2025-2026  
**Équipe** : Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae

Ce document recense, pour chaque variable, l'action de preprocessing et sa justification.  
Les statistiques d'imputation et de scaling du pipeline sont **ajustées sur le train uniquement** en Phase 3.

---

## Légende des actions

| Action | Description |
|--------|-------------|
| Aucune | Variable cible ou déjà exploitable |
| Imputation médiane | Phase 1 (`data_collection.py`) |
| UNKNOWN | Modalité manquante catégorielle |
| Conservation | Outliers physiquement plausibles |
| One-Hot | Encodage nominal (pipeline) |
| StandardScaler | Normalisation (pipeline) |
| Pas de transformation | Arbres / variable cible |

---

## Tableau par variable

| Variable | Rôle | Manquantes | Nettoyage / outliers | Encodage / scaling | Justification |
|----------|------|------------|----------------------|-------------------|---------------|
| `is_potentially_hazardous` | Cible | Aucune imputation | — | — | Label NASA natif |
| `absolute_magnitude_h` | Numérique | Médiane (collecte) | Outliers conservés | StandardScaler | Taille/brillance ; outliers = grands NEO |
| `estimated_diameter_min_km` | Numérique | Médiane | Conservés | StandardScaler | Borne basse diamètre |
| `is_sentry_object` | Binaire | Conversion 0/1 | — | StandardScaler | Surveillance impact |
| `relative_velocity_km_per_second` | Numérique | Médiane | Conservés | StandardScaler | Énergie cinétique relative |
| `miss_distance_astronomical` | Numérique | Médiane | Conservés | StandardScaler | Proximité passage |
| `orbiting_body` | Catégorielle | UNKNOWN | — | One-Hot | Nominal, faible cardinalité |
| `n_approaches` | Numérique | Médiane | Conservés | StandardScaler | Historique d'approches |
| `min_miss_distance_au` | Numérique | Médiane | Conservés | StandardScaler | Proxy MOID |
| `max_velocity_km_s` | Numérique | Médiane | Conservés | StandardScaler | Vitesse max historique |
| `semi_major_axis` | Numérique | Médiane | Conservés | StandardScaler | Taille orbitale |
| `eccentricity` | Numérique | Médiane | Conservés | StandardScaler | Ellipticité |
| `inclination` | Numérique | Médiane | Conservés | StandardScaler | Inclinaison orbitale |
| `perihelion_distance` | Numérique | Médiane | Conservés | StandardScaler | Critère NEO |
| `aphelion_distance` | Numérique | Médiane | Conservés | StandardScaler | Aphélie |
| `orbital_period` | Numérique | Médiane | Conservés | StandardScaler | Période de révolution |
| `perihelion_argument` | Numérique | Médiane | Conservés | StandardScaler | Orientation orbite |
| `orbit_uncertainty` | Numérique | Médiane | Conservés | StandardScaler | Qualité orbitale JPL |
| `minimum_orbit_intersection` | Numérique | Médiane | Conservés | StandardScaler | **Critère NASA #2 (MOID)** — feature signal |
| `data_arc_in_days` | Numérique | Médiane | Conservés | StandardScaler | Fenêtre d'observation |
| `orbit_class_type` | Catégorielle | UNKNOWN | — | One-Hot | Famille orbitale MPC |
| `diameter_mean_km` | Ingéniée | Médiane | Conservés | StandardScaler | **Critère NASA #1** — feature signal |
| `diameter_uncertainty` | Ingéniée | Médiane | Conservés | StandardScaler | Fiabilité taille |
| `perihelion_to_aphelion_ratio` | Ingéniée | Médiane | Conservés | StandardScaler | Forme orbitale |
| `threat_ratio` | Ingéniée | Médiane | Conservés | StandardScaler | MOID / taille — feature signal |
| `velocity_distance_ratio` | Ingéniée | Médiane | Conservés | StandardScaler | Rapport vitesse / distance |
| `observation_reliability` | Ingéniée | Médiane | Conservés | StandardScaler | Incertitude / durée obs. |

---

## Split et déséquilibre (Phase 3)

| Élément | Décision |
|---------|----------|
| Proportions | 60 % train / 20 % validation / 20 % test |
| Stratification | `stratify=y` sur `is_potentially_hazardous` |
| Reproductibilité | `random_state=42` |
| Fichiers | `data/processed/train.csv`, `val.csv`, `test.csv` |
| Baseline | Pas de rééchantillonnage + `class_weight='balanced'` |
| Oversampling | SMOTE (train uniquement, via `imblearn.pipeline`) |
| Undersampling | RandomUnderSampler (train uniquement) |

---

## Références notebooks

- EDA et analyse du déséquilibre : `notebooks/02_eda.ipynb`
- Preprocessing, split, stratégies : `notebooks/03_preprocessing.ipynb`
- Pipeline sérialisé : `models/preprocessor.joblib`
