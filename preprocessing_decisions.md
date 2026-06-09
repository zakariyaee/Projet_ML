# Tableau De Décision - Preprocessing
**Projet** : Classification des astéroïdes potentiellement dangereux (PHAs)  
**Module** : Machine Learning - ENSA Tétouan - Pr. Yacine EL YOUNOUSSI - 2025-2026  
**Équipe** : Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae

Ce document détaille, pour chaque variable du dataset, les décisions prises pendant la préparation des données : **suppression ou conservation**, **imputation**, **encodage**, **transformation** et **justification métier / technique**.

Les transformations finales sont encapsulées dans `models/preprocessor.joblib` sous forme de `ColumnTransformer`. Les statistiques d'imputation et de scaling utilisées par le pipeline sont ajustées sur le **jeu d'entraînement uniquement**, afin d'éviter le data leakage.

---

## Synthèse Du Preprocessing

| Élément | Décision |
|---------|----------|
| Variable cible | `is_potentially_hazardous` |
| Problème ML | Classification binaire supervisée déséquilibrée |
| Dataset initial après collecte | 20 000 lignes × 27 colonnes |
| Dataset final pour modélisation | 22 features + 1 cible |
| Variables supprimées avant modélisation | 4 variables multicolinéaires : `estimated_diameter_min_km`, `eccentricity`, `aphelion_distance`, `orbital_period` |
| Variables catégorielles | `orbiting_body`, `orbit_class_type` |
| Encodage catégoriel | `OneHotEncoder(handle_unknown="ignore")` |
| Variables numériques | 20 variables numériques conservées |
| Imputation numérique | `SimpleImputer(strategy="median")` dans le pipeline |
| Imputation catégorielle | `SimpleImputer(strategy="constant", fill_value="UNKNOWN")` |
| Normalisation | `StandardScaler` sur les variables numériques |
| Outliers | Conservés car physiquement plausibles et porteurs de signal |
| Split | 60 % train / 20 % validation / 20 % test, stratifié sur la cible |

---

## Légende Des Actions

| Action | Signification |
|--------|---------------|
| Conservée | Variable gardée dans le dataset final de modélisation |
| Supprimée | Variable retirée avant l'entraînement |
| Ingéniée | Variable créée par feature engineering |
| Médiane | Imputation numérique par la médiane |
| UNKNOWN | Imputation catégorielle par la modalité `UNKNOWN` |
| One-Hot | Encodage nominal par `OneHotEncoder` |
| StandardScaler | Centrage-réduction des variables numériques |
| Aucune | Pas de transformation appliquée |

---

## Variables Supprimées Pendant La Collecte

Ces variables sont extraites ou présentes dans les données brutes, mais supprimées directement dans `src/data_collection.py` avant la génération du dataset tabulaire final.

| Variable | Rôle Initial | Action Effectuée | Imputation | Encodage / Transformation | Justification |
|----------|--------------|------------------|------------|---------------------------|---------------|
| `neo_id` | Identifiant NASA | Supprimée | Aucune | Aucune | Identifiant technique sans valeur prédictive ; risque d'apprentissage d'un identifiant au lieu d'un comportement physique. |
| `name` | Nom de l'astéroïde | Supprimée | Aucune | Aucune | Identifiant textuel non exploitable pour la classification ; très forte cardinalité. |
| `estimated_diameter_max_km` | Diamètre maximum estimé | Supprimée après feature engineering | Médiane si manquante avant calcul | Utilisée pour créer `diameter_mean_km` et `diameter_uncertainty` | Remplacée par deux variables plus stables : diamètre moyen et incertitude du diamètre. |
| `miss_distance_lunar` | Distance de passage en distances lunaires | Supprimée | Médiane si manquante avant suppression | Aucune | Redondante avec `miss_distance_astronomical`, qui exprime la même information dans une unité standard astronomique. |

---

## Tableau De Décision Par Variable

| Variable | Type / Rôle | Action Effectuée | Imputation | Encodage | Transformation | Justification |
|----------|-------------|------------------|------------|----------|----------------|---------------|
| `is_potentially_hazardous` | Cible binaire | Conservée comme variable cible | Aucune | Aucun | Aucune | Label natif fourni par l'API NASA ; ne doit jamais être imputé ni transformé. |
| `absolute_magnitude_h` | Numérique physique | Conservée | Médiane | Aucun | StandardScaler | Magnitude absolue liée à la taille/brillance ; variable importante pour différencier petits et grands objets. |
| `estimated_diameter_min_km` | Numérique physique | Supprimée | Médiane avant analyse | Aucun | Aucune | Forte redondance avec `diameter_mean_km` ; suppression pour réduire la multicolinéarité. |
| `is_sentry_object` | Binaire | Conservée | Valeur 0/1 générée à la collecte | Aucun | StandardScaler | Indique si l'objet est suivi par le système Sentry ; signal métier utile. |
| `relative_velocity_km_per_second` | Numérique d'approche | Conservée | Médiane | Aucun | StandardScaler | Vitesse relative lors du passage ; utile pour estimer le niveau de danger potentiel. |
| `miss_distance_astronomical` | Numérique d'approche | Conservée | Médiane | Aucun | StandardScaler | Distance de passage en UA ; proxy direct de la proximité avec la Terre. |
| `orbiting_body` | Catégorielle nominale | Conservée | `UNKNOWN` | One-Hot | Aucune | Corps autour duquel l'approche est enregistrée ; faible cardinalité, donc One-Hot adapté. |
| `n_approaches` | Numérique de comptage | Conservée | Médiane | Aucun | StandardScaler | Nombre d'approches enregistrées ; reflète l'historique d'observation. |
| `min_miss_distance_au` | Numérique d'approche | Conservée | Médiane | Aucun | StandardScaler | Distance minimale historique ; variable proche du risque orbital. |
| `max_velocity_km_s` | Numérique d'approche | Conservée | Médiane | Aucun | StandardScaler | Vitesse maximale observée ; complète la vitesse relative courante. |
| `semi_major_axis` | Numérique orbitale | Conservée | Médiane | Aucun | StandardScaler | Paramètre central décrivant la taille de l'orbite. |
| `eccentricity` | Numérique orbitale | Supprimée | Médiane avant analyse | Aucun | Aucune | Très corrélée avec `perihelion_to_aphelion_ratio`, variable ingéniée plus interprétable. |
| `inclination` | Numérique orbitale | Conservée | Médiane | Aucun | StandardScaler | Inclinaison de l'orbite ; peut distinguer des familles orbitales. |
| `perihelion_distance` | Numérique orbitale | Conservée | Médiane | Aucun | StandardScaler | Distance minimale au Soleil ; importante pour caractériser les NEOs. |
| `aphelion_distance` | Numérique orbitale | Supprimée | Médiane avant analyse | Aucun | Aucune | Forte redondance avec `semi_major_axis` et `orbital_period`. |
| `orbital_period` | Numérique orbitale | Supprimée | Médiane avant analyse | Aucun | Aucune | Redondante avec `semi_major_axis` via la dynamique orbitale ; suppression pour limiter la multicolinéarité. |
| `perihelion_argument` | Numérique orbitale | Conservée | Médiane | Aucun | StandardScaler | Orientation de l'orbite ; information complémentaire non redondante. |
| `orbit_uncertainty` | Numérique qualité | Conservée | Médiane | Aucun | StandardScaler | Indice d'incertitude orbitale ; renseigne la fiabilité de la trajectoire estimée. |
| `minimum_orbit_intersection` | Numérique orbitale | Conservée | Médiane | Aucun | StandardScaler | MOID ; critère NASA central pour la classification PHA. |
| `data_arc_in_days` | Numérique qualité | Conservée | Médiane | Aucun | StandardScaler | Durée d'observation ; plus elle est longue, plus l'orbite est fiable. |
| `orbit_class_type` | Catégorielle nominale | Conservée | `UNKNOWN` | One-Hot | Aucune | Classe orbitale MPC (`APO`, `AMO`, `ATE`, etc.) ; faible cardinalité, donc One-Hot adapté. |
| `diameter_mean_km` | Numérique ingéniée | Conservée | Médiane | Aucun | StandardScaler | Diamètre moyen dérivé de min/max ; critère NASA majeur pour les PHAs. |
| `diameter_uncertainty` | Numérique ingéniée | Conservée | Médiane | Aucun | StandardScaler | Mesure l'incertitude sur la taille ; utile pour capter la fiabilité de l'estimation. |
| `perihelion_to_aphelion_ratio` | Numérique ingéniée | Conservée | Médiane | Aucun | StandardScaler | Résume la forme de l'orbite ; remplace avantageusement une partie de l'information portée par `eccentricity`. |
| `threat_ratio` | Numérique ingéniée | Conservée | Médiane | Aucun | StandardScaler | Ratio MOID / diamètre ; combine proximité orbitale et taille, deux facteurs de dangerosité. |
| `velocity_distance_ratio` | Numérique ingéniée | Conservée | Médiane | Aucun | StandardScaler | Ratio vitesse / distance ; capture la combinaison "objet rapide et proche". |
| `observation_reliability` | Numérique ingéniée | Conservée | Médiane | Aucun | StandardScaler | Ratio incertitude / durée d'observation ; mesure synthétique de fiabilité orbitale. |

---

## Outliers Et Incohérences

| Élément Vérifié | Décision | Justification |
|-----------------|----------|---------------|
| Doublons exacts | Suppression dans le pipeline de collecte | Les doublons peuvent apparaître en cas de reprise de collecte API ; ils ne doivent pas biaiser l'entraînement. |
| Valeurs manquantes | Imputation numérique par médiane, catégorielle par `UNKNOWN` | Stratégie robuste aux distributions asymétriques et compatible avec le déploiement. |
| Outliers numériques | Conservation | Les valeurs extrêmes correspondent souvent à de vrais astéroïdes très grands, très rapides ou très proches ; elles portent un signal métier. |
| Valeurs catégorielles inconnues | Conservation sous `UNKNOWN` | Évite de supprimer des lignes et permet au modèle de traiter les cas incomplets. |
| Variables fortement corrélées | Suppression de 4 variables | Réduit la redondance et simplifie le modèle sans perte d'information majeure. |

---

## Variables Finales Utilisées Par Le Modèle

Les variables finales présentes dans `data/processed/train.csv`, `data/processed/val.csv` et `data/processed/test.csv` sont :

```text
absolute_magnitude_h
is_sentry_object
relative_velocity_km_per_second
miss_distance_astronomical
orbiting_body
n_approaches
min_miss_distance_au
max_velocity_km_s
semi_major_axis
inclination
perihelion_distance
perihelion_argument
orbit_uncertainty
minimum_orbit_intersection
data_arc_in_days
orbit_class_type
diameter_mean_km
diameter_uncertainty
perihelion_to_aphelion_ratio
threat_ratio
velocity_distance_ratio
observation_reliability
```

La cible associée est :

```text
is_potentially_hazardous
```

---

## Split Et Gestion Du Déséquilibre

| Élément | Décision |
|---------|----------|
| Proportions | 60 % train / 20 % validation / 20 % test |
| Stratification | `stratify=y` sur `is_potentially_hazardous` |
| Reproductibilité | `random_state=42` |
| Fichiers générés | `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv` |
| Baseline | Distribution naturelle + `class_weight='balanced'` pour les modèles compatibles |
| Oversampling | SMOTE appliqué uniquement sur le train via `imblearn.pipeline.Pipeline` |
| Undersampling | RandomUnderSampler appliqué uniquement sur le train |
| Objectif | Comparer les stratégies en Phase 3 sans fuite de données vers validation/test |

---

## Pipeline Sérialisé

Le pipeline de preprocessing est sauvegardé dans :

```text
models/preprocessor.joblib
```

Structure logique du pipeline :

```text
ColumnTransformer
├── num_pipeline
│   ├── SimpleImputer(strategy="median")
│   └── StandardScaler()
└── cat_pipeline
    ├── SimpleImputer(strategy="constant", fill_value="UNKNOWN")
    └── OneHotEncoder(handle_unknown="ignore")
```

---

## Références

- EDA et analyse du déséquilibre : `notebooks/02_eda.ipynb`
- Nettoyage, transformations et pipeline : `notebooks/03_preprocessing.ipynb`
- Pipeline sérialisé : `models/preprocessor.joblib`
