# Documentation du Dataset - Astéroïdes Potentiellement Dangereux (PHAs)

**Module** : Machine Learning - Projet de Fin de Module  
**Encadrant** : Pr. Y. EL YOUNOUSSI  
**Année académique** : 2025–2026  

---

## a) Identification

| Attribut | Valeur |
|----------|--------|
| **Nom du dataset** | `dataset.csv` |
| **Auteurs** | Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae |
| **Date de collecte** | 10 mai 2026 |
| **Version** | 1.0 |
| **Script de génération** | `src/data_collection.py` - version 4.0-json |
| **Format** | CSV (séparateur virgule, encodage UTF-8) |

---

## b) Source

| Attribut | Valeur |
|----------|--------|
| **API utilisée** | NASA NeoWs - Near Earth Object Web Service |
| **URL de l'API** | `https://api.nasa.gov/neo/rest/v1/neo/browse` |
| **Documentation officielle** | [https://api.nasa.gov/](https://api.nasa.gov/) |
| **Endpoint interrogé** | `/neo/browse` (pagination complète) |
| **Paramètres de requête** | `page`, `size=20`, `api_key` |
| **Authentification** | Clé API personnelle (inscription gratuite sur api.nasa.gov) - 1 000 requêtes/heure |
| **Format de réponse** | JSON |
| **Données brutes sauvegardées** | `data/raw/asteroids_raw.json` |
| **Date d'accès** | 10 mai 2026 |
| **Licence** | Données publiques NASA - Open Data, aucune restriction d'utilisation |
| **Référence CNEOS** | [https://cneos.jpl.nasa.gov/about/neo_groups.html](https://cneos.jpl.nasa.gov/about/neo_groups.html) |

---

## c) Description

### c.1 Objectif du Dataset

Ce dataset a été constitué pour résoudre un problème de **classification binaire supervisée déséquilibrée** dans le domaine de la Défense Planétaire. Il permet d'entraîner un modèle de Machine Learning capable de prédire si un astéroïde géocroiseur (NEO) est **potentiellement dangereux (PHA)** selon les critères officiels de la NASA, à partir de ses caractéristiques physiques et orbitales.

**Problème métier adressé** : optimiser l'allocation des ressources d'observation télescopique en priorisant automatiquement les objets qui méritent une surveillance renforcée, parmi les dizaines de milliers de NEOs recensés par le CNEOS (Centre for Near Earth Object Studies).

### c.2 Dimensions

| Métrique | Valeur |
|----------|--------|
| **Nombre de lignes (astéroïdes)** | 20 000 |
| **Nombre de colonnes total** | 27 |
| **Features explicatives** | 26 |
| **Variable cible** | 1 (`is_potentially_hazardous`) |
| **Variables numériques** | 24 |
| **Variables catégorielles** | 2 (`orbiting_body`, `orbit_class_type`) |

### c.3 Schéma Détaillé des Variables

#### Variable Cible

| Nom | Type | Description | Valeurs Possibles |
|-----|------|-------------|-------------------|
| `is_potentially_hazardous` | Binaire (int) | L'astéroïde satisfait-il les deux critères NASA de dangerosité potentielle ? Fourni nativement par l'API via le champ `is_potentially_hazardous_asteroid`. | **0** = Non-PHA, **1** = PHA |

**Définition NASA officielle** (source : CNEOS JPL) : Un astéroïde est classifié PHA si et seulement si :
- Son diamètre estimé est **supérieur à 140 mètres** (`diameter_mean_km > 0,14`)
- **ET** sa distance minimale d'intersection orbitale avec la Terre (MOID) est **inférieure à 0,05 UA** (`minimum_orbit_intersection < 0,05`)

---

#### Groupe 1 - Features Physiques (Taille et Brillance)

| Nom | Type | Unité | Description | Plage de Valeurs |
|-----|------|-------|-------------|------------------|
| `absolute_magnitude_h` | Numérique (float) | Magnitude (H) | Magnitude absolue de l'astéroïde. Plus H est faible, plus l'objet est grand et/ou réfléchissant. Les PHAs ont généralement H < 22. | ~10 à ~33 |
| `estimated_diameter_min_km` | Numérique (float) | km | Diamètre minimum estimé à partir de la magnitude absolue et d'un albédo supposé | ~0,001 à ~50 |
| `diameter_mean_km` ⚙️ | Numérique (float) | km | **Feature ingéniée** : diamètre moyen = (min + max) / 2. Critère NASA #1 (seuil : 0,14 km). Feature la plus prédictive avec `minimum_orbit_intersection`. | ~0,001 à ~50 |
| `diameter_uncertainty` ⚙️ | Numérique (float) | km | **Feature ingéniée** : incertitude de mesure = max − min. Grande valeur = mesure peu fiable. | ~0 à ~10 |

> **Note** : `estimated_diameter_max_km` est extraite de l'API mais supprimée après feature engineering (remplacée par `diameter_mean_km` et `diameter_uncertainty` plus stables).

---

#### Groupe 2 - Feature de Surveillance Active

| Nom | Type | Description | Valeurs Possibles |
|-----|------|-------------|-------------------|
| `is_sentry_object` | Binaire (int) | L'astéroïde est-il actuellement suivi par le système **Sentry** de la NASA (système automatisé de détection des risques d'impact futurs) ? Un objet Sentry présente une probabilité d'impact non nulle sur les 100 prochaines années. | 0 = Non, 1 = Oui |

---

#### Groupe 3 - Features d'Approche (Passage le Plus Proche de la Terre)

Ces features sont extraites des enregistrements d'approches proches (`close_approach_data`) de l'API. L'approche la plus récente est utilisée pour les features principales ; les statistiques agrégées portent sur l'ensemble des approches historiques enregistrées.

| Nom | Type | Unité | Description | Plage de Valeurs |
|-----|------|-------|-------------|------------------|
| `relative_velocity_km_per_second` | Numérique (float) | km/s | Vitesse relative par rapport à la Terre lors du passage le plus récent enregistré | ~0 à ~50 |
| `miss_distance_astronomical` | Numérique (float) | UA | Distance de passage la plus proche (approche la plus récente). Critère NASA #2 indirect (proxy du MOID). 1 UA ≈ 149 597 871 km | ~0 à ~0,5 pour les PHAs, jusqu'à ~10 |
| `orbiting_body` | Catégorielle (str) | - | Corps autour duquel l'astéroïde passe lors de l'approche enregistrée. Quasi-exclusivement "EARTH" pour les NEOs surveillés. | "EARTH", "MARS", "VENUS", etc. |
| `n_approaches` | Numérique (int) | Compteur | Nombre total d'approches proches de la Terre enregistrées dans la base de données NASA pour cet objet | 0 à ~150+ |
| `min_miss_distance_au` | Numérique (float) | UA | Distance minimale historique parmi toutes les approches enregistrées. Proxy du MOID. | ~0 à ~5 |
| `max_velocity_km_s` | Numérique (float) | km/s | Vitesse maximale enregistrée parmi toutes les approches historiques | ~0 à ~50 |

> **Note** : `miss_distance_lunar` (distance en rayons lunaires) est extraite de l'API mais supprimée car redondante avec `miss_distance_astronomical`.

---

#### Groupe 4 - Features Orbitales (Paramètres de Trajectoire)

Extraites depuis le champ `orbital_data` de l'API NASA/JPL. Ces paramètres décrivent la trajectoire de l'astéroïde autour du Soleil.

| Nom | Type | Unité | Description | Plage de Valeurs |
|-----|------|-------|-------------|------------------|
| `semi_major_axis` | Numérique (float) | UA | Demi-grand axe de l'orbite elliptique. Détermine la taille de l'orbite et la période orbitale. | ~0,5 à ~5 |
| `eccentricity` | Numérique (float) | - | Excentricité de l'orbite (0 = cercle parfait, proche de 1 = très elliptique). Les PHAs ont souvent une eccentricité élevée les amenant près de la Terre. | 0 à ~0,99 |
| `inclination` | Numérique (float) | degrés | Inclinaison du plan orbital par rapport au plan de l'écliptique (plan de l'orbite terrestre) | 0° à ~180° |
| `perihelion_distance` | Numérique (float) | UA | Distance au point le plus proche du Soleil (périhélie). Les PHAs ont un périhélie < 1,3 UA (définition NEO). | ~0,1 à ~1,3 |
| `aphelion_distance` | Numérique (float) | UA | Distance au point le plus éloigné du Soleil (aphélie) | ~0,5 à ~10 |
| `orbital_period` | Numérique (float) | jours | Période de révolution complète autour du Soleil | ~100 à ~3 000 |
| `perihelion_argument` | Numérique (float) | degrés | Argument du périhélie - angle définissant l'orientation de l'orbite dans son plan | 0° à 360° |
| `orbit_uncertainty` | Numérique (int) | 0–9 | Code d'incertitude orbitale JPL : 0 = orbite très bien déterminée, 9 = très incertaine. Calculé à partir du nombre et de la qualité des observations. | 0 à 9 |
| `minimum_orbit_intersection` | Numérique (float) | UA | **MOID** - Distance minimale entre l'orbite de l'astéroïde et l'orbite terrestre. **Critère NASA #2** (seuil : < 0,05 UA). Feature la plus discriminante avec `diameter_mean_km`. | ~0 à ~5 |
| `data_arc_in_days` | Numérique (float) | jours | Durée totale de la fenêtre d'observation ayant permis de calculer l'orbite. Plus grande = orbite mieux déterminée. | ~1 à ~50 000 |
| `orbit_class_type` | Catégorielle (str) | - | Type orbital de l'astéroïde selon la classification du MPC (Minor Planet Center). Détermine la famille à laquelle appartient l'objet. | "AMO", "APO", "ATE", "IEO" |

**Classes orbitales des NEOs** :

| Code | Nom | Description |
|------|-----|-------------|
| `AMO` | Amor | Orbite entre Mars et Terre ; n'intersectent pas l'orbite terrestre |
| `APO` | Apollo | Orbite croisant l'orbite terrestre, demi-grand axe > 1 UA |
| `ATE` | Aten | Orbite croisant l'orbite terrestre, demi-grand axe < 1 UA |
| `IEO` | Atira | Orbite entièrement à l'intérieur de l'orbite terrestre |

---

#### Groupe 5 - Features Ingéniées (Feature Engineering)

Six features supplémentaires ont été construites à partir des features brutes pour améliorer le pouvoir prédictif du modèle.

| Nom | Formule | Description | Justification |
|-----|---------|-------------|---------------|
| `diameter_mean_km` | `(min + max) / 2` | Taille représentative de l'astéroïde | Plus robuste que min ou max seul face aux incertitudes de mesure |
| `diameter_uncertainty` | `max − min` | Amplitude de l'intervalle d'estimation du diamètre | Une grande incertitude indique une mesure peu fiable (peu d'observations) |
| `perihelion_to_aphelion_ratio` | `perihelion / aphelion` | Indicateur de la forme orbitale | Proche de 0 = orbite très elliptique (passe près du Soleil ET loin) ; proche de 1 = orbite quasi-circulaire |
| `threat_ratio` | `minimum_orbit_intersection / diameter_mean_km` | **Indicateur composite de menace** | Un ratio faible signifie : objet gros ET orbite proche de la Terre → double facteur de dangerosité |
| `velocity_distance_ratio` | `relative_velocity / miss_distance_astronomical` | Énergie cinétique relative par unité de distance | Valeur élevée = objet rapide ET proche = fort potentiel destructeur en cas d'impact |
| `observation_reliability` | `orbit_uncertainty / data_arc_in_days` | Fiabilité relative des données orbitales | Incertitude élevée sur une courte période d'observation = données peu fiables |

---

### c.4 Distribution des Classes

| Classe | Label | Effectif | Proportion |
|--------|-------|----------|------------|
| Non-PHA | 0 | ~18 300 | ~91,5 % |
| PHA | 1 | ~1 700 | ~8,5 % |
| **Total** | - | **20 000** | **100 %** |

**Ratio de déséquilibre** : ~10,8:1 (non-PHAs / PHAs) - conforme à la contrainte imposée (classe minoritaire entre 5 % et 25 %).

**Nature du déséquilibre** : Le déséquilibre est **naturel et physiquement justifié**. La NASA estime qu'environ 2 368 PHAs sont répertoriés parmi les ~36 000 NEOs connus (CNEOS, mai 2026), soit un ratio réel de ~6,6 %. Notre dataset (~8,5 %) reflète cette réalité avec une légère sur-représentation liée à la méthode d'échantillonnage par l'API NeoWs. Ce déséquilibre **n'est pas artificiel** - il n'a pas été forcé par des techniques de sur- ou sous-échantillonnage.

**Graphique de distribution** :

![Distribution des classes](../notebooks/dist_classes.png)

---

## d) Qualité des Données

### d.1 Valeurs Manquantes

Les valeurs manquantes proviennent de deux sources principales :
1. Objets NEOs récemment découverts avec peu d'observations (données orbitales incomplètes) ;
2. Astéroïdes sans approche proche enregistrée dans la base de données NASA.

| Groupe de colonnes | Taux de valeurs manquantes estimé | Stratégie d'imputation |
|--------------------|----------------------------------|------------------------|
| Features physiques (`absolute_magnitude_h`, `estimated_diameter_min_km`) | < 2 % | Imputation par la médiane de la colonne |
| Features d'approche (`relative_velocity_km_per_second`, `miss_distance_astronomical`) | < 5 % | Imputation par la médiane de la colonne |
| Features orbitales (`semi_major_axis`, `eccentricity`, etc.) | < 3 % | Imputation par la médiane de la colonne |
| Features ingéniées (`threat_ratio`, `velocity_distance_ratio`) | < 5 % | Imputation par la médiane (héritée des features sources) |
| Variables catégorielles (`orbiting_body`, `orbit_class_type`) | < 1 % | Remplacement par "UNKNOWN" |
| **Variable cible** (`is_potentially_hazardous`) | **0 %** | Fournie nativement par l'API - aucune imputation |

> **Note méthodologique** : La variable cible n'est jamais imputée. Elle est directement fournie par la NASA via le champ `is_potentially_hazardous_asteroid` de l'API NeoWs, sans aucune inférence ou reconstruction de notre part.

### d.2 Traitements Appliqués (Pipeline `data_collection.py`)

Le script `src/data_collection.py` applique les traitements suivants dans l'ordre :

| Étape | Traitement | Détail |
|-------|------------|--------|
| 1 | **Collecte paginée** | Interrogation de l'endpoint `/neo/browse` avec `size=20`, gestion du rate limiting (pause 4 s entre requêtes), retry automatique (3 tentatives, backoff 10/20/30 s) |
| 2 | **Sauvegarde brute** | Toutes les 50 pages + à la fin : `data/raw/asteroids_raw.json` (reprise possible en cas d'interruption) |
| 3 | **Extraction des features** | Parsing du JSON pour chaque NEO : features physiques, d'approche (dernière + agrégées), orbitales |
| 4 | **Feature engineering** | Création de 6 features dérivées (voir Groupe 5) ; suppression de 4 colonnes redondantes (`estimated_diameter_max_km`, `miss_distance_lunar`, `neo_id`, `name`) |
| 5 | **Suppression des doublons** | `df.drop_duplicates()` - les doublons peuvent survenir en cas de reprise de collecte |
| 6 | **Imputation des NaN** | Médiane par colonne numérique (hors cible) ; "UNKNOWN" pour les catégorielles |
| 7 | **Normalisation catégorielle** | `.str.strip().str.upper()` sur `orbiting_body` et `orbit_class_type` |
| 8 | **Vérification des contraintes** | Taille ≥ 10 000, features ≥ 8, déséquilibre 5–25 %, mix numérique/catégoriel |
| 9 | **Sauvegarde finale** | `data/dataset.csv` (20 000 lignes) + `data/sample.csv` (100 premières lignes) |

## e) Références

- NASA NeoWs API Documentation : [https://api.nasa.gov/](https://api.nasa.gov/)
- NASA Center for Near Earth Object Studies (CNEOS) : [https://cneos.jpl.nasa.gov/](https://cneos.jpl.nasa.gov/)
- Définition officielle des PHAs - NASA : [https://cneos.jpl.nasa.gov/about/neo_groups.html](https://cneos.jpl.nasa.gov/about/neo_groups.html)
- JPL Small-Body Database Browser : [https://ssd.jpl.nasa.gov/](https://ssd.jpl.nasa.gov/)
- NASA Planetary Defense Coordination Office : [https://www.nasa.gov/planetarydefense](https://www.nasa.gov/planetarydefense)
- Statistiques NEOs (CNEOS Discovery Statistics) : [https://cneos.jpl.nasa.gov/stats/](https://cneos.jpl.nasa.gov/stats/)

---

> **Dataset généré automatiquement** par `src/data_collection.py` (version 4.0-json) - NASA NeoWs API - collecte du 10 mai 2026.  
> Auteurs : Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae - ENSA Tétouan, 2025–2026.