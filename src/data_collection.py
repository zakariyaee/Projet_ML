"""
data_collection.py
==================
Script de collecte de donnees NASA NeoWs pour la classification supervisee
des asteroides potentiellement dangereux (PHAs).

Ce script est concu pour etre :
    - Reproductible : memes parametres = memes resultats
    - Documente : docstrings et commentaires pour chaque fonction
    - Robuste : gestion des erreurs, retry, reprise automatique

Objectif  : collecter ~20 000 asteroides via l'endpoint /neo/browse,
            sauvegarder les donnees brutes en JSON,
            appliquer le feature engineering,
            et generer un dataset CSV pret pour la Phase 2.

API       : NASA NeoWs — https://api.nasa.gov/neo/rest/v1/neo/browse
Auteur    : [Ferdaouss Bouchennou - Zakariyae El Allouche - Sanae Tafraouti]
Date      : 2025-2026
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# ============================================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================================
# Le fichier .env doit contenir : NASA_API_KEY=votre_cle_api_nasa
# Obtenir une cle gratuite sur : https://api.nasa.gov/
load_dotenv()

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

API_KEY: str = os.getenv("NASA_API_KEY", "DEMO_KEY")
"""Cle API NASA. DEMO_KEY est limitee a 50 req/jour. Cle perso = 1000 req/heure."""

TARGET_ROWS: int = 20_000
"""Nombre total d'asteroides a collecter (contrainte prof : >= 10 000)."""

PAGE_SIZE: int = 20
"""Nombre d'objets par page (max recommande par l'API NASA = 20)."""

SLEEP_BETWEEN: float = 4.0
"""Delai en secondes entre chaque requete pour respecter le rate limiting."""

BASE_URL: str = "https://api.nasa.gov/neo/rest/v1/neo/browse"
"""URL de base de l'endpoint NeoWs de la NASA."""

# ============================================================================
# CHEMINS DE FICHIERS
# ============================================================================

DATA_DIR: Path = Path("data")
"""Repertoire principal des donnees."""

RAW_DIR: Path = DATA_DIR / "raw"
"""Repertoire des donnees brutes (JSON)."""

RAW_JSON: Path = RAW_DIR / "asteroids_raw.json"
"""Fichier de sauvegarde des donnees brutes au format JSON."""

DATASET_CSV: Path = DATA_DIR / "dataset.csv"
"""Dataset final tabulaire au format CSV."""

SAMPLE_CSV: Path = DATA_DIR / "sample.csv"
"""Echantillon de 100 lignes pour verification rapide."""

LOG_FILE: Path = DATA_DIR / "collection.log"
"""Fichier de logs de la collecte."""

# Creation des repertoires s'ils n'existent pas
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION DU LOGGING
# ============================================================================
# Chaque appel API est loggue avec timestamp, endpoint et statut pour faciliter
# le debug en cas de probleme.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Variable cible du probleme de classification supervisee
TARGET_VARIABLE: str = "is_potentially_hazardous"


# ============================================================================
# ETAPE 1 — COLLECTE BRUTE VIA API
# ============================================================================

def fetch_page(page: int, session: requests.Session, max_retries: int = 3) -> dict:
    """
    Interroge l'API NASA NeoWs pour une page donnee avec gestion du retry.

    Cette fonction :
        - Construit la requete HTTP avec pagination
        - Loggue chaque appel (timestamp, endpoint, statut, rate limiting)
        - Respecte le rate limiting (pause si quota presque epuise)
        - Gere les erreurs reseau (timeout, connexion) avec retry automatique
        - Gere les codes HTTP d'erreur via raise_for_status()

    Args:
        page: Numero de page a recuperer (0-indexe).
        session: Session requests reutilisable pour optimiser les connexions.
        max_retries: Nombre maximum de tentatives en cas d'echec (defaut: 3).

    Returns:
        dict: Reponse JSON de l'API contenant la liste des asteroides.

    Raises:
        requests.HTTPError: Si le statut HTTP est une erreur apres max_retries.
        requests.ReadTimeout: Si le timeout est depasse apres max_retries.
    """
    params = {
        "page": page,
        "size": PAGE_SIZE,
        "api_key": API_KEY,
    }

    for attempt in range(max_retries):
        try:
            # Envoi de la requete avec timeout de 60 secondes
            response = session.get(BASE_URL, params=params, timeout=60)

            # Log de l'appel API (timestamp, endpoint, statut)
            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            log.info(
                f"API Call | Page {page:4d} | Status {response.status_code} | "
                f"RateRemaining={remaining} | Tentative {attempt + 1}/{max_retries}"
            )

            # Pause automatique si le quota est presque epuise (rate limiting)
            if remaining != "?" and int(remaining) < 10:
                log.warning("Quota proche de la limite — pause de 3600 secondes.")
                time.sleep(3600)

            # Leve une exception si le statut HTTP est une erreur (4xx, 5xx)
            response.raise_for_status()
            return response.json()

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            # Gestion des erreurs reseau avec retry et backoff exponentiel
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                log.warning(
                    f"Timeout/Reseau page {page}, tentative {attempt + 1}/{max_retries}. "
                    f"Retry dans {wait}s..."
                )
                time.sleep(wait)
            else:
                log.error(f"Echec definitif page {page} apres {max_retries} tentatives.")
                raise

    return {}


def collect_raw_data(target_rows: int) -> List[Dict]:
    """
    Collecte les donnees brutes depuis l'API NeoWs jusqu'a atteindre target_rows.

    Cette fonction :
        - Gere la pagination automatique
        - Sauvegarde regulierement les donnees brutes en JSON (toutes les 50 pages)
        - Permet la reprise en cas d'interruption (relit le JSON existant)
        - Respecte le rate limiting de l'API NASA
        - Gere les erreurs reseau et les codes HTTP

    Args:
        target_rows: Nombre total d'asteroides a collecter.

    Returns:
        List[Dict]: Liste des objets NEO bruts retournes par l'API.
    """
    raw_objects: List[Dict] = []
    start_page: int = 0

    # --- REPRISE AUTOMATIQUE ---
    # Si un fichier JSON brut existe deja, on relit les donnees pour reprendre
    # la collecte sans re-requerter les pages deja traitees.
    if RAW_JSON.exists():
        try:
            log.info(f"Reprise depuis sauvegarde brute : {RAW_JSON}")
            with open(RAW_JSON, "r", encoding="utf-8") as f:
                raw_objects = json.load(f)
            start_page = len(raw_objects) // PAGE_SIZE
            log.info(
                f"{len(raw_objects)} objets bruts deja collectes — "
                f"reprise a la page {start_page}"
            )
        except Exception as e:
            log.warning(f"Fichier brut corrompu, recommencement depuis zero : {e}")
            raw_objects = []
            start_page = 0

    # Calcul du nombre de pages restantes a collecter
    pages_needed = (target_rows - len(raw_objects)) // PAGE_SIZE + 1

    # Session requests reutilisable pour optimiser les connexions TCP
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    for i in range(pages_needed):
        current_page = start_page + i

        # Arret si l'objectif est atteint
        if len(raw_objects) >= target_rows:
            log.info(f"Objectif atteint ! {len(raw_objects)} asteroides collectes.")
            break

        try:
            # Appel API avec gestion de la pagination
            data = fetch_page(current_page, session)
            neo_objects = data.get("near_earth_objects", [])

            if not neo_objects:
                log.warning(f"Page {current_page} vide — fin de la pagination.")
                break

            # Ajout des objets bruts a la liste
            raw_objects.extend(neo_objects)
            log.info(
                f"Page {current_page}: {len(raw_objects)}/{target_rows} "
                f"objets bruts collectes"
            )

            # --- SAUVEGARDE REGULIERE DES DONNEES BRUTES ---
            # sauvegarder regulierement le dataset partiel
            # en cas d'interruption. Format JSON pour eviter de re-requerter.
            if (i + 1) % 50 == 0 and raw_objects:
                with open(RAW_JSON, "w", encoding="utf-8") as f:
                    json.dump(raw_objects, f, indent=2, ensure_ascii=False)
                log.info(
                    f"Sauvegarde brute JSON : {len(raw_objects)} objets | "
                    f"Fichier : {RAW_JSON}"
                )

        except Exception as e:
            log.error(f"Erreur page {current_page}: {e}")
            # Sauvegarde d'urgence avant d'arreter pour ne pas perdre la progression
            if raw_objects:
                with open(RAW_JSON, "w", encoding="utf-8") as f:
                    json.dump(raw_objects, f, indent=2, ensure_ascii=False)
                log.info(
                    f"Sauvegarde d'urgence brute : {len(raw_objects)} objets dans {RAW_JSON}"
                )
            time.sleep(30)
            continue

        # Respect du rate limiting (pause entre les requetes)
        time.sleep(SLEEP_BETWEEN)

    # --- SAUVEGARDE FINALE DES DONNEES BRUTES ---
    # sauvegarder les donnees brutes au format JSON
    # pour eviter de re-requerter a chaque execution.
    if raw_objects:
        with open(RAW_JSON, "w", encoding="utf-8") as f:
            json.dump(raw_objects[:target_rows], f, indent=2, ensure_ascii=False)
        log.info(
            f"Sauvegarde finale brute : {len(raw_objects[:target_rows])} objets "
            f"dans {RAW_JSON}"
        )

    return raw_objects[:target_rows]


# ============================================================================
# ETAPE 2 — EXTRACTION DES FEATURES BRUTES
# ============================================================================

def extract_close_approach_features(neo: Dict) -> Dict:
    """
    Extrait les features liees aux approches proches de la Terre.

    Args:
        neo: Dictionnaire brut d'un objet NEO retourne par l'API.

    Returns:
        Dict: Features d'approche (vitesse, distance, corps orbite, etc.).
    """
    approaches = neo.get("close_approach_data", [])

    # Filtrage des approches vers la Terre uniquement
    earth_approaches = [
        a for a in approaches
        if a.get("orbiting_body", "").upper() == "EARTH"
    ]

    # Fallback si aucune approche Terre n'est disponible
    if not earth_approaches:
        earth_approaches = approaches

    # Si aucune approche du tout, retourner des valeurs manquantes
    if not earth_approaches:
        return {
            "relative_velocity_km_per_second": np.nan,
            "miss_distance_astronomical": np.nan,
            "miss_distance_lunar": np.nan,
            "orbiting_body": "Unknown",
            "n_approaches": 0,
            "min_miss_distance_au": np.nan,
            "max_velocity_km_s": np.nan,
        }

    # Approche la plus recente pour les features principales
    latest = earth_approaches[-1]

    # Calcul des statistiques sur toutes les approches Terre
    velocities = []
    miss_distances = []

    for a in earth_approaches:
        try:
            velocities.append(
                float(a["relative_velocity"]["kilometers_per_second"])
            )
            miss_distances.append(
                float(a["miss_distance"]["astronomical"])
            )
        except (KeyError, ValueError):
            continue

    return {
        "relative_velocity_km_per_second": float(
            latest["relative_velocity"].get("kilometers_per_second", np.nan)
        ),
        "miss_distance_astronomical": float(
            latest["miss_distance"].get("astronomical", np.nan)
        ),
        "miss_distance_lunar": float(
            latest["miss_distance"].get("lunar", np.nan)
        ),
        "orbiting_body": latest.get("orbiting_body", "Unknown"),
        "n_approaches": len(earth_approaches),
        "min_miss_distance_au": min(miss_distances) if miss_distances else np.nan,
        "max_velocity_km_s": max(velocities) if velocities else np.nan,
    }


def extract_orbital_features(neo: Dict) -> Dict:
    """
    Extrait les parametres orbitaux depuis orbital_data.

    Args:
        neo: Dictionnaire brut d'un objet NEO retourne par l'API.

    Returns:
        Dict: Features orbitales (excentricite, periode, inclinaison, etc.).
    """
    orb = neo.get("orbital_data", {})

    def safe_float(key: str) -> float:
        """Conversion securisee en float avec gestion des valeurs manquantes."""
        try:
            return float(orb.get(key, np.nan))
        except (ValueError, TypeError):
            return np.nan

    orbit_class = orb.get("orbit_class", {})

    return {
        "semi_major_axis": safe_float("semi_major_axis"),
        "eccentricity": safe_float("eccentricity"),
        "inclination": safe_float("inclination"),
        "perihelion_distance": safe_float("perihelion_distance"),
        "aphelion_distance": safe_float("aphelion_distance"),
        "orbital_period": safe_float("orbital_period"),
        "perihelion_argument": safe_float("perihelion_argument"),
        "orbit_uncertainty": safe_float("orbit_uncertainty"),
        "minimum_orbit_intersection": safe_float("minimum_orbit_intersection"),
        "data_arc_in_days": safe_float("data_arc_in_days"),
        "orbit_class_type": orbit_class.get("orbit_class_type", "Unknown"),
    }


def extract_features(neo: Dict) -> Dict:
    """
    Extrait toutes les features brutes d'un objet NEO + la variable cible.

    Args:
        neo: Dictionnaire brut d'un objet NEO retourne par l'API NASA.

    Returns:
        Dict: Dictionnaire complet avec toutes les features et la variable cible.
    """
    # Diametre estime en kilometres
    diam = neo.get("estimated_diameter", {}).get("kilometers", {})
    diam_min = diam.get("estimated_diameter_min")
    diam_max = diam.get("estimated_diameter_max")

    # Conversion securisee en float
    try:
        diam_min = float(diam_min) if diam_min is not None else np.nan
        diam_max = float(diam_max) if diam_max is not None else np.nan
    except (ValueError, TypeError):
        diam_min, diam_max = np.nan, np.nan

    # Features de base (physiques et identifiants)
    base = {
        "neo_id": neo.get("id"),
        "name": neo.get("name"),
        "absolute_magnitude_h": float(neo.get("absolute_magnitude_h", np.nan)),
        "estimated_diameter_min_km": diam_min,
        "estimated_diameter_max_km": diam_max,
        "is_sentry_object": int(neo.get("is_sentry_object", False)),
        # Variable cible : classification binaire fournie nativement par l'API NASA
        "is_potentially_hazardous": int(
            neo.get("is_potentially_hazardous_asteroid", False)
        ),
    }

    # Fusion des features d'approche et orbitales
    approach_features = extract_close_approach_features(neo)
    orbital_features = extract_orbital_features(neo)

    return {**base, **approach_features, **orbital_features}


# ============================================================================
# ETAPE 3 — FEATURE ENGINEERING
# ============================================================================

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cree les features ingeniees a partir des features brutes et supprime
    les colonnes redondantes.

    Features creees :
        - diameter_mean_km : taille representative (moyenne min/max)
        - diameter_uncertainty : incertitude de mesure du diametre
        - perihelion_to_aphelion_ratio : forme de l'orbite
        - threat_ratio : indicateur de menace (distance danger / taille)
        - velocity_distance_ratio : energie cinetique par unite de distance
        - observation_reliability : fiabilite relative des observations

    Args:
        df: DataFrame brut avec les features extraites de l'API.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les features ingeniees.
    """
    log.info("Application du feature engineering...")

    # 1. Taille moyenne et incertitude (plus stable que min ou max seul)
    df["diameter_mean_km"] = (
        df["estimated_diameter_min_km"] + df["estimated_diameter_max_km"]
    ) / 2
    df["diameter_uncertainty"] = (
        df["estimated_diameter_max_km"] - df["estimated_diameter_min_km"]
    )

    # 2. Forme orbitale : 0 = tres elliptique, proche de 1 = quasi-circulaire
    df["perihelion_to_aphelion_ratio"] = np.where(
        df["aphelion_distance"] > 0,
        df["perihelion_distance"] / df["aphelion_distance"],
        np.nan
    )

    # 3. Indicateur de menace : MOID (Minimum Orbit Intersection Distance)
    # relatif a la taille. Plus petit = plus menacant.
    df["threat_ratio"] = np.where(
        df["diameter_mean_km"] > 0,
        df["minimum_orbit_intersection"] / df["diameter_mean_km"],
        np.nan
    )

    # 4. Ratio vitesse/distance : energie cinetique relative par unite de distance
    df["velocity_distance_ratio"] = np.where(
        df["miss_distance_astronomical"] > 0,
        df["relative_velocity_km_per_second"] / df["miss_distance_astronomical"],
        np.nan
    )

    # 5. Fiabilite des observations : incertitude orbitale relative a l'anciennete
    df["observation_reliability"] = np.where(
        df["data_arc_in_days"] > 0,
        df["orbit_uncertainty"] / df["data_arc_in_days"],
        np.nan
    )

    # Suppression des features redondantes ou non predictives
    cols_to_drop = [
        "estimated_diameter_max_km",   # remplacee par diameter_mean_km
        "miss_distance_lunar",          # redondant avec astronomical
        "neo_id",                       # identifiant non predictif
        "name",                         # identifiant non predictif
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    log.info(f"Feature engineering termine. Shape : {df.shape}")
    return df


# ============================================================================
# ETAPE 4 — NETTOYAGE DU DATASET
# ============================================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset final :
        - Supprime les doublons eventuels
        - Impute les valeurs manquantes numeriques par la mediane
        - Normalise les colonnes categorielles
        - Verifie le desequilibre de la variable cible

    Args:
        df: DataFrame apres feature engineering.

    Returns:
        pd.DataFrame: Dataset nettoye et pret pour la modelisation.
    """
    log.info("Nettoyage du dataset...")

    # Suppression des doublons
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Doublons supprimes : {before - len(df)}")

    # Imputation des valeurs manquantes numeriques par la mediane
    # La cible est exclue de l'imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_VARIABLE]

    for col in num_cols:
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(median_val)
            log.info(
                f"  {col} : {n_missing} NaN imputes par mediane ({median_val:.4f})"
            )

    # Normalisation des colonnes categorielles
    for col in ["orbiting_body", "orbit_class_type"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper().fillna("UNKNOWN")

    # Verification du desequilibre de la variable cible
    target_counts = df[TARGET_VARIABLE].value_counts()
    pct_positive = target_counts.get(1, 0) / len(df) * 100
    log.info(f"Distribution cible : {target_counts.to_dict()}")
    log.info(f"Classe minoritaire (PHAs) : {pct_positive:.1f}%")

    # Alertes si le desequilibre est hors fourchette requise (5% - 25%)
    if not (5 <= pct_positive <= 25):
        log.warning(
            f"ATTENTION Desequilibre hors fourchette 5-25% : {pct_positive:.1f}%"
        )

    log.info(f"Dataset final : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df


def verify_constraints(df: pd.DataFrame) -> bool:
    """
    Verifie que le dataset respecte les contraintes imposees par le professeur.

    Contraintes :
        - Taille totale >= 10 000 lignes
        - Nombre de features >= 8 (apres feature engineering)
        - Classe minoritaire entre 5% et 25% du total
        - Melange de variables numeriques et categorielles

    Args:
        df: Dataset final nettoye.

    Returns:
        bool: True si toutes les contraintes sont respectees, False sinon.
    """
    n_features = df.shape[1] - 1  # -1 pour la variable cible
    pct_pha = df[TARGET_VARIABLE].sum() / len(df) * 100
    n_num = len(df.select_dtypes(include=[np.number]).columns) - 1
    n_cat = len(df.select_dtypes(include=["object", "category", "bool"]).columns)

    checks = {
        f"Taille >= 10 000        : {len(df)}": len(df) >= 10000,
        f"Features >= 8           : {n_features}": n_features >= 8,
        f"Desequilibre 5-25%     : {pct_pha:.1f}%": 5 <= pct_pha <= 25,
        f"Mix numeriques         : {n_num}": n_num > 0,
        f"Mix categorielles      : {n_cat}": n_cat > 0,
    }

    log.info("=" * 60)
    log.info("VERIFICATION DES CONTRAINTES")
    for check, result in checks.items():
        status = "OK" if result else "KO"
        log.info(f"  [{status}] {check}")
    log.info("=" * 60)

    return all(checks.values())


# ============================================================================
# MAIN — ORCHESTRATION DU PIPELINE COMPLET
# ============================================================================

def main():
    """
    Fonction principale orchestrant le pipeline complet de collecte :
        1. Collecte brute via API (avec sauvegarde JSON)
        2. Extraction des features
        3. Feature engineering
        4. Nettoyage
        5. Verification des contraintes
        6. Sauvegarde finale (CSV + echantillon)
    """
    log.info("=" * 60)
    log.info("DEMARRAGE DE LA COLLECTE — NASA NeoWs v4.0-json")
    log.info(f"Cible : {TARGET_ROWS} asteroides")
    log.info(
        f"Cle API : {'personnelle' if API_KEY != 'DEMO_KEY' else 'DEMO_KEY (limitee)'}")
    log.info("=" * 60)

    if API_KEY == "DEMO_KEY":
        log.warning(
            "Vous utilisez la DEMO_KEY (50 req/jour). "
            "Inscrivez-vous sur https://api.nasa.gov/ pour obtenir "
            "une cle personnelle (1000 req/heure)."
        )

    # --- ETAPE 1 : Collecte des donnees brutes ---
    raw_data = collect_raw_data(TARGET_ROWS)
    log.info(f"Objets bruts collectes : {len(raw_data)}")

    # --- ETAPE 2 : Extraction des features ---
    log.info("Extraction des features brutes...")
    records = []
    for neo in raw_data:
        try:
            records.append(extract_features(neo))
        except Exception as e:
            log.warning(f"Erreur extraction {neo.get('id', '?')} : {e}")
            continue

    df = pd.DataFrame(records)
    log.info(f"DataFrame brut : {df.shape}")

    # --- ETAPE 3 : Feature engineering ---
    df = apply_feature_engineering(df)

    # --- ETAPE 4 : Nettoyage ---
    df = clean_dataset(df)

    # --- ETAPE 5 : Verification des contraintes ---
    constraints_ok = verify_constraints(df)
    if not constraints_ok:
        log.error("Certaines contraintes ne sont pas respectees !")

    # --- ETAPE 6 : Sauvegarde finale ---
    df.to_csv(DATASET_CSV, index=False)
    df.head(100).to_csv(SAMPLE_CSV, index=False)

    log.info(f"Dataset final sauvegarde      : {DATASET_CSV}")
    log.info(f"Echantillon (100 lignes)      : {SAMPLE_CSV}")
    log.info(f"Donnees brutes sauvegardees   : {RAW_JSON}")

    # --- Resume final ---
    log.info("=" * 60)
    log.info("RESUME FINAL")
    log.info(
        f"  Donnees brutes (JSON)      : {RAW_JSON} "
        f"({len(raw_data)} objets)"
    )
    log.info(f"  Dataset final (CSV)        : {DATASET_CSV} ({len(df)} lignes)")
    log.info(f"  Features                   : {df.shape[1] - 1}")
    log.info(f"  PHAs (classe=1)            : {df[TARGET_VARIABLE].sum()}")
    log.info(f"  Non-PHAs (classe=0)        : {(df[TARGET_VARIABLE] == 0).sum()}")
    log.info(f"  Colonnes                   : {list(df.columns)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()