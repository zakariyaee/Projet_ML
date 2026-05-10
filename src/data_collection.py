"""
data_collection.py
==================
Collecte de donnees NASA NeoWs pour la classification des asteroides
potentiellement dangereux (PHAs).

Version 3.0 - Robuste avec retry + checkpoint CSV
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CHARGEMENT .ENV
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

API_KEY        = os.getenv("NASA_API_KEY", "DEMO_KEY")
TARGET_ROWS    = 20_000
PAGE_SIZE      = 20
SLEEP_BETWEEN  = 4.0
BASE_URL       = "https://api.nasa.gov/neo/rest/v1/neo/browse"

DATA_DIR       = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_CSV = DATA_DIR / "checkpoint.csv"
DATASET_CSV    = DATA_DIR / "dataset.csv"
SAMPLE_CSV     = DATA_DIR / "sample.csv"
LOG_FILE       = DATA_DIR / "collection.log"

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

TARGET_VARIABLE = "is_potentially_hazardous"


# ─────────────────────────────────────────────
# ETAPE 1 — COLLECTE BRUTE AVEC RETRY
# ─────────────────────────────────────────────

def fetch_page(page: int, session: requests.Session, max_retries: int = 3) -> dict:
    """
    Interroge l'API NeoWs avec retry en cas de timeout.
    """
    params = {
        "page"    : page,
        "size"    : PAGE_SIZE,
        "api_key" : API_KEY,
    }

    for attempt in range(max_retries):
        try:
            response = session.get(BASE_URL, params=params, timeout=60)

            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            log.info(f"Page {page:4d} | status={response.status_code} | rate_remaining={remaining} | tentative={attempt+1}")

            if remaining != "?" and int(remaining) < 10:
                log.warning("Quota proche de la limite — pause de 3600 secondes.")
                time.sleep(3600)

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                log.warning(f"Timeout/Reseau page {page}, tentative {attempt+1}/{max_retries}. Retry dans {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Echec definitif page {page} apres {max_retries} tentatives.")
                raise

    return {}


# ─────────────────────────────────────────────
# ETAPE 2 — EXTRACTION DES FEATURES
# ─────────────────────────────────────────────

def extract_close_approach_features(neo: dict) -> dict:
    approaches = neo.get("close_approach_data", [])

    earth_approaches = [
        a for a in approaches
        if a.get("orbiting_body", "").upper() == "EARTH"
    ]
    if not earth_approaches:
        earth_approaches = approaches

    if not earth_approaches:
        return {
            "relative_velocity_km_per_second" : np.nan,
            "miss_distance_astronomical"      : np.nan,
            "miss_distance_lunar"             : np.nan,
            "orbiting_body"                   : "Unknown",
            "n_approaches"                    : 0,
            "min_miss_distance_au"            : np.nan,
            "max_velocity_km_s"               : np.nan,
        }

    latest = earth_approaches[-1]

    velocities = []
    miss_distances = []

    for a in earth_approaches:
        try:
            velocities.append(float(a["relative_velocity"]["kilometers_per_second"]))
            miss_distances.append(float(a["miss_distance"]["astronomical"]))
        except (KeyError, ValueError):
            continue

    return {
        "relative_velocity_km_per_second" : float(latest["relative_velocity"].get("kilometers_per_second", np.nan)),
        "miss_distance_astronomical"      : float(latest["miss_distance"].get("astronomical", np.nan)),
        "miss_distance_lunar"             : float(latest["miss_distance"].get("lunar", np.nan)),
        "orbiting_body"                   : latest.get("orbiting_body", "Unknown"),
        "n_approaches"                    : len(earth_approaches),
        "min_miss_distance_au"            : min(miss_distances) if miss_distances else np.nan,
        "max_velocity_km_s"               : max(velocities) if velocities else np.nan,
    }


def extract_orbital_features(neo: dict) -> dict:
    orb = neo.get("orbital_data", {})

    def safe_float(key: str) -> float:
        try:
            return float(orb.get(key, np.nan))
        except (ValueError, TypeError):
            return np.nan

    orbit_class = orb.get("orbit_class", {})

    return {
        "semi_major_axis"               : safe_float("semi_major_axis"),
        "eccentricity"                  : safe_float("eccentricity"),
        "inclination"                   : safe_float("inclination"),
        "perihelion_distance"           : safe_float("perihelion_distance"),
        "aphelion_distance"             : safe_float("aphelion_distance"),
        "orbital_period"                : safe_float("orbital_period"),
        "perihelion_argument"           : safe_float("perihelion_argument"),
        "orbit_uncertainty"             : safe_float("orbit_uncertainty"),
        "minimum_orbit_intersection"    : safe_float("minimum_orbit_intersection"),
        "data_arc_in_days"              : safe_float("data_arc_in_days"),
        "orbit_class_type"              : orbit_class.get("orbit_class_type", "Unknown"),
    }


def extract_features(neo: dict) -> dict:
    diam = neo.get("estimated_diameter", {}).get("kilometers", {})
    diam_min = diam.get("estimated_diameter_min")
    diam_max = diam.get("estimated_diameter_max")

    try:
        diam_min = float(diam_min) if diam_min is not None else np.nan
        diam_max = float(diam_max) if diam_max is not None else np.nan
    except (ValueError, TypeError):
        diam_min, diam_max = np.nan, np.nan

    base = {
        "neo_id"                          : neo.get("id"),
        "name"                            : neo.get("name"),
        "absolute_magnitude_h"            : float(neo.get("absolute_magnitude_h", np.nan)),
        "estimated_diameter_min_km"       : diam_min,
        "estimated_diameter_max_km"       : diam_max,
        "is_sentry_object"                : int(neo.get("is_sentry_object", False)),
        "is_potentially_hazardous"        : int(neo.get("is_potentially_hazardous_asteroid", False)),
    }

    approach_features = extract_close_approach_features(neo)
    orbital_features  = extract_orbital_features(neo)

    return {**base, **approach_features, **orbital_features}


# ─────────────────────────────────────────────
# ETAPE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Application du feature engineering...")

    df["diameter_mean_km"] = (df["estimated_diameter_min_km"] + df["estimated_diameter_max_km"]) / 2
    df["diameter_uncertainty"] = df["estimated_diameter_max_km"] - df["estimated_diameter_min_km"]

    df["perihelion_to_aphelion_ratio"] = np.where(
        df["aphelion_distance"] > 0,
        df["perihelion_distance"] / df["aphelion_distance"],
        np.nan
    )

    df["threat_ratio"] = np.where(
        df["diameter_mean_km"] > 0,
        df["minimum_orbit_intersection"] / df["diameter_mean_km"],
        np.nan
    )

    df["velocity_distance_ratio"] = np.where(
        df["miss_distance_astronomical"] > 0,
        df["relative_velocity_km_per_second"] / df["miss_distance_astronomical"],
        np.nan
    )

    df["observation_reliability"] = np.where(
        df["data_arc_in_days"] > 0,
        df["orbit_uncertainty"] / df["data_arc_in_days"],
        np.nan
    )

    cols_to_drop = [
        "estimated_diameter_max_km",
        "miss_distance_lunar",
        "neo_id",
        "name",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    log.info(f"Feature engineering termine. Shape : {df.shape}")
    return df


# ─────────────────────────────────────────────
# ETAPE 4 — NETTOYAGE
# ─────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Nettoyage du dataset...")

    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Doublons supprimes : {before - len(df)}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_VARIABLE]

    for col in num_cols:
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(median_val)
            log.info(f"  {col} : {n_missing} NaN imputes par mediane ({median_val:.4f})")

    for col in ["orbiting_body", "orbit_class_type"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper().fillna("UNKNOWN")

    target_counts = df[TARGET_VARIABLE].value_counts()
    pct_positive = target_counts.get(1, 0) / len(df) * 100
    log.info(f"Distribution cible : {target_counts.to_dict()}")
    log.info(f"Classe minoritaire (PHAs) : {pct_positive:.1f}%")

    if not (5 <= pct_positive <= 25):
        log.warning(f"ATTENTION Desequilibre hors fourchette 5-25% : {pct_positive:.1f}%")

    log.info(f"Dataset final : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df


def verify_constraints(df: pd.DataFrame) -> bool:
    n_features = df.shape[1] - 1
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


# ─────────────────────────────────────────────
# MAIN — AVEC CHECKPOINT CSV ET REPRISE
# ─────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("DEMARRAGE DE LA COLLECTE — NASA NeoWs v3.0")
    log.info(f"Cible : {TARGET_ROWS} asteroides")
    log.info(f"Cle API : {'personnelle' if API_KEY != 'DEMO_KEY' else 'DEMO_KEY (limitee)'}")
    log.info("=" * 60)

    if API_KEY == "DEMO_KEY":
        log.warning("Vous utilisez la DEMO_KEY (50 req/jour). Inscrivez-vous sur https://api.nasa.gov/")

    # ── REPRISE DEPUIS CHECKPOINT ─────────────────────────────
    records = []
    start_page = 0

    if CHECKPOINT_CSV.exists():
        try:
            df_ckpt = pd.read_csv(CHECKPOINT_CSV)
            records = df_ckpt.to_dict('records')
            start_page = len(records) // PAGE_SIZE
            log.info(f"REPRISE depuis checkpoint : {len(records)} objets deja traites, page {start_page}")
        except Exception as e:
            log.warning(f"Checkpoint corrompu, recommencement depuis zero : {e}")
            records = []
            start_page = 0

    # ── COLLECTE + EXTRACTION EN DIRECT ───────────────────────
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    for page in range(start_page, 9999):
        if len(records) >= TARGET_ROWS:
            log.info(f"Objectif atteint ! {len(records)} asteroides collectes.")
            break

        try:
            data = fetch_page(page, session)
            neo_objects = data.get("near_earth_objects", [])

            if not neo_objects:
                log.warning(f"Page {page} vide — fin de la pagination.")
                break

            for neo in neo_objects:
                try:
                    records.append(extract_features(neo))
                except Exception as e:
                    log.warning(f"Erreur extraction {neo.get('id', '?')} : {e}")
                    continue

            log.info(f"Page {page}: {len(records)}/{TARGET_ROWS} objets collectes")

            # ── CHECKPOINT CSV TOUTES LES 50 PAGES (~1000 objets) ──
            if page % 50 == 0 and records:
                pd.DataFrame(records).to_csv(CHECKPOINT_CSV, index=False)
                log.info(f"Checkpoint CSV sauvegarde : {len(records)} objets")

        except Exception as e:
            log.error(f"Erreur page {page}: {e}")
            # Sauvegarde d'urgence avant d'arreter
            if records:
                pd.DataFrame(records).to_csv(CHECKPOINT_CSV, index=False)
                log.info(f"Sauvegarde d'urgence : {len(records)} objets")
            time.sleep(30)
            continue

        time.sleep(SLEEP_BETWEEN)

    # ── SAUVEGARDE FINALE CHECKPOINT ──────────────────────────
    if records:
        pd.DataFrame(records).to_csv(CHECKPOINT_CSV, index=False)
        log.info(f"Checkpoint final : {len(records)} objets")

    # ── FEATURE ENGINEERING ───────────────────────────────────
    log.info("Extraction et feature engineering...")
    df = pd.DataFrame(records)
    log.info(f"DataFrame brut : {df.shape}")

    df = apply_feature_engineering(df)
    df = clean_dataset(df)

    # ── VERIFICATION ──────────────────────────────────────────
    constraints_ok = verify_constraints(df)
    if not constraints_ok:
        log.error("Certaines contraintes ne sont pas respectees !")

    # ── SAUVEGARDE FINALE ─────────────────────────────────────
    df.to_csv(DATASET_CSV, index=False)
    df.head(100).to_csv(SAMPLE_CSV, index=False)

    # Supprimer le checkpoint (plus necessaire)
    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()
        log.info("Checkpoint supprime (collecte terminee).")

    log.info(f"Dataset sauvegarde      : {DATASET_CSV}")
    log.info(f"Echantillon (100 lignes): {SAMPLE_CSV}")

    # ── RESUME FINAL ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESUME FINAL")
    log.info(f"  Lignes totales     : {len(df)}")
    log.info(f"  Features           : {df.shape[1] - 1}")
    log.info(f"  PHAs (classe=1)    : {df[TARGET_VARIABLE].sum()}")
    log.info(f"  Non-PHAs (classe=0): {(df[TARGET_VARIABLE] == 0).sum()}")
    log.info(f"  Colonnes           : {list(df.columns)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()