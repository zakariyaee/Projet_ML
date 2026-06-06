import streamlit as st
import requests
import pandas as pd
import io
import json

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Détection des PHAs",
    page_icon="☄️",
    layout="wide"
)

st.title("Système de Détection des Astéroïdes Potentiellement Dangereux (PHAs)")

# Tabs
tab1, tab2, tab3 = st.tabs(["ℹ️ Information", "🎯 Prédiction Unitaire", "📂 Prédiction par Lot"])

with tab1:
    st.header("À propos du modèle")
    st.write("""
    Ce système utilise un modèle d'apprentissage automatique (Machine Learning) pour classifier les astéroïdes géocroiseurs.
    Le but est d'identifier si un astéroïde est **Potentiellement Dangereux (PHA)** ou non, afin d'optimiser l'allocation des ressources d'observation des télescopes.
    
    ### Performances du modèle
    """)
    
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            info = response.json()
            st.json(info)
            st.success(f"Modèle actuel : **{info['model_key']}** avec un seuil de **{info['threshold']}**")
        else:
            st.error("Impossible de récupérer les informations du modèle depuis l'API.")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
        
    st.write("""
    **Asymétrie des coûts** : Manquer un astéroïde dangereux a un coût inestimable (risque civilisationnel), 
    tandis qu'une fausse alerte coûte simplement du temps d'observation. Le seuil de prédiction a été ajusté en conséquence pour maximiser la sécurité (Rappel).
    """)

with tab2:
    st.header("Prédiction Unitaire")
    st.write("Veuillez remplir les caractéristiques de l'astéroïde pour obtenir une prédiction.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Caractéristiques Physiques")
            absolute_magnitude_h = st.number_input("Magnitude absolue (H)", value=16.53)
            estimated_diameter_min_km = st.number_input("Diamètre min estimé (km)", value=1.31)
            diameter_mean_km = st.number_input("Diamètre moyen (km)", value=2.12)
            diameter_uncertainty = st.number_input("Incertitude du diamètre (km)", value=1.62)
            
        with col2:
            st.subheader("Approches Proches")
            orbiting_body = st.selectbox("Corps orbité", ["EARTH", "JUPTR", "MARS", "VENUS", "UNKNOWN"])
            is_sentry_object = st.selectbox("Objet Sentry (0/1)", [0, 1])
            n_approaches = st.number_input("Nombre d'approches", value=18, step=1)
            relative_velocity_km_per_second = st.number_input("Vitesse relative (km/s)", value=26.25)
            max_velocity_km_s = st.number_input("Vitesse max (km/s)", value=34.13)
            miss_distance_astronomical = st.number_input("Distance de passage (UA)", value=0.106)
            min_miss_distance_au = st.number_input("Distance de passage min (UA)", value=0.039)
            
        with col3:
            st.subheader("Paramètres Orbitaux")
            orbit_class_type = st.selectbox("Classe d'orbite", ["APO", "ATE", "AMO", "UNKNOWN"])
            semi_major_axis = st.number_input("Demi-grand axe (UA)", value=1.078)
            eccentricity = st.number_input("Excentricité", value=0.827)
            inclination = st.number_input("Inclinaison (degrés)", value=22.8)
            perihelion_distance = st.number_input("Distance au périhélie (UA)", value=0.186)
            aphelion_distance = st.number_input("Distance à l'aphélie (UA)", value=1.96)
            orbital_period = st.number_input("Période orbitale (jours)", value=408.83)
            perihelion_argument = st.number_input("Argument du périhélie (degrés)", value=31.43)
            minimum_orbit_intersection = st.number_input("MOID (UA)", value=0.033)
            
        st.subheader("Features Ingéniérées supplémentaires")
        col4, col5, col6, col7 = st.columns(4)
        orbit_uncertainty = col4.number_input("Incertitude orbitale", value=0.0)
        data_arc_in_days = col5.number_input("Arc de données (jours)", value=27807.0)
        perihelion_to_aphelion_ratio = col6.number_input("Ratio Périhélie/Aphélie", value=0.094)
        threat_ratio = col7.number_input("Ratio de menace", value=0.015)
        
        col8, col9 = st.columns(2)
        velocity_distance_ratio = col8.number_input("Ratio Vitesse/Distance", value=245.43)
        observation_reliability = col9.number_input("Fiabilité de l'observation", value=0.0)

        submit_button = st.form_submit_button(label="🚀 Prédire le Risque")

    if submit_button:
        # Preparer le payload
        payload = {
            "absolute_magnitude_h": absolute_magnitude_h,
            "estimated_diameter_min_km": estimated_diameter_min_km,
            "is_sentry_object": int(is_sentry_object),
            "relative_velocity_km_per_second": relative_velocity_km_per_second,
            "miss_distance_astronomical": miss_distance_astronomical,
            "orbiting_body": orbiting_body,
            "n_approaches": int(n_approaches),
            "min_miss_distance_au": min_miss_distance_au,
            "max_velocity_km_s": max_velocity_km_s,
            "semi_major_axis": semi_major_axis,
            "eccentricity": eccentricity,
            "inclination": inclination,
            "perihelion_distance": perihelion_distance,
            "aphelion_distance": aphelion_distance,
            "orbital_period": orbital_period,
            "perihelion_argument": perihelion_argument,
            "orbit_uncertainty": orbit_uncertainty,
            "minimum_orbit_intersection": minimum_orbit_intersection,
            "data_arc_in_days": data_arc_in_days,
            "orbit_class_type": orbit_class_type,
            "diameter_mean_km": diameter_mean_km,
            "diameter_uncertainty": diameter_uncertainty,
            "perihelion_to_aphelion_ratio": perihelion_to_aphelion_ratio,
            "threat_ratio": threat_ratio,
            "velocity_distance_ratio": velocity_distance_ratio,
            "observation_reliability": observation_reliability
        }
        
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    
                    if "Risque élevé" in result["prediction"]:
                        st.error(f"⚠️ **ALERTE ROUGE** : {result['prediction']}")
                    else:
                        st.success(f"✅ **SÉCURISÉ** : {result['prediction']}")
                        
                    st.write(f"- Probabilité d'être un PHA : **{result['probability']:.2%}** (Seuil : {result['threshold']})")
                    st.write(f"- Confiance du modèle : **{result['confidence'].capitalize()}**")
                    
                else:
                    st.error(f"Erreur API : {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {e}")

with tab3:
    st.header("Prédiction par Lot (Batch)")
    st.write("Téléchargez un fichier CSV contenant les caractéristiques de plusieurs astéroïdes.")
    
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.write("Aperçu des données :")
        df_preview = pd.read_csv(uploaded_file)
        st.dataframe(df_preview.head())
        
        # Reset file pointer for request
        uploaded_file.seek(0)
        
        if st.button("Lancer les prédictions par lot"):
            with st.spinner("Traitement du fichier par l'API..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/predict/batch", files=files)
                    
                    if response.status_code == 200:
                        st.success("Traitement terminé avec succès !")
                        
                        # Afficher le bouton de telechargement
                        st.download_button(
                            label="📥 Télécharger les résultats",
                            data=response.content,
                            file_name=f"predictions_{uploaded_file.name}",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Erreur API : {response.text}")
                except Exception as e:
                    st.error(f"Erreur de connexion à l'API : {e}")
