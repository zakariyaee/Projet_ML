# Fiche de Cadrage - Classification des Astéroïdes Potentiellement Dangereux (PHAs)

**Module** : Machine Learning - Projet de Fin de Module  
**Encadrant** : Pr. Y. EL YOUNOUSSI  
**Auteurs** : Bouchennou Ferdaouss · El Allouche Zakariyae · Tafraouti Sanae  
**Année académique** : 2025–2026  
**Version** : 1.0 - Phase 1  

---

## 1. Domaine et Problématique Métier

### 1.1 Domaine

**Planetary Defense / Astronomie** - Surveillance et classification des objets géocroiseurs (Near-Earth Objects, NEOs).

### 1.2 Contexte Opérationnel

La NASA et l'Agence Spatiale Européenne (ESA) assurent conjointement la surveillance de l'espace proche de la Terre dans le cadre des programmes de **Défense Planétaire**. À ce jour, le Centre d'Études des Objets Géocroiseurs (CNEOS) du JPL recense environ **36 000 astéroïdes géocroiseurs** (NEOs), dont seulement **2 368 sont officiellement classifiés "Potentially Hazardous Asteroids" (PHAs)** selon les critères physiques et orbitaux de la NASA (données CNEOS, mai 2026).

La capacité d'observation des télescopes au sol et en orbite est limitée. Il est physiquement impossible d'observer en continu l'ensemble des NEOs connus, sans même considérer les objets non encore découverts. Les équipes d'astronomie ont donc besoin d'un **système automatique de priorisation** capable d'identifier rapidement, parmi les milliers d'objets détectés, ceux qui méritent une attention observationnelle immédiate.

### 1.3 Problématique Métier

> **Comment classifier automatiquement un astéroïde géocroiseur comme "potentiellement dangereux" (PHA) ou "non dangereux" (non-PHA) à partir de ses caractéristiques physiques et orbitales, afin d'optimiser l'allocation des ressources d'observation télescopique ?**

---

## 2. Définition Officielle NASA d'un Astéroïde Potentiellement Dangereux

Selon le Centre d'Études des Objets Géocroiseurs de la NASA (CNEOS), un astéroïde est classifié **PHA** si et seulement si les **deux critères suivants** sont simultanément satisfaits :

| Critère | Seuil NASA | Justification Physique |
|---------|------------|------------------------|
| **Taille minimale** | Diamètre estimé > **140 mètres** | En dessous de ce seuil, un impact causerait des dégâts locaux mais pas une catastrophe à l'échelle régionale ou planétaire |
| **Proximité orbitale** | MOID < **0,05 UA** | La distance minimale entre l'orbite de l'astéroïde et celle de la Terre doit être inférieure à 7 479 894 km (≈ 0,05 × 149 597 871 km) |

Ces deux critères correspondent directement aux variables `diameter_mean_km` et `minimum_orbit_intersection` du dataset.

---

## 3. Objectifs Métiers

| Objectif Métier | Indicateur Métier |
|-----------------|-------------------|
| Détecter la quasi-totalité des PHAs pour déclencher une alerte précoce et permettre une réponse de défense planétaire | Taux de PHAs détectés parmi tous les PHAs réels |
| Limiter les fausses alertes pour ne pas saturer les ressources télescopiques et maîtriser les coûts opérationnels | Nombre de nuits d'observation gaspillées sur des objets inoffensifs |
| Optimiser l'allocation du temps d'observation des télescopes de surveillance | Réduction du temps alloué aux objets non dangereux |

---

## 4. Traduction Métier → Machine Learning

| Objectif Métier | Objectif ML | Métrique |
|-----------------|-------------|----------|
| Détecter la quasi-totalité des PHAs (OM-1) | Maximiser le rappel sur la classe "dangereux" (classe 1) | **Recall ≥ 0,90** |
| Limiter les fausses alertes (OM-2) | Maintenir une précision minimale acceptable | **Precision ≥ 0,40** |
| Équilibrer détections et fausses alertes (OM-3) | Maximiser le compromis précision–rappel global sur données déséquilibrées | **PR-AUC** |

**Type de tâche ML** : Classification binaire supervisée déséquilibrée  
**Variable cible** : `is_potentially_hazardous` (0 = non-PHA, 1 = PHA)  
**Déséquilibre mesuré** : 9,75 % de PHAs (1 951 sur 20 000) - naturel, non artificiel  
**Source de la cible** : Fournie nativement par l'API NASA (`is_potentially_hazardous_asteroid`) - pas d'ingénierie artificielle de la variable cible.

---

## 5. Justification du Seuil de Recall ≥ 0,90

### 5.1 Analyse comparative des niveaux de Recall

Sur la base du recensement CNEOS (**2 368 PHAs réels** dans la base NASA, mai 2026) :

| Niveau de Recall | PHAs non détectés | Évaluation |
|------------------|-------------------|------------|
| 0,70 | ~710 astéroïdes dangereux ignorés |  Inacceptable - risque sécuritaire majeur |
| 0,80 | ~474 astéroïdes dangereux ignorés |  Insuffisant pour un système de défense planétaire |
| **0,90** | **~237 astéroïdes dangereux ignorés** | **Compromis optimal - standard retenu** |
| 0,95 | ~118 astéroïdes dangereux ignorés |  Idéal mais génère trop de faux positifs |
| 0,99 | ~24 astéroïdes dangereux ignorés |  Pratiquement inaccessible sans saturer le système |

### 5.2 Réalisme et atteignabilité du seuil 0,90

Ce seuil est **ambitieux mais réaliste** pour les raisons suivantes :
- Les critères NASA (diamètre > 140 m et MOID < 0,05 UA) sont **objectifs et mesurables** ;
- Les variables `diameter_mean_km` et `minimum_orbit_intersection` sont **fortement corrélées** à la variable cible ;
- Ce niveau de recall est un **standard industriel** en détection d'anomalies critiques (fraude bancaire, diagnostic médical, contrôle de sûreté aérospatiale).

> **Note** : Ce seuil pourra être révisé à l'issue de la Phase 2 (modélisation). Une valeur de Recall ≥ 0,85 restera acceptable si la contrainte de Precision ≥ 0,40 est satisfaite.

---

## 6. Analyse du Coût Métier Asymétrique

### 6.1 Matrice de Confusion et Nature des Erreurs

|  | **Prédit : Non-PHA** | **Prédit : PHA** |
|--|----------------------|------------------|
| **Réel : Non-PHA** | Vrai Négatif (TN) |  Faux Positif (FP) |
| **Réel : PHA** |  **Faux Négatif (FN)** | Vrai Positif (TP) |

### 6.2 Quantification Asymétrique des Coûts

| Type d'Erreur | Conséquence Opérationnelle | Coût Estimé |
|---------------|---------------------------|-------------|
| **Faux Négatif** (PHA non détecté) | Astéroïde dangereux ignoré - risque d'impact non anticipé, absence de mesures de mitigation | **Risque civilisationnel** : potentiellement des millions de victimes et des destructions à l'échelle régionale ou continentale (ex. : un impacteur de 300 m libère ~1 000 mégatonnes d'énergie) |
| **Faux Positif** (non-PHA classifié PHA) | Nuit d'observation télescopique allouée à un objet inoffensif | **≈ 10 000 € à 50 000 € par nuit** (≈ 110 000 MAD à 550 000 MAD) selon le type de télescope et l'observatoire |

**Source des estimations de coût** : Les coûts d'opération des grands observatoires terrestres (ESO, Keck, VLT) sont publiquement documentés entre 30 000 € et 80 000 € par nuit d'observation. Les télescopes spécialisés de surveillance NEO ont des coûts opérationnels plus faibles, estimés entre 10 000 € et 50 000 € par nuit.

### 6.3 Conclusion sur l'Orientation du Modèle

L'asymétrie entre le coût d'un faux négatif (risque civilisationnel inestimable) et le coût d'un faux positif (≈ 50 000 € / 550 000 MAD) est **de plusieurs ordres de grandeur**. Cette asymétrie justifie sans ambiguïté de **privilégier le Recall** comme objectif prioritaire du modèle, au détriment d'une précision maximale.

---

## 7. Choix et Justification des Métriques

### 7.1 Métrique Principale : PR-AUC (Aire sous la courbe Précision–Rappel)

Le dataset présente un déséquilibre structurel naturel mesuré à **9,75 % de PHAs**. Dans un tel contexte :
- L'**accuracy** est une métrique trompeuse : un modèle naïf prédisant toujours "non-PHA" atteindrait 90,25 % d'accuracy sans aucun apprentissage ;
- Le **ROC-AUC** est optimiste sur les données déséquilibrées car il intègre les vrais négatifs (très nombreux) dans son calcul ;
- La **PR-AUC** évalue le compromis précision–rappel sur **tous les seuils de décision possibles**, ce qui est adapté à notre problème avec coût asymétrique et classe minoritaire.

### 7.2 Métrique Secondaire : F1-Score

Le F1-Score est la moyenne harmonique entre la Precision et le Recall. Il constitue un indicateur synthétique du compromis entre les deux, utile pour comparer les modèles lors de la phase de modélisation.

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 7.3 Métriques de Contrôle

| Métrique | Seuil | Rôle |
|----------|-------|------|
| **Recall** | ≥ 0,90 | Objectif principal - détecter la quasi-totalité des PHAs |
| **Precision** | ≥ 0,40 | Limite les fausses alertes à un niveau opérationnel acceptable |

### 7.4 Métriques Explicitement Rejetées

| Métrique | Raison du Rejet |
|----------|-----------------|
|  **Accuracy seule** | Trompeuse sur données déséquilibrées : 90,25 % de non-PHAs → accuracy de 90,25 % sans apprentissage |
|  **ROC-AUC seule** | Trop optimiste sur données déséquilibrées ; intègre les vrais négatifs (très nombreux) et surestime la performance réelle sur la classe minoritaire |

---

## 8. Informations Techniques du Projet

| Attribut | Valeur |
|----------|--------|
| **API** | NASA NeoWs (Near Earth Object Web Service) |
| **URL API** | `https://api.nasa.gov/neo/rest/v1/neo/browse` |
| **Authentification** | Clé API personnelle (gratuite, 1 000 requêtes/heure) |
| **Bibliothèques principales** | `pandas`, `numpy`, `requests`, `python-dotenv` |
| **Dataset** | 20 000 lignes × 27 colonnes (26 features + 1 cible) |
| **Tâche ML** | Classification binaire supervisée déséquilibrée |
| **Variable cible** | `is_potentially_hazardous` (native API NASA) |
| **Déséquilibre mesuré** | 9,75 % de PHAs (1 951 PHAs / 18 049 non-PHAs) - naturel, non artificiel |
| **Script de collecte** | `src/data_collection.py` - version 4.0-json |

---

## 9. Références

- NASA Center for Near Earth Object Studies (CNEOS) : [https://cneos.jpl.nasa.gov/](https://cneos.jpl.nasa.gov/)
- Définition officielle des PHAs (NASA) : [https://cneos.jpl.nasa.gov/about/neo_groups.html](https://cneos.jpl.nasa.gov/about/neo_groups.html)
- NASA NeoWs API Documentation : [https://api.nasa.gov/](https://api.nasa.gov/)
- JPL Small-Body Database : [https://ssd.jpl.nasa.gov/](https://ssd.jpl.nasa.gov/)
- NASA Planetary Defense Coordination Office : [https://www.nasa.gov/planetarydefense](https://www.nasa.gov/planetarydefense)

---

> **Résumé exécutif** : Ce projet vise à construire un modèle de classification binaire supervisée pour détecter automatiquement les astéroïdes potentiellement dangereux (PHAs) à partir des données de l'API NASA NeoWs. Le dataset contient 20 000 astéroïdes avec un déséquilibre naturel mesuré à 9,75 % de PHAs (1 951 PHAs / 18 049 non-PHAs). L'asymétrie des coûts d'erreur - faux négatif = risque civilisationnel versus faux positif ≈ 50 000 € (550 000 MAD) - oriente le choix vers la maximisation du Recall (≥ 0,90) et l'utilisation de la PR-AUC comme métrique principale d'évaluation.