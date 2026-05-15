# Fiche de Cadrage — Classification des Astéroïdes Potentiellement Dangereux (PHAs)

## 1. Objectifs Métiers Quantifiés

### 1.1 Domaine
**Planetary Defense / Astronomie** — Surveillance des objets proches de la Terre (NEOs).

### 1.2 Problématique Métier
La NASA et l'Agence Spatiale Européenne (ESA) surveillent environ **30 000 astéroïdes proches de la Terre**. Seulement **~2 200 sont classés "potentiellement dangereux"** (PHAs). Les astronomes ne peuvent pas observer les 30 000 objets en permanence et ont besoin d'un système automatique pour prioriser les télescopes.

### 1.3 Objectifs Métiers Quantifiés

| Objectif Métier | Quantification |
|-----------------|----------------|
| Détecter 90% des PHAs pour alerte précoce | **Recall ≥ 0.90** sur la classe "dangereux" |
| Limiter les fausses alertes | **Precision ≥ 0.40** (coût d'une nuit télescope ~10 000€ — 50 000€) |
| Optimiser l'allocation des ressources télescopes | Réduire le temps d'observation alloué aux objets inoffensifs |

---

### 1.4 Justification du Seuil de Recall ≥ 0.90

> **⚠️ Clarification importante** : Le seuil de performance (Recall ≥ 0.90) est différent du seuil de classification (généralement 0.5).

#### Distinction des deux seuils

| Élément | Seuil de classification | Seuil de performance |
|---------|------------------------|----------------------|
| **Quoi ?** | Probabilité à partir de laquelle on prédit "dangereux" | Objectif minimal de rappel à atteindre |
| **Valeur** | Généralement **0.5** (par défaut) | **0.90** (choix métier) |
| **Rôle** | Décision binaire : PHA ou non | Évaluation du modèle |

#### Pourquoi 0.90 et pas 0.80 ou 0.95 ?

Le choix de **Recall ≥ 0.90** repose sur une analyse du coût des erreurs et des standards industriels en détection d'anomalies critiques :

| Niveau de Recall | PHA manqués (sur ~2 200 réels) | Évaluation |
|------------------|-------------------------------|------------|
| **0.70** | ~660 astéroïdes dangereux non détectés | ❌ Trop risqué pour la sécurité planétaire |
| **0.80** | ~440 astéroïdes dangereux non détectés | ❌ Encore insuffisant |
| **0.90** | ~220 astéroïdes dangereux non détectés | ✅ **Compromis optimal** |
| **0.95** | ~110 astéroïdes dangereux non détectés | ⚠️ Idéal mais trop coûteux en faux positifs |
| **0.99** | ~22 astéroïdes dangereux non détectés | ❌ Irréaliste — modèle inutilisable |

#### Comment atteindre Recall ≥ 0.90 ?

Ce n'est **pas** en fixant le seuil de classification à 0.9, mais en :

1. **Optimisant le modèle** pour bien apprendre les patterns des PHAs (features physiques et orbitales)
2. **Rééchantillonnant** (SMOTE, undersampling, class weights) pour compenser le déséquilibre
3. **Ajustant le seuil de décision** si nécessaire (descente à 0.3 ou 0.2 pour capter plus de PHAs)

#### Exemple concret d'ajustement de seuil

| Seuil de décision | Recall | Precision | Interprétation |
|-------------------|--------|-----------|----------------|
| 0.5 (défaut) | ~0.85 | ~0.50 | Standard, mais insuffisant |
| **0.3** | **~0.92** | ~0.35 | ✅ **Plus de PHA détectés** — acceptable |
| 0.1 | ~0.98 | ~0.15 | ❌ Trop de faux positifs — modèle inutilisable |

#### Réalisme du seuil 0.90

Ce seuil est **ambitieux mais atteignable** car :
- Les critères NASA sont **physiques et objectifs** (diamètre > 140m ET distance < 0.05 UA)
- Les features `diameter_mean_km` et `minimum_orbit_intersection` sont **fortement corrélées** avec la cible
- C'est un standard industriel en détection d'anomalies critiques (fraude bancaire, diagnostic médical, sécurité aérospatiale)

> **Note** : Ce seuil pourra être ajusté à la baisse (≥ 0.85) ou à la hausse (≥ 0.95) selon les résultats réels obtenus lors de la phase de modélisation, tout en restant cohérent avec la contrainte métier de détecter la quasi-totalité des PHAs.

---


## 2. Traduction Métier → Machine Learning

| Objectif Métier | Objectif ML | Métrique Principale |
|-------------------|-------------|---------------------|
| Détecter 90% des PHAs pour alerte précoce | Maximiser le rappel (recall) sur la classe "dangereux" | **Recall ≥ 0.90** |
| Limiter les fausses alertes (coût télescope ~50k€/nuit) | Maintenir une précision correcte | **Precision ≥ 0.40** |
| Équilibre global détections/alertes | Maximiser le compromis precision-recall | **PR-AUC** |

---

## 3. Analyse du Coût Asymétrique

### 3.1 Matrice de Confusion et Coûts

| | Prédit Non-PHA | Prédit PHA |
|---|---|---|
| **Réel Non-PHA** | ✅ Vrai Négatif | ❌ Faux Positif |
| **Réel PHA** | ❌ Faux Négatif | ✅ Vrai Positif |

### 3.2 Coût des Erreurs

| Type d'Erreur | Conséquence | Coût Estimé |
|---------------|-------------|-------------|
| **Faux Négatif** (PHA non détecté) | Astéroïde dangereux ignoré, risque d'impact non anticipé | **Risque civilisationnel** (potentiellement des millions de vies) |
| **Faux Positif** (non-PHA classé PHA) | Nuit d'observation télescope gaspillée sur un objet inoffensif | **~10 000€ — 50 000€** par nuit |

### 3.3 Orientation du Modèle
**→ On privilégie le RECALL** car un faux négatif est catastrophique (risque civilisationnel) alors qu'un faux positif est un coût financier maîtrisé.

---

## 4. Choix des Métriques Justifié

### 4.1 Métrique Principale : PR-AUC
- **Pourquoi PR-AUC ?** Le dataset est déséquilibré (~8.5% de PHAs). L'accuracy et le ROC-AUC sont optimistes sur les données déséquilibrées et ne reflètent pas bien la performance sur la classe minoritaire.
- **PR-AUC** évalue le compromis precision-recall sur tous les seuils possibles, ce qui est adapté à notre problème avec coût asymétrique.

### 4.2 Métrique Secondaire : F2-Score
- **Pourquoi F2-Score ?** Le F2-score privilégie le recall par rapport à la precision (β=2), ce qui correspond à notre objectif métier de détecter 90% des PHAs.

### 4.3 Métriques de Contrôle
- **Recall** : Doit être ≥ 0.90 (objectif métier)
- **Precision** : Doit être ≥ 0.40 (limite les fausses alertes)

### 4.4 Métriques Refusées
- ❌ **Accuracy seule** : Optimiste sur données déséquilibrées (91.5% de non-PHAs → accuracy de 91.5% sans rien apprendre)
- ❌ **ROC-AUC seule** : Optimiste sur données déséquilibrées, ne reflète pas la performance sur la classe minoritaire

---

## 5. Informations du Projet

| Attribut | Valeur |
|----------|--------|
| **Auteur** | [Votre nom] |
| **Date** | 2025-2026 |
| **API** | NASA NeoWs (Near Earth Object Web Service) |
| **URL API** | https://api.nasa.gov/neo/rest/v1/neo/browse |
| **Langage** | Python 3.x |
| **Librairies** | pandas, numpy, requests, python-dotenv |
| **Dataset** | ~20 000 lignes × 27 colonnes (26 features + 1 cible) |
| **Tâche ML** | Classification binaire déséquilibrée |
| **Variable cible** | `is_potentially_hazardous` (native API NASA) |
| **Déséquilibre** | ~7-10% PHAs (naturel, pas artificiel) |

---

> **Résumé** : Ce projet vise à construire un modèle de classification binaire pour détecter les astéroïdes potentiellement dangereux. Le coût asymétrique (faux négatif = risque civilisationnel >> faux positif = ~50k€) oriente le choix vers le recall et la métrique PR-AUC. Les critères NASA officiels (diamètre > 140m et MOID < 0.05 UA) structurent les features clés du modèle.
