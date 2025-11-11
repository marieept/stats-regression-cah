# Projet Analyse de Données : Statistiques, Régression & Classification

Ce projet explore les méthodes fondamentales d'analyse de données à travers les statistiques descriptives, la régression linéaire et la classification ascendante hiérarchique (CAH). Il combine implémentation manuelle et automatisée pour analyser des données bidimensionnelles puis des données réelles multidimensionnelles.

## 👥 Membres
Marie EPINAT - Louis ROBILLARD - Laure WARLOP

**Contexte :** Projet électif 3e année ISEN Ouest | Module : Analyse de données & Cycle | Juin 2025

## 📂 Organisation du projet
```
stats-regression-cah/
├── projet.py                           # Programme principal (6 parties)
├── Data_PE_2025-CSI3_CIR3.xlsx        # Dataset 29 individus × 9 variables
└── README.md
```

## 🎯 Fonctionnalités

### Statistiques & Régression (Parties I-IV)
* **Statistiques descriptives** : moyenne, médiane, variance, écart-type sur 7 points 2D
* **Régression linéaire** : calcul coefficients b₀, b₁ et R²
* **Tests statistiques** : hypothèses (Student), p-valeurs, intervalles de confiance 95%
* **Résultat** : R²=0.0486 → absence de relation linéaire significative

### Classification Hiérarchique (Partie V)
* **Implémentation manuelle** : matrices de distances, formation progressive des clusters Γ₁ à Γ₆
* **Distances multiples** : euclidienne, Manhattan, Chebyshev, Ward
* **Automatisation** : `scipy.cluster.hierarchy.linkage()` avec méthode Ward
* **Dendrogramme** : visualisation hiérarchique avec seuil de coupure optimal

### Évaluation & Validation (Partie VI)
* **Coefficient de silhouette** : mesure cohésion/séparation des clusters (0.4907 pour k=2)
* **Comparaison** : CAH vs k-means (CAH supérieur : 0.4907 > 0.4150)
* **Visualisations avancées** : ACP (76.6% variance), t-SNE, heatmap distances
* **Analyse statistique** : profils par cluster (moyennes, médianes, variances)

## 🛠️ Prérequis
```bash
# Installation des dépendances
pip install numpy pandas matplotlib scipy scikit-learn seaborn openpyxl

# Bibliothèques utilisées
# - numpy, pandas : manipulation de données
# - matplotlib, seaborn : visualisations
# - scipy : tests statistiques, CAH
# - scikit-learn : k-means, PCA, t-SNE, métriques
```

## 🚀 Utilisation
```bash
# 1. Placer Data_PE_2025-CSI3_CIR3.xlsx dans le répertoire
# 2. Exécuter le script
python projet.py

# 3. Résultats générés :
# - Graphiques de régression et dendrogrammes
# - Matrices de distances (affichage console)
# - Visualisations ACP, t-SNE, heatmap
# - Statistiques par cluster
```

## 📊 Chaîne de traitement

**Phase 1 (7 points 2D)** : Statistiques → Régression → Tests → CAH manuelle  
**Phase 2 (29 individus)** : Normalisation → CAH Ward → Évaluation → Visualisations

---
