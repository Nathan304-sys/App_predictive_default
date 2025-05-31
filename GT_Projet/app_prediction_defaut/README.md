# Application de Prédiction des Défauts de Paiement

Cette application Streamlit permet de prédire le risque de défaut de paiement en utilisant des modèles de machine learning.

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application :
```bash
streamlit run app.py
```

## Fonctionnalités

L'application comprend trois sections principales :

1. **Prédiction**
   - Formulaire de saisie des caractéristiques du client
   - Prédiction du risque de défaut
   - Affichage des résultats

2. **Visualisation**
   - Graphiques interactifs
   - Distribution des scores de crédit
   - Analyse des taux de défaut

3. **À propos**
   - Documentation de l'application
   - Description des variables utilisées

## Structure des Variables

### Caractéristiques du Client
- Âge
- Revenu mensuel
- Score de crédit

### Caractéristiques du Contrat
- Montant du prêt
- Durée du prêt
- Taux d'intérêt

### Comportement de Paiement
- Nombre de retards de paiement
- Utilisation du crédit
- Durée du crédit
- Ratio dette/revenu

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request. 