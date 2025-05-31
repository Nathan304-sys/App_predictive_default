# %% [markdown]
# # Modèles Bayésiens pour la Prédiction des Impayés
# 
# Ce notebook implémente deux approches bayésiennes pour la prédiction des impayés :
# 1. Régression Logistique Bayésienne
# 2. Bayesian Model Averaging (BMA)

# %%
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pymc3 as pm
import arviz as az
from sklearn.linear_model import BayesianRidge
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
plt.style.use('seaborn')
sns.set_palette('husl')

# %% [markdown]
# ## 1. Chargement et préparation des données

# %%
# Chargement des données
base_leasing = pd.read_excel("../base_exau_2.xlsx")

# Création de l'identifiant unique pour chaque échéance
base_leasing['id_echeance'] = base_leasing['code_client'].astype(str) + '_' + \
                            base_leasing['reference_lettrage'] + '_' + \
                            base_leasing['n_echance'].astype(str)

# Définition de id_echeance comme index
base_leasing.set_index('id_echeance', inplace=True)

# Variables catégorielles à dumméiser
categorical_columns = ['objet_credit_groupe', 'type', 'segment', 'profil_activite', 
                      'secteur_risque', 'forme_juridique', 'reseau', 'retard']

# Dumméisation des variables catégorielles
base_leasing_dummy = pd.get_dummies(base_leasing, columns=categorical_columns, drop_first=True)

# Variables numériques à utiliser
numeric_features = ['retard_jours', 'taux_paiement', 'ech_impaye_avant', 'montant_credit',
                   'total_echeance', 'capital_rembourse', 'capital_restant', 'nbre_ech',
                   'taux_interet', 'age_credit_jours', 'n_echance', 'age_entreprise', 'nb_cpt']

# Création des features
features = numeric_features + [col for col in base_leasing_dummy.columns if col.startswith(tuple(categorical_columns))]

# Création de la variable cible binaire
y = (base_leasing_dummy['statut'] == 'impaye').astype(int)

# Création de la matrice X
X = base_leasing_dummy[features]

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation des variables numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 2. Régression Logistique Bayésienne avec PyMC3

# %%
# Création du modèle bayésien avec PyMC3
def create_bayesian_logistic_model(X, y):
    n_features = X.shape[1]
    
    with pm.Model() as model:
        # Priors pour les coefficients
        beta = pm.Normal('beta', mu=0, sd=10, shape=n_features)
        
        # Prior pour l'intercept
        alpha = pm.Normal('alpha', mu=0, sd=10)
        
        # Fonction logistique
        p = pm.math.sigmoid(alpha + pm.math.dot(X, beta))
        
        # Likelihood
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
        
    return model

# Entraînement du modèle
print("Entraînement du modèle de régression logistique bayésienne...")
model = create_bayesian_logistic_model(X_train_scaled, y_train)

with model:
    # Échantillonnage MCMC
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Analyse des résultats
print("\nRésumé des paramètres :")
print(az.summary(trace, var_names=['alpha', 'beta']))

# Prédictions
with model:
    pm.set_data({'X': X_test_scaled})
    ppc = pm.sample_posterior_predictive(trace, var_names=['y_obs'])

y_pred_bayes = (ppc['y_obs'].mean(axis=0) > 0.5).astype(int)

# Évaluation du modèle
print("\nRésultats de la régression logistique bayésienne :")
print(f"Accuracy: {(y_pred_bayes == y_test).mean():.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, ppc['y_obs'].mean(axis=0)):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_bayes))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_bayes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Régression Logistique Bayésienne')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 3. Bayesian Model Averaging (BMA)

# %%
# Création de plusieurs modèles pour le BMA
def create_bma_models(X_train, y_train, X_test, y_test):
    # Liste des modèles à utiliser
    models = [
        ('bayesian_ridge', BayesianRidge()),
        ('logistic', pm.glm.GLM()),
        ('random_forest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    # Entraînement des modèles et calcul des poids
    model_predictions = []
    model_weights = []
    
    for name, model in models:
        print(f"\nEntraînement du modèle {name}...")
        
        # Entraînement et prédictions
        if name == 'logistic':
            with pm.Model() as model:
                # Définition du modèle logistique
                beta = pm.Normal('beta', mu=0, sd=10, shape=X_train.shape[1])
                alpha = pm.Normal('alpha', mu=0, sd=10)
                p = pm.math.sigmoid(alpha + pm.math.dot(X_train, beta))
                y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)
                
                # Échantillonnage
                trace = pm.sample(1000, tune=1000)
                
                # Prédictions
                with model:
                    pm.set_data({'X': X_test})
                    ppc = pm.sample_posterior_predictive(trace)
                pred = ppc['y_obs'].mean(axis=0)
        else:
            model.fit(X_train, y_train)
            pred = model.predict_proba(X_test)[:, 1]
        
        # Calcul du score (AUC-ROC)
        score = roc_auc_score(y_test, pred)
        model_predictions.append(pred)
        model_weights.append(score)
    
    # Normalisation des poids
    model_weights = np.array(model_weights)
    model_weights = model_weights / model_weights.sum()
    
    # Calcul des prédictions finales
    final_predictions = np.zeros(len(y_test))
    for pred, weight in zip(model_predictions, model_weights):
        final_predictions += pred * weight
    
    return final_predictions, model_weights

# Application du BMA
print("Application du Bayesian Model Averaging...")
bma_predictions, model_weights = create_bma_models(X_train_scaled, y_train, X_test_scaled, y_test)

# Conversion des prédictions en classes
y_pred_bma = (bma_predictions > 0.5).astype(int)

# Évaluation du modèle BMA
print("\nRésultats du Bayesian Model Averaging :")
print(f"Poids des modèles : {model_weights}")
print(f"Accuracy: {(y_pred_bma == y_test).mean():.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, bma_predictions):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_bma))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_bma)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Bayesian Model Averaging')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 4. Comparaison des modèles

# %%
# Création d'un DataFrame pour comparer les performances
results = pd.DataFrame({
    'Modèle': ['Régression Logistique Bayésienne', 'Bayesian Model Averaging'],
    'Accuracy': [(y_pred_bayes == y_test).mean(), (y_pred_bma == y_test).mean()],
    'AUC-ROC': [roc_auc_score(y_test, ppc['y_obs'].mean(axis=0)), 
                roc_auc_score(y_test, bma_predictions)]
})

print("\nComparaison des performances :")
print(results)

# Visualisation des performances
plt.figure(figsize=(10, 6))
results.set_index('Modèle')[['Accuracy', 'AUC-ROC']].plot(kind='bar')
plt.title('Comparaison des Performances des Modèles Bayésiens')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 