# %% [markdown]
# # Modélisation prédictive des impayés dans les financements leasing
# 
# Ce notebook contient l'analyse et la modélisation prédictive des impayés dans les financements leasing à Afriland First Bank.

# %%
# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

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

# Affichage des informations sur la base de données
print("Informations sur la base de données :")
print(f"Nombre total d'échéances : {len(base_leasing)}")
print(f"Nombre de clients uniques : {base_leasing['code_client'].nunique()}")
print(f"Nombre de contrats uniques : {base_leasing['reference_lettrage'].nunique()}")

# Distribution de la variable cible
print("\nDistribution de la variable statut :")
print(base_leasing['statut'].value_counts(normalize=True))

# %% [markdown]
# ## 2. Prétraitement des variables catégorielles

# %%
# Variables catégorielles à dumméiser
categorical_columns = ['objet_credit_groupe', 'type', 'segment', 'profil_activite', 
                      'secteur_risque', 'forme_juridique', 'reseau', 'retard']

# Dumméisation des variables catégorielles
base_leasing_dummy = pd.get_dummies(base_leasing, columns=categorical_columns, drop_first=True)

# Affichage des nouvelles colonnes créées par la dumméisation
print("Nouvelles colonnes créées par la dumméisation :")
print([col for col in base_leasing_dummy.columns if col.startswith(tuple(categorical_columns))])

# %% [markdown]
# ## 3. Préparation des features et division des données

# %%
# Variables numériques à utiliser
numeric_features = ['retard_jours', 'taux_paiement', 'ech_impaye_avant', 'montant_credit',
                   'total_echeance', 'capital_rembourse', 'capital_restant', 'nbre_ech',
                   'taux_interet', 'age_credit_jours', 'n_echance', 'age_entreprise', 'nb_cpt']

# Création des features en combinant variables numériques et dummifiées
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

print("Dimensions des ensembles d'entraînement et de test :")
print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")

# %% [markdown]
# ## 4. Définition et évaluation des modèles avec gestion de l'overfitting

# %%
# Création d'un dictionnaire pour stocker les modèles avec leurs paramètres optimisés
models = {
    'Logistic Regression': LogisticRegression(
        C=0.1,  # Paramètre de régularisation (plus petit = plus de régularisation)
        max_iter=1000,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,  # Limite la profondeur de l'arbre
        min_samples_split=10,  # Nombre minimum d'échantillons requis pour diviser un nœud
        min_samples_leaf=5,  # Nombre minimum d'échantillons requis dans un nœud feuille
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',  # Nombre de features à considérer pour chaque split
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,  # Utilise 80% des données pour chaque arbre
        colsample_bytree=0.8,  # Utilise 80% des features pour chaque arbre
        random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# Fonction pour évaluer les modèles avec cross-validation
def evaluate_models_with_cv(models, X, y, cv=5):
    results = {}
    for name, model in models.items():
        print(f"\nÉvaluation du modèle {name} avec cross-validation...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        print(f"Scores de cross-validation (AUC-ROC): {cv_scores}")
        print(f"Moyenne des scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Entraînement sur l'ensemble complet
        model.fit(X, y)
        
        # Feature importance pour les modèles qui le supportent
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            print("\nTop 5 des features les plus importantes :")
            print(feature_importance.head())
        
        results[name] = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    return results

# Évaluation des modèles avec cross-validation
cv_results = evaluate_models_with_cv(models, X_train_scaled, y_train)

# %% [markdown]
# ## 5. Optimisation des hyperparamètres avec GridSearchCV

# %%
# Exemple pour Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10]
}

grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

print("Optimisation des hyperparamètres pour Random Forest...")
grid_search_rf.fit(X_train_scaled, y_train)

print("\nMeilleurs paramètres :")
print(grid_search_rf.best_params_)
print("\nMeilleur score :")
print(grid_search_rf.best_score_)

# %% [markdown]
# ## 6. Feature Selection

# %%
# Feature selection avec Random Forest
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    max_features=20  # Sélectionne les 20 features les plus importantes
)

# Application de la sélection de features
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Récupération des features sélectionnées
selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
print("\nFeatures sélectionnées :")
print(selected_features)

# Entraînement du modèle final avec les features sélectionnées
final_model = RandomForestClassifier(**grid_search_rf.best_params_, random_state=42)
final_model.fit(X_train_selected, y_train)

# Évaluation du modèle final
y_pred = final_model.predict(X_test_selected)
y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]

print("\nRésultats du modèle final :")
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Modèle Final')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show() 