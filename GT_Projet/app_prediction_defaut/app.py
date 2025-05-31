import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
import os
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Défauts de Paiement - Afriland First Bank",
    page_icon="💰",
    layout="wide"
)

# Style CSS personnalisé Afriland First Bank
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #1a1a1a;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #cc0000;
        color: white;
        padding: 0.8rem 1rem;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #990000;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #cc0000;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #cc0000;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: transparent;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background-color: #2a2a2a;
        padding: 1rem;
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2a2a2a;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #2a2a2a;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application avec logo
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

base_leasing = load_data("base_leasing_finale.xlsx")

filename_joblib = 'H:/ISE3_Nathan/GT/gt/Notre_code/Modelisation/logistic_regression_model.joblib'

col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://www.afrilandfirstbank.com/wp-content/uploads/2020/05/logo-afriland.png", width=150)
with col2:
    st.title("Prédiction des Défauts de Paiement")
    st.markdown("---")
    

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une section", ["Prédiction", "Visualisation", "À propos"])

if page == "Prédiction":
    st.header("Formulaire de Prédiction")

    # Fonctions d'encodage et de modélisation
    def encode2(df):
        categorical_columns = ['objet_credit_groupe', 'type', 'segment', 'profil_activite', 
                            'secteur_risque', 'forme_juridique', 'reseau', 'cat_age_entreprise', 'Retard']

        df_dummy = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        numeric_features = ['echa_impaye_avant', 'montant_credit',
                        'total_echeance', 'capital_rembourse', 'capital_restant', 'nbre_ech',
                        'taux_interet', 'age_credit_jours', 'nb_cpt', 'cum_taux_paiement']

        features = numeric_features + [col for col in df_dummy.columns if col.startswith(tuple(categorical_columns))]

        X = df_dummy[features]
        if hasattr(X, 'columns'): 
            X.columns = X.columns.str.replace('[\\[\\]<]', '_', regex=True)

        return X

    def model_final(model, X, threshold=0):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        decision_function_result = model.decision_function(X)
        predict_proba_result = model.predict_proba(X)[:,1]
        return decision_function_result > threshold, predict_proba_result

    # Création des colonnes pour une meilleure organisation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Caractéristiques du Client</div>', unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/business.png", width=100)
        
        # Variables catégorielles du client
    #     'code_client', 'reference_lettrage', 'n_echance', 'date_echeance',
    #    'montant_credit', 'total_echeance', 'capital_rembourse',
    #    'capital_restant', 'date_operation', 'montant', 'nbre_ech',
    #    'taux_interet', 'date_mise_place', 'objet_credit_groupe', 'nb_cpt',
    #    'type', 'segment', 'profil_activite', 'secteur_risque',
    #    'forme_juridique', 'date_creation', 'reseau', 'age_credit_jours',
    #    'cat_age_entreprise', 'Retard', 'echa_impaye_avant',
    #    'cum_taux_paiement', 'statut'
        segment = st.selectbox(
            "Segment de l'entreprise",
            ['GE', 'ME', 'PE', 'INS', 'TPE', 'ASS', 'PAR'],
            help="GE: Grande Entreprise, ME: Moyenne Entreprise, PE: Petite Entreprise, etc."
        )
        
        type_entreprise = st.selectbox(
            "Type d'entreprise",
            ['Société', 'Entreprise Individuelle']
        )
        
        forme_juridique = st.selectbox(
            "Forme juridique",
            ['SA', 'SARL', 'EURL', 'GIE', 'SAS', 'ASCO', 'SP', 'SCI']
        )
        
        profil_activite = st.selectbox(
            "Profil d'activité",
            ['Autres', "Organisme de l\'Etat", 'Entreprise Privee Individuelle',
             'Entreprise incubees', 'Association', 'Profession Liberale']
        )
        
        secteur_risque = st.selectbox(
            "Secteur d'activité",
            ['Commerce', 'Secteur Public', 'Activités Agro-Pastorales',
             'Activités Financières et Assurance', 'Industries', 'Construction',
             'Energie', 'Production Autres Services', 'Transports', 'Télécommunications']
        )
        
        reseau = st.selectbox(
            "Réseau d'affectation",
            ['Yaounde Centre', 'Douala Centre', 'Ouest', 'Zone Nord', 'Douala Nord',
             'Douala Sud', 'Yaounde Sud', 'Est', 'Yaounde Nord']
        )
        
        cat_age_entreprise = st.selectbox(
            "Âge de l'entreprise",
            ['20+ ans', '10-20 ans', '3-10 ans', '<3 ans']
        )
        
    with col2:
        st.markdown('<div class="section-header">Caractéristiques du Contrat</div>', unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/contract.png", width=100)
        
        montant_credit = st.number_input("Montant du crédit", min_value=0, value=10000)
        total_echeance = st.number_input("Montant de l'échéance", min_value=0, value=1000)
        capital_rembourse = st.number_input("Capital remboursé", min_value=0, value=0)
        capital_restant = st.number_input("Capital restant", min_value=0, value=10000)
        nbre_ech = st.number_input("Nombre total d'échéances", min_value=1, value=12)
        taux_interet = st.slider("Taux d'intérêt (%)", 0.0, 30.0, 5.0)
        age_credit_jours = st.number_input("Âge du crédit (jours)", min_value=0, value=0)
        
        objet_credit = st.selectbox(
            "Objet du crédit",
            ['VOITURE', 'AUTRE', 'CAMION', 'MACHINE', 'BUS', 'REMORQUE', 'ENGIN']
        )
    
    st.markdown('<div class="section-header">Comportement de Paiement</div>', unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/000000/payment-history.png", width=100)
    
    col3, col4 = st.columns(2)
    
    with col3:
        #retard_jours = st.number_input("Nombre de jours de retard", min_value=0, value=0)
        cum_taux_paiement = st.slider("Cumul du Taux de paiement (%)", 0.0, 100.0, 100.0)
        ech_impaye_avant = st.number_input("Nombre d'échéances impayées précédentes", min_value=0, value=0)
        n_ech = st.number_input("Position de l'échéance", min_value=1, value=1)
    
    with col4:
        statut_retard = st.selectbox(
            "Statut de retard",
            ['pas_retard', 'retard']
        )
        nb_cpt = st.number_input("Nombre de comptes", min_value=1, value=1)


    # Bouton de prédiction avec style Afriland
    if st.button("Prédire le risque de défaut"):
        st.markdown("---")
        st.markdown("### Résultat de la Prédiction")
        
        # Création du DataFrame de prédiction avec les valeurs saisies
        df_prediction = pd.DataFrame({"echa_impaye_avant":ech_impaye_avant, 'montant_credit':montant_credit,
                'total_echeance':total_echeance, 'capital_rembourse':capital_rembourse, 'capital_restant':capital_restant, 'nbre_ech':nbre_ech,
                'taux_interet':taux_interet, 'age_credit_jours':age_credit_jours, 'nb_cpt':nb_cpt, 'cum_taux_paiement':cum_taux_paiement,
                'objet_credit_groupe':objet_credit, 'type':type_entreprise, 'segment':segment, 'profil_activite':profil_activite, 
                'secteur_risque':secteur_risque, 'forme_juridique':forme_juridique, 'reseau':reseau, 
                'cat_age_entreprise':cat_age_entreprise, 'Retard':statut_retard},index=[n_ech])    
            
        # Encodage des données pour la prédiction
        X_base = encode2(base_leasing) 
        df_prediction_dummies = encode2(df_prediction)
        df_prediction_dummies = df_prediction_dummies.reindex(columns=X_base.columns, fill_value=False)

        # Exécution de la prédiction
        # Pour charger le modèle plus tard:
        @st.cache_resource
        def load_model(path):
            return joblib.load(path)

        loaded_model_joblib = load_model("H:/ISE3_Nathan/GT/gt/Notre_code/Modelisation/logistic_regression_model.joblib")
        #print("Modèle chargé avec joblib.")
        y_pred_loaded = model_final(loaded_model_joblib, df_prediction_dummies)
        
        prediction = y_pred_loaded[0]
        probabilite = y_pred_loaded[1][0]

        if prediction == False:
            st.success("✅ Faible risque de défaut")
        else:
            st.error("⚠️ Risque élevé de défaut")
            
        st.metric("Probabilité de défaut", f"{float(probabilite):.2%}")

elif page == "Visualisation":
    st.header("Visualisation des Données")
    
    # Exemple de graphiques (à adapter avec vos données réelles)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des Scores de Crédit")
        # Création d'un graphique fictif
        scores = np.random.normal(650, 100, 1000)
        fig = px.histogram(scores, title="Distribution des Scores de Crédit")
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Taux de Défaut par Âge")
        # Création d'un graphique fictif
        ages = np.random.randint(18, 80, 1000)
        defauts = np.random.binomial(1, 0.3, 1000)
        df = pd.DataFrame({'Age': ages, 'Défaut': defauts})
        fig = px.box(df, x='Défaut', y='Age', title="Âge vs Défaut")
        st.plotly_chart(fig)

else:  # À propos
    st.header("À propos de l'Application")
    st.markdown("""
    Cette application permet de prédire le risque de défaut de paiement en utilisant des modèles de machine learning.
    
    ### Fonctionnalités
    - Prédiction du risque de défaut
    - Visualisation des données
    - Interface utilisateur intuitive
    
    ### Variables utilisées
    1. **Caractéristiques du Client**
       - Segment de l'entreprise
       - Type d'entreprise
       - Forme juridique
       - Profil d'activité
       - Secteur d'activité
       - Réseau d'affectation
       - Âge de l'entreprise
    
    2. **Caractéristiques du Contrat**
       - Montant du crédit
       - Échéances
       - Capital remboursé/restant
       - Taux d'intérêt
       - Objet du crédit
    
    3. **Comportement de Paiement**
       - Historique des retards
       - Taux de paiement
       - Échéances impayées
       - Position de l'échéance
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Afriland First Bank - Application de Prédiction des Défauts de Paiement") 