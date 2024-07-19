import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import sklearn

# Configuration de la page
st.set_page_config(page_title="Prédiction du Diabète", page_icon="🩺", layout="centered")

# Fonction pour afficher la page d'accueil
def accueil():
    st.image("diabete_banner.jpeg", width=500)  
    st.title('Bienvenue sur l\'application de Prédiction du Diabète')
    st.markdown('**Développé par Ettien Kouassi Yann Guy Axel**')
    st.markdown("""
    <style>
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button('Passer à la Prédiction'):
        st.session_state.page = 'Prédiction'

# Fonction pour afficher la page de prédiction
def prediction():
    st.title('Prédiction du Diabète')
    st.markdown('**Développé par Ettien Kouassi Yann Guy Axel**')
    
    # CHARGEMENT DU MODEL
    try:
        with open("model_rf_end.pkl", "rb") as file:
            model = pkl.load(file)
    except FileNotFoundError:
        st.error("Le fichier du modèle est introuvable. Assurez-vous que 'model_rf_end.pkl' est dans le répertoire.")
        st.stop()

    # DEFINITION DE LA FONCTION D'INFERENCE (PERMET DE FAIRE LA PREDICTION)
    def inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        diabete_predict = model.predict(data.reshape(1, -1))
        return diabete_predict

    # CHAMPS DE SAISIE
    st.header("Veuillez entrer les informations suivantes :")
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Nombre de grossesses :", min_value=0, step=1)
        Glucose = st.number_input("Concentration de glucose dans le plasma sanguin à jeun :", min_value=0, step=1)
        BloodPressure = st.number_input("Pression artérielle diastolique (mm Hg) :", min_value=0, step=1)
        SkinThickness = st.number_input("Épaisseur du pli cutané tricipital (mm) :", min_value=0, step=1)

    with col2:
        Insulin = st.number_input("Taux d'insuline sérique de 2 heures (mu U/ml) :", min_value=0, step=1)
        BMI = st.number_input("Indice de masse corporelle (kg/m²) :", min_value=0.0, step=0.1, format="%.1f")
        DiabetesPedigreeFunction = st.number_input("Fonction pedigree du diabète :", min_value=0.0, step=0.1, format="%.3f")
        Age = st.number_input("Âge (années) :", min_value=0, step=1)

    # CREATION DU BOUTON DE PREDICTION
    if st.button('Prédire'):
        result_pred = inference(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        
        if result_pred[0] == 0:
            st.warning("La personne n'est pas susceptible d'avoir un diabète.")
        elif result_pred[0] == 1:
            st.success("La personne est susceptible d'avoir un diabète.")

# Utilisation de la session pour suivre l'état de la page
if 'page' not in st.session_state:
    st.session_state.page = 'Accueil'

# Affichage de la page en fonction de l'état
if st.session_state.page == 'Accueil':
    accueil()
elif st.session_state.page == 'Prédiction':
    prediction()
