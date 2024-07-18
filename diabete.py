import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle as pkl

# DESCRIPTION
st.title('DIABETES PREDICTION')

st.text('DEVELOPPER PAR ETTIEN KOUASSI YANN GUY AXEL')

 #CHARGEMENT DU MODEL
file = open("model_rf_end.pkl","rb")
model = pkl.load(file)
file.close()

 ##4 DEFINITION DE LA FONCTION D'INTERFERENCE(PERMET DE FAIRE LA PREDICTION)

def inference(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    data = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    #df = pd.DataFrame(data,columns=['household_size', 'age_of_respondent', 'location_type', 'cellphone_access', 'gender_of_respondent', 'education_level_Primary', 'DiabetesPedigreeFunction', 'education_level_Vocational', 'employed_Government', 'employed_Private'])
    diabete_predict = model.predict(data.reshape(1,-1))
    return diabete_predict

#CHAMPS DE SAISIE

Pregnancies=st.number_input("Nombre de grossesses :" ,min_value=0 ,step=1)
Glucose = st.number_input("concentration de glucose dans le plasma sanguin à jeun :" ,min_value=0 ,step=1)
BloodPressure=  st.number_input(" pression artérielle diastolique (mm Hg) :" ,min_value=0 ,step=1)
SkinThickness =  st.number_input(" épaisseur du pli cutané tricipital (mm) :" ,min_value=0 ,step=1)
Insulin = st.number_input(" taux d'insuline sérique de 2 heures (mu U/ml):" ,min_value=0 ,step=1)
BMI = st.number_input(" indice de masse corporelle (kg/m²):" ,min_value=0.0 ,step=0.1,format="%.1f")
DiabetesPedigreeFunction= st.number_input(" fonction pedigree du diabète:" ,min_value=0.0 ,step=0.1,format="%.3f")
Age =st.number_input(" âge (années):" ,min_value=0 ,step=1)

 #6CREATION DU BOUTON DE PREDICTION

if st.button('Prédire') :
    result_pred = inference(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        
    if result_pred[0]== 0 :
        st.warning("La personne n'est pas succeptible d'avoir un Diabetes")
    elif result_pred[0] ==1:
        st.success("La personne est succeptible d'avoir un Diabetes")