import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib

def main():
    st.title('Wine Quality Prediction Model Using Logistic Regression')
    filename = 'Logistic_model.pkl'
    loaded_model = joblib.load(filename)
        #Caching the model for faster loading
        # @st.cache
        # def predict(fixed_acidity, volatile_acidity,  citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    col1, col2, col3 = st.columns(3)
    with col1:
            fixed_acidity = st.number_input('Fixed Acidity:', min_value=4.0, max_value=16.0, value=8.0)
            residual_sugar = st.number_input('Residual Sugar:', min_value=0.0, max_value=1.0, value=0.25)
            total_sulfur_dioxide = st.number_input('Total Sulfure Dioxide', min_value=5.0, max_value=290.0, value=38.0)
    with col2:
            volatile_acidity = st.number_input('Volatile Acidity:', min_value=0.1, max_value=2.0, value=0.50)
            chlorides = st.number_input('Chlorides:', min_value=0.012, max_value=0.68, value=0.07)
            density = st.number_input('Wine Density', min_value=0.5, max_value=1.5, value=1.0)
    with col3:
            citric_acid = st.number_input('Citric Acid:', min_value=0.0, max_value=1.0, value=0.25)
            free_sulfur_dioxide = st.number_input('Free Sulfure Dioxide', min_value=1.0, max_value=72.0, value=14.0)
            pH = st.number_input('Wine pH', min_value=2.5, max_value=4.5, value=3.3)
    col_1, col_2 = st.columns(2)
    with col_1:
            sulphates = st.number_input('Sulphates Level', min_value=0.3, max_value=2.5, value=0.6)
    with col_2:
            alcohol = st.number_input('Alcohol Level', min_value=6.0, max_value=20.0, value=10.0)

    input_dict = {'fixed_acidity':fixed_acidity, 'volatile_acidity':volatile_acidity, 'citric_acid':citric_acid, 
        'residual_sugar':residual_sugar, 'chlorides':chlorides, 'free_sulfur_dioxide':free_sulfur_dioxide, 
        'total_sulfur_dioxide':total_sulfur_dioxide, 'density':density, 'pH':pH, 'suphates':sulphates, 'alcohol':alcohol}
    input_df = pd.DataFrame(input_dict, index=[0])
        ## predict button
    predict_button = st.button('Predict Quality')

    if predict_button:

            quality = loaded_model.predict(input_df)
            if quality == 1:
                st.success("Predicted Quality: Good")
            else:
                st.error("Predicted Quality: Bad")
            
            precision, recall, f1, acc = st.columns(4)
            st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            color: rgb(30, 103, 119);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size: 20px;
            }
            </style>
            """
            , unsafe_allow_html=True)

            with precision:
                st.metric(label="Precision Score", value="76%")
            with recall:
                st.metric(label="Recall Score", value="76%")
            with f1:
                st.metric(label="F1 Score", value="75%")
            with acc:
                st.metric(label="Accuracy Score", value="75%")

        





if __name__ == '__main__':
    main()