import streamlit as st
import pandas as pd


def app():
    st.sidebar.header("Input Parameters")

    # Pregnancies
    pregnancies_value = st.sidebar.number_input(
        'Enter value for `Pregnancies`',
        min_value=0,
        max_value=20,
        value=1
    )

    # Glucose
    glucose_value = st.sidebar.number_input(
        'Enter value for `Glucose`',
        min_value=0,
        max_value=250,
        value=100
    )

    # Insulin
    insulin_value = st.sidebar.number_input(
        'Enter value for `Insulin`',
        min_value=0,
        max_value=1000,
        value=100
    )

    # BMI
    bmi_value = st.sidebar.number_input(
        'Enter value for `BMI`',
        min_value=0.0,
        max_value=100.0,
        value=37.0,
        format="%.1f"
    )

    # Age
    age_value = st.sidebar.number_input(
        'Enter value for `Age`',
        min_value=0,
        max_value=100,
        value=25
    )

    st.sidebar.markdown('---')
    return pd.DataFrame([[pregnancies_value, glucose_value, insulin_value, bmi_value, age_value]], 
                        columns=['Pregnancies', 'Glucose', 'Insulin','BMI','Age'])    