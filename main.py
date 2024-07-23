import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from loader import df, col
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
from function import *
import numpy as np



st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

def stream_data():
    st.sidebar.header("Input Parameters")
    

    columns = st.sidebar.multiselect("Select Columns to Display", options=col, default=col)

    st.markdown(
        """
        <style>
        .stSlider label {
            color: #FFAA1D;
            font-size: 80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    input_data = {}
    for column in columns:
        input_data[column] = st.sidebar.slider(f'Select value for ( {column} )',
                                               min_value=0, 
                                               max_value=int(df[column].max()), 
                                               value=int(df[column].mean()))


    model, accuracy = train_model(df)
    text = f"Model Accuracy: {accuracy * 100:.2f}%"

    if len(input_data) > 0:
        prediction, probability = predict_diabetes(model, input_data, col)
        re = f'Diabetes' if prediction[0] == 1 else 'No Diabetes'
        text = f"\nPrediction: {re}\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        text = f"\nProbability: {probability[0][1] * 100:.2f}%\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)



    else:
        text = "\nPlease select values for all features to make a prediction.\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.02)




st.write_stream(stream_data)
