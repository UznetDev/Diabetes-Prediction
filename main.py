import pandas as pd
from function.function import *
import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
from app.header import app
app()

# Inputs
from app.input import app
input_data =  app()

# Prediction
from app.predict import app
app(input_data)

#### Explain
from app.explainer import app
app(input_data)

# Model performance
from app.performance import app
app()

# perm_importance
from app.perm_importance import app
app()

st.warning('This project (model) was created for learning purposes; the model may make mistakes. Please trust only qualified experts.')