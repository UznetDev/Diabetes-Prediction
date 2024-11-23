from function.function import *
import streamlit as st
from loader import page_icon


st.set_page_config(
    page_title="Diabetes Prediction with AI",
    page_icon=page_icon,
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

# About
from app.about import app
app()
