import streamlit as st
from data.base import about_diabets

def app():
    st.markdown(about_diabets)