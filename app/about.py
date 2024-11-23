import streamlit as st
from data.base import about_diabets, warn

def app():
    st.markdown(about_diabets)
    st.warning(warn)
