import streamlit as st
from data.base import head, st_style, footer


def app():
    st.markdown(st_style, 
            unsafe_allow_html=True)

    st.markdown(footer, 
                unsafe_allow_html=True)


    st.markdown(head, 
        unsafe_allow_html=True
    )