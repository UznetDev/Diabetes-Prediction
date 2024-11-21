import streamlit as st
import pandas as pd
from loader import (accuracy_result, 
                    f1_result, 
                    recall_result, 
                    precision_result, 
                    roc_auc)
from function.function import make_donut


def app():
    st.markdown("### Model performance")
    # Model score
    cols = st.columns(5)
    color = 'blue'

    # Accuracy Score
    cols[0].markdown("**Accuracy Score**\n\nPercentage of correct predictions.")
    cols[0].altair_chart(make_donut(accuracy_result, 
                                    'Accuracy Score:',
                                    input_color=color,
                                    R=140,
                                    innerRadius=40,
                                    cornerRadius=10))

    # F1 Score
    cols[1].markdown("**F1 Score**\n\nBalance of Precision and Recall.")
    cols[1].altair_chart(make_donut(f1_result, 
                                    'F1 Score',
                                    input_color=color,
                                    R=140,
                                    innerRadius=40,
                                    cornerRadius=10))

    # Recall Score
    cols[2].markdown("**Recall Score**\n\nProportion of actual positives identified.")
    cols[2].altair_chart(make_donut(recall_result, 
                                    'Recall Score',
                                    input_color=color,
                                    R=140,
                                    innerRadius=40,
                                    cornerRadius=10))

    # Precision Score
    cols[3].markdown("**Precision Score**\n\nProportion of positive predictions that are correct.")
    cols[3].altair_chart(make_donut(precision_result, 
                                    'Precision Score:',
                                    input_color=color,
                                    R=140,
                                    innerRadius=40,
                                    cornerRadius=10))

    # ROC AUC Score
    cols[4].markdown("**ROC AUC Score**\n\nAbility to distinguish between classes.")
    cols[4].altair_chart(make_donut(roc_auc,
                                    'ROC AUC Score:',
                                    input_color=color,
                                    R=140,
                                    innerRadius=40,
                                    cornerRadius=10))
