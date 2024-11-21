import time
from loader import model
from data.config import X, y
from data.base import head, st_style, footer, mrk
from function.function import *
import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import plotly.express as px
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)



st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(st_style, 
            unsafe_allow_html=True)


st.markdown(footer, unsafe_allow_html=True)

st.sidebar.header("Input Parameters")



st.markdown(head, 
    unsafe_allow_html=True
)

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

thresholds = 0.32
input_data = pd.DataFrame([[pregnancies_value, glucose_value, insulin_value, bmi_value, age_value]], 
                      columns=['Pregnancies', 'Glucose', 'Insulin','BMI','Age'])
prediction = model.predict_proba(input_data)[:, 1]

cols = st.columns(2)

def stream_data():
    is_diabetes = f'Diabetes' if prediction >= thresholds else 'No Diabetes'
    text = f"Model Accuracy: 78.57%\n\n"
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)
    text = f"\nPrediction: {is_diabetes}\n"
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)
    text = f"\nProbability: {(prediction * 100).round(2)[0]}%\n"
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)
    
    return 80

cols[0].write_stream(stream_data)


is_diabetes = f'<strong>Warning:</strong> Diabetes!' if prediction >= thresholds else 'No Diabetes'
color = f'red' if prediction >= thresholds else 'blue'


cols[1].markdown(mrk.format(is_diabetes, color), unsafe_allow_html=True)
cols[1].write('\n\n\n\n\n')
donut_chart_population = make_donut((prediction * 100).round(2)[0], 
                                    'Diabetes Risk',
                                    input_color=color)

cols[1].altair_chart(donut_chart_population)



#### Explain

sample_transformed = model.named_steps['feature_engineering'].transform(input_data)
explainer = shap.TreeExplainer(model.named_steps['random_forest'])
shap_values_single = explainer.shap_values(sample_transformed)

shap_values_class_1 = shap_values_single[0][:, 1]  


def stream_data():
    text = f"""Your inputs:\n
`Pregnancies`: {pregnancies_value}\n
`Glucose`: {glucose_value}\n
`Insulin`: {insulin_value}\n
`BMI`: {bmi_value}\n
`Age`: {age_value}\n
"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

# Layout with two columns
cols = st.columns(2)

# Column 1: Stream user input
with cols[0]:
    st.markdown("### Input Streaming")
    st.markdown("#### See your inputs in real-time below!")
    for word in stream_data():
        st.write(word)

# SHAP Waterfall Plot
fig, ax = plt.subplots()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_class_1,
        base_values=explainer.expected_value[0],
        data=sample_transformed.iloc[0],
        feature_names=sample_transformed.columns.tolist()
    ), show=False
)
fig.patch.set_facecolor("lightblue")
fig.patch.set_alpha(0.3)
ax.set_facecolor("#023047")
ax.patch.set_alpha(0.5)

# Column 2: SHAP Waterfall Plot
with cols[1]:
    st.markdown("### SHAP Waterfall Plot")
    st.markdown(
        """
        - 🟡 **Base Value**: Expected model prediction without considering input features.
        - 🟡 **Feature Contributions**: Bars represent individual feature impact.
        - 🟡 **Output Prediction**: Sum of base value and contributions gives final output.
        """
    )
    st.pyplot(fig)

# SHAP Force Plot
force_plot_html = shap.force_plot(
    base_value=explainer.expected_value[1],
    shap_values=shap_values_single[0][:, 1],
    features=sample_transformed.iloc[0],
    feature_names=sample_transformed.columns.tolist()
)

# Explanation column
st.markdown(
    """
    ### Column Explanations
    - 🟡 **Input Streaming**: Displays user inputs dynamically in real-time.
    - 🟡 **SHAP Waterfall Plot**: Visualizes how each feature contributes to the model prediction.
    - 🟡 **SHAP Force Plot**: Interactive plot showing positive/negative feature contributions.
    \n\n\n\n""",
    unsafe_allow_html=True,
)

# Add SHAP JS visualization
force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
st.markdown("### SHAP Waterfall Plot")
st.components.v1.html(force_plot_html, height=400)


st.markdown("### Model performance")
# Model score
cols = st.columns(5)
color = 'blue'

thresholds = 0.32
y_score = model.predict_proba(X)[:, 1]
y_pred = (y_score >= thresholds).astype(int)

# Accuracy Score
accuracy_result = round(accuracy_score(y, y_pred) * 100, 2)
cols[0].markdown("**Accuracy Score**\n\nPercentage of correct predictions.")
cols[0].altair_chart(make_donut(accuracy_result, 
                                'Accuracy Score:',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# F1 Score
f1_result = (f1_score(y, y_pred) * 100).round(2)
cols[1].markdown("**F1 Score**\n\nBalance of Precision and Recall.")
cols[1].altair_chart(make_donut(f1_result, 
                                'F1 Score',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# Recall Score
recall_result = (recall_score(y, y_pred) * 100).round(2)
cols[2].markdown("**Recall Score**\n\nProportion of actual positives identified.")
cols[2].altair_chart(make_donut(recall_result, 
                                'Recall Score',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# Precision Score
precision_result = (precision_score(y, y_pred) * 100).round(2)
cols[3].markdown("**Precision Score**\n\nProportion of positive predictions that are correct.")
cols[3].altair_chart(make_donut(precision_result, 
                                'Precision Score:',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# ROC AUC Score
roc_auc = (roc_auc_score(y, y_score)*100).round(2)
cols[4].markdown("**ROC AUC Score**\n\nAbility to distinguish between classes.")
cols[4].altair_chart(make_donut(roc_auc,
                                'ROC AUC Score:',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# perm_importance
cols = st.columns(2)
perm_importance = permutation_importance(model, X, model.predict(X), n_repeats=5, random_state=42)

perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)
fig = px.bar(perm_importance_df, y='Feature', x='Importance', orientation='h',
             labels={'Importance': 'Permutation Importance', 'Feature': 'Feature'},
             title='Permutation Importance of Features for model')
fig.update_layout(yaxis=dict(autorange='reversed'))
cols[0].plotly_chart(fig)