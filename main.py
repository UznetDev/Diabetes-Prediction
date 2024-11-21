import time
from loader import model
from function.function import *
import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("Input Parameters")


st.markdown(
    """
    <div style="text-align: 
    center; 
    font-size: 40px; 
    font-weight: bold; 
    color: #2E86C1;
    margin-bottom: 20px;">
        🌟 Diabetes Prediction App 🌟
    </div>
    <div style="text-align: center; font-size: 18px; color: #5D6D7E; margin-bottom: 60px;">
        Harness the power of machine learning to predict diabetes and provide insights!
    </div>
    """, 
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

mrk = f"""
<div style="background-color: {color}; 
color: white; 
margin-bottom: 50px;
padding: 10px;
max-width: 300px;
text-align: center;
border-radius: 5px; text-align: center;">
    {is_diabetes}
</div>
"""

cols[1].markdown(mrk, unsafe_allow_html=True)
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
    return 80

cols = st.columns(2)

cols[0].write_stream(stream_data)


fig, ax = plt.subplots()

values = shap_values_class_1
base_value = explainer.expected_value[0]
data = sample_transformed.iloc[0]


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

cols[1].pyplot(fig)


force_plot_html = shap.force_plot(
    base_value=explainer.expected_value[1],
    shap_values=shap_values_single[0][:, 1],
    features=sample_transformed.iloc[0],
    feature_names=sample_transformed.columns.tolist()
)

force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
st.components.v1.html(force_plot_html, height=400)

# Model score
cols = st.columns(5)
color = 'blue'

# Accuracy Score
cols[0].markdown("**Accuracy Score**\n\nPercentage of correct predictions.")
cols[0].altair_chart(make_donut(78.57, 
                                'Accuracy Score:',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# F1 Score
cols[1].markdown("**F1 Score**\n\nBalance of Precision and Recall.")
cols[1].altair_chart(make_donut(75.56, 
                                'F1 Score',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# Recall Score
cols[2].markdown("**Recall Score**\n\nProportion of actual positives identified.")
cols[2].altair_chart(make_donut(94.44, 
                                'Recall Score',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# Precision Score
cols[3].markdown("**Precision Score**\n\nProportion of positive predictions that are correct.")
cols[3].altair_chart(make_donut(62.96, 
                                'Precision Score:',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))

# ROC AUC Score
cols[4].markdown("**ROC AUC Score**\n\nAbility to distinguish between classes.")
cols[4].altair_chart(make_donut(83.67, 
                                'ROC AUC Score:',
                                input_color=color,
                                R=140,
                                innerRadius=40,
                                cornerRadius=10))
