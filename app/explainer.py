import streamlit as st
import shap
import time
from loader import model
import matplotlib.pyplot as plt


def app(input_data):
    sample_transformed = model.named_steps['feature_engineering'].transform(input_data)
    explainer = shap.TreeExplainer(model.named_steps['model'])
    shap_values_single = explainer.shap_values(sample_transformed)

    shap_values_class_1 = shap_values_single[0][:, 1]  


    def stream_data():
        text = f"""
Your inputs:\n
`Pregnancies`: {float(input_data.iloc[0]['Pregnancies'])}\n
`Glucose`: {float(input_data.iloc[0]['Pregnancies'])}\n
`Insulin`: {float(input_data.iloc[0]['Pregnancies'])}\n
`BMI`: {float(input_data.iloc[0]['Pregnancies'])}\n
`Age`: {float(input_data.iloc[0]['Pregnancies'])}
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