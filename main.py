import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loader import df, col
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
from function import *
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




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

    # eatures = ['pregnancies', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age']
    features = columns

    if features.count('glucose'):
        features.remove('glucose')

    if features:
        X = df[features]
        y = df['glucose']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        fig = go.Figure()

        for i, feature in enumerate(features):
            X_feature = X_scaled[:, i].reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X_feature, y, random_state=0)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            x_range = np.linspace(X_feature.min(), X_feature.max(), 100)
            y_range = model.predict(x_range.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(x=X_train.squeeze(), y=y_train, name=f'Train {feature}', mode='markers', marker=dict(opacity=0.5)))
            fig.add_trace(go.Scatter(x=X_test.squeeze(), y=y_test, name=f'Test {feature}', mode='markers', marker=dict(opacity=0.5)))
            
            fig.add_trace(go.Scatter(x=x_range, y=y_range, name=f'Regression {feature}', mode='lines'))

        fig.update_layout(
            xaxis_title="Normalized Feature Value",
            yaxis_title="Glucose",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )


        st.plotly_chart(fig)


st.write_stream(stream_data)


