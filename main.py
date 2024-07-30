import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loader import df, col
from function import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from loader import df
import streamlit as st



st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if len(input_data) > 0:
    model, accuracy = train_model(df)
    prediction, probability = predict_diabetes(model, input_data, col)

def stream_data():
    text = f"Model Accuracy: {accuracy * 100:.2f}%"

    if len(input_data) > 0:

        re = f'Diabetes' if prediction[0] == 1 else 'No Diabetes'
        text = f"\nPrediction: {re}\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        text = f"\nProbability: {probability[0][1] * 100:.2f}%\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.05)
        return 80

    else:
        text = "\nPlease select values for all features to make a prediction.\n"
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.02)

if len(input_data) > 0:
    cols = st.columns(2)

    cols[0].write_stream(stream_data)

    donut_chart_population = make_donut(round(probability[0][1] * 100, 2), 'Population raise')
    cols[1].altair_chart(donut_chart_population)

#---------------------------------------------------------------------------

features = columns

if features.count('glucose'):
    features.remove('glucose')
cols = st.columns(2)
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

        fig.add_trace(go.Scatter(x=X_train.squeeze(), y=y_train, name=f'Train {feature}', mode='markers',
                                 marker=dict(opacity=0.5)))
        fig.add_trace(
            go.Scatter(x=X_test.squeeze(), y=y_test, name=f'Test {feature}', mode='markers', marker=dict(opacity=0.5)))

        fig.add_trace(go.Scatter(x=x_range, y=y_range, name=f'Regression {feature}', mode='lines'))

    fig.update_layout(
        xaxis_title="Normalized Feature Value",
        yaxis_title="Glucose",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    cols[0].plotly_chart(fig)

fig = px.scatter_3d(df, x='insulin', y='dpf', z='glucose',
                    color='glucose', title='3D Scatter Plot of Insulin, Glucose, and DPF')

X = df[['insulin', 'dpf']]
y = df['glucose']
model = LinearRegression()
model.fit(X, y)

insulin_range = np.linspace(X['insulin'].min(), X['insulin'].max(), 50)
glucose_range = np.linspace(X['dpf'].min(), X['dpf'].max(), 50)
insulin_grid, glucose_grid = np.meshgrid(insulin_range, glucose_range)
dpf_pred = model.predict(np.c_[insulin_grid.ravel(), glucose_grid.ravel()]).reshape(insulin_grid.shape)

fig.add_trace(go.Surface(x=insulin_grid, y=glucose_grid, z=dpf_pred, colorscale='Viridis', opacity=0.5))

cols[1].write(fig)

