import dill as pickle
import pandas as pd 
from PIL import Image


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('data/diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']

summary_plot = Image.open('image/summary_plot.png')