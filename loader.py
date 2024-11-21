import dill as pickle
from PIL import Image


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

summary_plot = Image.open('image/summary_plot.png')