import dill as pickle
import pandas as pd 


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)