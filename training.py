import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from function.model import Model

data = pd.read_csv('datasets/diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']


Model.fit(X, y)

y_pred = Model.predict_proba(X)[:, 1]

print("ROC_AUC Score: ", (roc_auc_score(y, y_pred) * 100).round(2))

joblib.dump(Model, 'model.pkl')