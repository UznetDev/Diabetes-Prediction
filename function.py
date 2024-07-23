from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from loader import df, pd


def train_model(data):
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def predict_diabetes(model, input_data, feature_columns):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    for column in feature_columns:
        if column not in input_data:
            input_df[column] = df[column].mean()
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction, probability