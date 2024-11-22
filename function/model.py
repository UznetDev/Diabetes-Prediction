from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from function.transformers import FeatureEngineering, WoEEncoding, ColumnSelector


selected_columns = [
    'Pregnancies', 'Glucose', 'BMI', 'PregnancyRatio',
    'RiskScore', 'InsulinEfficiency', 'Glucose_BMI', 'BMI_Age',
    'Glucose_woe', 'RiskScore_woe'
]

# Pipeline setup
Model = Pipeline([
    ('feature_engineering', FeatureEngineering()),
    ('woe_encoding', WoEEncoding()),
    ('column_selector', ColumnSelector(selected_columns)),
    ('model', RandomForestClassifier(max_depth=6,
                                     n_estimators=300,
                                     criterion='entropy'))
])