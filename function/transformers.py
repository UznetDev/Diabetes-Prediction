import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.epsilon = 1e-5

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data['PregnancyRatio'] = data['Pregnancies'] / (data['Age'] + self.epsilon)
        data['RiskScore'] = (0.5 * data['Glucose'] + 0.3 * data['BMI'] + 0.2 * data['Age'])
        data['InsulinEfficiency'] = (data['Insulin'] + self.epsilon) / (data['Glucose'] + self.epsilon)
        data['Glucose_BMI'] = (data['Glucose'] + self.epsilon) / (data['BMI'] + self.epsilon)
        data['BMI_Age'] = data['BMI'] * data['Age']
        return data


class WoEEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_bins = {
            'Pregnancies': [-np.inf, 1.7, 5.1, 8.5, np.inf],
            'Glucose': [-np.inf, 90.6, 119.4, 159.2, np.inf],
            'BMI': [-np.inf, 26.84, 38.26, np.inf],
            'RiskScore': [-np.inf, 55.61, 77.51, 99.41, np.inf],
        }
        self.woe_mappings = {}

    def fit(self, X, y):
        y = pd.Series(y, name='target')
        for feature, bins in self.feature_bins.items():
            X[f'{feature}_cat'] = pd.cut(X[feature], bins=bins)
            woe_df = self._calculate_woe(X, f'{feature}_cat', y)
            self.woe_mappings[feature] = woe_df.set_index(f'{feature}_cat')['WOE'].to_dict()
        return self

    def transform(self, X):
        data = X.copy()
        for feature in self.feature_bins.keys():
            data[f'{feature}_cat'] = pd.cut(data[feature], bins=self.feature_bins[feature])
            data[f'{feature}_woe'] = data[f'{feature}_cat'].map(self.woe_mappings[feature])
            data.drop(columns=[f'{feature}_cat'], inplace=True)
        return data

    def _calculate_woe(self, data, feature_name, y):
        data['target'] = y
        grouped = data.groupby(feature_name, observed=False)['target'].value_counts().unstack(fill_value=0)
        grouped.columns = ['non_events', 'events']
        grouped['event_rate'] = grouped['events'] / grouped['events'].sum()
        grouped['non_event_rate'] = grouped['non_events'] / grouped['non_events'].sum()
        grouped['WOE'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
        return grouped.reset_index()
    

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]