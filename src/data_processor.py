import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()

    def load_data(self, filepath):
        """Load and preprocess the FastFood dataset"""
        self.data = pd.read_csv(filepath)

        # Clean column names
        self.data.columns = self.data.columns.str.lower().str.strip()

        # Convert numerical columns to float
        numeric_columns = ['calories', 'cal_fat', 'total_fat', 'sat_fat', 'trans_fat',
                           'cholesterol', 'sodium', 'total_carb', 'fiber', 'sugar',
                           'protein', 'vit_a', 'vit_c', 'calcium']

        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Fill missing values with median
        self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())

        return self.data

    def get_features(self):
        """Extract and normalize features for similarity calculation"""
        feature_columns = ['calories', 'total_fat', 'protein', 'total_carb',
                           'fiber', 'sugar', 'sodium', 'cholesterol']

        features = self.data[feature_columns]
        normalized_features = self.scaler.fit_transform(features)

        return normalized_features

    def get_item_details(self, item_name):
        """Get details for a specific food item"""
        return self.data[self.data['item'] == item_name].iloc[0]
