#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 2 — 02_preprocess_features.py

Purpose:
- Load prepared data from prepared_data.pkl.
- Apply preprocessing: One-Hot Encoding for categorical features, Standard Scaling for numerical features.
- Split data into training and testing sets.
- Save processed data to processed_data.pkl for the next step.

Dependencies:
- pickle
- pandas
- sklearn.compose.ColumnTransformer
- sklearn.preprocessing.OneHotEncoder
- sklearn.preprocessing.StandardScaler
- sklearn.model_selection.train_test_split

Input:
- prepared_data.pkl: Contains X (DataFrame), y (Series), categorical_features (list), numerical_features (list)

Output:
- processed_data.pkl: Contains X_train, X_test, y_train, y_test, feature_names (list of processed feature names)
"""

import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Load prepared data
    with open('prepared_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    categorical_features = data['categorical_features']
    numerical_features = data['numerical_features']

    print("Loaded data from prepared_data.pkl")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    print(f"X_processed shape: {X_processed.shape}")
    print(f"Feature names length: {len(feature_names)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    print(f"Train set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

    # Save processed data
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved processed data to processed_data.pkl")

if __name__ == "__main__":
    main()