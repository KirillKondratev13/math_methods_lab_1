#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 3 — 03_train_model.py

Purpose:
- Load X_train, y_train, and the list of feature names from processed_data.pkl.
- Import and initialize catboost.CatBoostClassifier with verbose=0 for silent training and random_state=42.
- Train the model on X_train and y_train.
- Extract feature importances from the model's feature_importances_ attribute.
- Create a pandas DataFrame with columns: 'feature' (feature names) and 'importance' (their importances).
- Save this DataFrame to feature_importances.csv, sorted in descending order of importance.

Dependencies:
- pickle
- pandas
- catboost

Steps:
1. Import necessary libraries: pickle, pandas, catboost.
2. Load the data from processed_data.pkl using pickle. Expect a dictionary with keys 'X_train', 'y_train', and 'feature_names'.
3. Initialize the CatBoostClassifier with the specified parameters.
4. Fit the model on X_train and y_train.
5. Retrieve the feature importances.
6. Create a DataFrame with feature names and their importances.
7. Sort the DataFrame by importance in descending order.
8. Save the DataFrame to 'feature_importances.csv' without the index.
"""

import pickle
import pandas as pd
from catboost import CatBoostClassifier

def main():
    # Load data
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_train = data['X_train']
    y_train = data['y_train']
    feature_names = data['feature_names']

    print("Loaded data from processed_data.pkl")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Feature names length: {len(feature_names)}")

    # Initialize and train model
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)

    print("Model trained successfully")

    # Extract feature importances
    importances = model.feature_importances_

    # Create DataFrame
    df_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort by importance descending
    df_importances = df_importances.sort_values('importance', ascending=False)

    # Save to CSV
    df_importances.to_csv('feature_importances.csv', index=False)

    print("Feature importances saved to feature_importances.csv")
    print(f"Top 5 features:\n{df_importances.head()}")

if __name__ == "__main__":
    main()