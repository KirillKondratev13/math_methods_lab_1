#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 5: 05_shap_explanations.py

Purpose: Use SHAP to explain the CatBoost model predictions.

- Load the trained CatBoost model from model.pkl.
- Load X_test and y_test from processed_data.pkl.
- Use shap.TreeExplainer to explain predictions on test data.
- Compute SHAP values for the first 5 examples from X_test.
- Print textual explanations for each example, showing contribution of each feature to the prediction.
- Save SHAP values to shap_values.csv.
- Create visualizations: summary plot for all features and waterfall plot for one example, saving them as shap_summary.png and shap_waterfall.png.
"""

import pickle
import pandas as pd
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Mapping for observation groups
group_mapping = {
    '0': 'хрАГ',
    '1': 'здоровые',
    '2': 'ГАГ',
    '3': 'ум.ПЭ',
    '4': 'тяж.ПЭ'
}

def main():
    # Load model
    model = CatBoostClassifier()
    model.load_model('model.pkl')

    # Load test data
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']

    print("Loaded model and test data")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP for first 5 examples
    X_sample = X_test[:5]
    shap_values_sample = explainer.shap_values(X_sample)

    print("Computed SHAP values for first 5 test examples")
    print(f"shap_values_sample shape: {shap_values_sample.shape}")

    # Compute SHAP for all X_test for summary plot
    shap_values = explainer.shap_values(X_test)

    print("Computed SHAP values for all test data")
    print(f"shap_values shape: {shap_values.shape}")

    # For multiclass, shap_values is list of arrays, one per class
    # We'll focus on the predicted class for each example
    predictions = model.predict(X_sample)
    prediction_probs = model.predict_proba(X_sample)

    for i in range(5):
        pred_class = int(predictions[i])
        true_class = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        print(f"\nExample {i+1}: Predicted class {group_mapping[str(pred_class)]}, True class {group_mapping[true_class]}")
        print(f"Prediction probabilities: {prediction_probs[i]}")

        # SHAP values for the predicted class
        sv = shap_values_sample[pred_class][i]
        feature_contribs = list(zip(feature_names, sv))
        # Sort by absolute contribution
        feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top contributing features:")
        for feat, val in feature_contribs[:10]:  # Top 10
            direction = "increased" if val > 0 else "decreased"
            print(f"  {feat}: {val:.4f} ({direction} probability)")

    # Save detailed SHAP results to Excel
    n_samples, n_features, n_classes = shap_values_sample.shape
    results = []
    for i in range(5):
        pred_class = int(predictions[i])
        true_class = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        base_val = explainer.expected_value[pred_class]
        prob = prediction_probs[i][pred_class]
        for j, feat in enumerate(feature_names):
            shap_val = shap_values_sample[i, j, pred_class]
            results.append({
                'example_index': i,
                'predicted_class': group_mapping[str(pred_class)],
                'true_class': group_mapping[true_class],
                'feature_name': feat,
                'shap_value': shap_val,
                'base_value': base_val,
                'probability': prob
            })
    results_df = pd.DataFrame(results)
    results_df.to_excel('shap_results.xlsx', index=False)

    print("Detailed SHAP results saved to shap_results.xlsx")

    print("SHAP values saved to shap_values.csv")

    # Summary plot for all features (using full X_test for better plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Summary plot saved to shap_summary.png")

    # Note: Waterfall plot skipped due to SHAP version compatibility issues
    print("Waterfall plot skipped (SHAP version compatibility)")

if __name__ == "__main__":
    main()