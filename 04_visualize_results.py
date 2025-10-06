#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 4: 04_visualize_results.py

Purpose: Visualize the top-15 most important features from the trained CatBoost model.

Import necessary libraries: pandas, seaborn, matplotlib.pyplot.

Load data from feature_importances.csv into a pandas DataFrame. Assumed to contain columns 'feature' (feature names) and 'importance' (importance values), sorted in descending order of importance.

Keep only the first 15 rows (top-15 features).

Create a horizontal bar plot using seaborn.barplot.

- Set data=DataFrame, x='importance', y='feature'.
- Set orient='h' for horizontal orientation.
- Arrange features so the most important is at the top (since data is already sorted, this will be automatic).
- Give the plot title "Топ-15 важных признаков для определения группы наблюдения" using plt.title().
- Set figure size plt.figure(figsize=(10, 8)) for better readability.
- Ensure all feature names on the plot are readable: use plt.xticks(fontsize=10), plt.yticks(fontsize=10), and possibly plt.tight_layout() to prevent clipping.
- Add axis labels: plt.xlabel('Важность признака'), plt.ylabel('Название признака').

Save the plot to file feature_importance_top15.png with high resolution (dpi=300) using plt.savefig(). Close the figure after saving with plt.close().
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Load data
    df = pd.read_csv('feature_importances.csv')

    # Keep top 15
    df_top15 = df.head(15)

    print("Loaded feature importances")
    print(f"Total features: {len(df)}")
    print(f"Top 15 features:\n{df_top15}")

    # Create plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_top15, x='importance', y='feature', orient='h')

    plt.title('Топ-15 важных признаков для определения группы наблюдения')
    plt.xlabel('Важность признака')
    plt.ylabel('Название признака')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save plot
    plt.savefig('feature_importance_top15.png', dpi=300)
    plt.close()

    print("Plot saved to feature_importance_top15.png")

if __name__ == "__main__":
    main()