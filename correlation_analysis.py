# Required libraries:
# pip install pandas openpyxl seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_correlation(file_path):
    """
    Reads an Excel file, performs correlation analysis, and saves the results
    in multiple formats.

    Args:
        file_path (str): The path to the Excel file.
    """
    try:
        # 1. Read header rows to determine column names
        headers_df = pd.read_excel(file_path, header=None, nrows=3, engine='openpyxl')
        
        # 2. Form column names
        column_names = []
        for col_idx in headers_df.columns:
            name_row_2 = headers_df.iloc[1, col_idx]
            name_row_3 = headers_df.iloc[2, col_idx]
            
            if pd.notna(name_row_3) and str(name_row_3).strip():
                column_names.append(name_row_3)
            elif pd.notna(name_row_2) and str(name_row_2).strip():
                column_names.append(name_row_2)
            else:
                column_names.append(f"Unnamed: {col_idx}")

        # 3. Read the data
        data_df = pd.read_excel(file_path, header=None, skiprows=3, names=column_names, engine='openpyxl')

        # 4. Convert data to numeric
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        # 5. Calculate the Pearson correlation matrix
        correlation_matrix = data_df.corr(method='pearson')

        # --- Saving Results to Excel Files ---

        # Task 1: Save the raw correlation matrix
        correlation_matrix.to_excel('correlation_matrix.xlsx', engine='openpyxl')
        print("Raw correlation matrix saved to 'correlation_matrix.xlsx'")

        # Task 2: Save the matrix with heatmap-style coloring
        # We can use pandas' styling capabilities for this
        styled_matrix = correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
        styled_matrix.to_excel('correlation_heatmap.xlsx', engine='openpyxl')
        print("Correlation matrix with heatmap coloring saved to 'correlation_heatmap.xlsx'")

        # Task 3: Create and save a sorted list of unique correlation pairs
        # Unstack the matrix to get pairs, drop self-correlations and duplicates
        sorted_pairs = correlation_matrix.unstack().reset_index()
        sorted_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        
        # Remove self-correlations (A-A)
        sorted_pairs = sorted_pairs[sorted_pairs['Feature 1'] != sorted_pairs['Feature 2']]

        # Remove duplicate pairs (B-A if A-B exists)
        sorted_pairs['sorted_features'] = sorted_pairs.apply(lambda x: tuple(sorted((x['Feature 1'], x['Feature 2']))), axis=1)
        sorted_pairs = sorted_pairs.drop_duplicates(subset='sorted_features').drop(columns='sorted_features')

        # Sort by the absolute value of the correlation in descending order
        sorted_pairs['Abs Correlation'] = sorted_pairs['Correlation'].abs()
        sorted_pairs = sorted_pairs.sort_values(by='Abs Correlation', ascending=False).drop(columns='Abs Correlation')

        sorted_pairs.to_excel('sorted_correlations.xlsx', index=False, engine='openpyxl')
        print("Sorted correlation pairs saved to 'sorted_correlations.xlsx'")

        # --- Original Visualization ---

        # 6. Visualize with a standard heatmap image
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            linewidths=.5,
            vmin=-1, 
            vmax=1
        )
        plt.title('Pearson Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        # 7. Save and show the heatmap image
        heatmap_filename = 'correlation_heatmap.png'
        plt.savefig(heatmap_filename)
        print(f"Heatmap image saved as '{heatmap_filename}'")
        plt.show()

        # 8. Print the correlation matrix to the console
        print("\nCorrelation Matrix:")
        print(correlation_matrix)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    excel_file_path = 'result_no_text_filled_more_than_80_filled.xlsx'
    analyze_correlation(excel_file_path)