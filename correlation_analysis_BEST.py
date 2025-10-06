# Required libraries:
# pip install pandas openpyxl seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_correlation(file_path):
    """
    Reads an Excel file, calculates the Pearson correlation matrix for numeric columns,
    and generates a heatmap.

    Args:
        file_path (str): The path to the Excel file.
    """
    try:
        # 1. Read header rows (2nd and 3rd) to determine column names
        headers_df = pd.read_excel(file_path, header=None, nrows=3, engine='openpyxl')
        
        # 2. Form column names based on the specified logic
        column_names = []
        for col_idx in headers_df.columns:
            name_row_2 = headers_df.iloc[1, col_idx]
            name_row_3 = headers_df.iloc[2, col_idx]
            
            # Use name from 3rd row if available
            if pd.notna(name_row_3) and str(name_row_3).strip():
                column_names.append(name_row_3)
            # Otherwise, use name from 2nd row if available
            elif pd.notna(name_row_2) and str(name_row_2).strip():
                column_names.append(name_row_2)
            # Fallback to pandas default name
            else:
                column_names.append(f"Unnamed: {col_idx}")

        # 3. Read the data from the 4th row onwards, assigning the custom column names
        data_df = pd.read_excel(file_path, header=None, skiprows=3, names=column_names, engine='openpyxl')

        # 4. Convert all data to numeric, replacing non-numeric values with NaN
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

        # 5. Calculate the Pearson correlation matrix
        correlation_matrix = data_df.corr(method='pearson')

        # 6. Visualize the correlation matrix with a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            linewidths=.5
        )
        plt.title('Pearson Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off

        # 7. Save the heatmap to a file
        heatmap_filename = 'correlation_heatmap.png'
        plt.savefig(heatmap_filename)
        print(f"Heatmap saved as '{heatmap_filename}'")
        
        # Display the plot
        plt.show()

        # 8. Print the correlation matrix to the console
        print("\nCorrelation Matrix:")
        print(correlation_matrix)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- IMPORTANT ---
    # Replace the placeholder with the actual path to your Excel file.
    excel_file_path = 'result_no_text_filled_more_than_80_filled.xlsx' # 'path/to/your/excel_file.xlsx'
    
    analyze_correlation(excel_file_path)