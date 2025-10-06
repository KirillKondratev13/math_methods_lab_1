import pandas as pd
import openpyxl

def calculate_precision(series):
    """Calculate the maximum number of decimal places in numerical values."""
    max_decimals = 0
    for val in series.dropna():
        if isinstance(val, (int, float)):
            str_val = str(val)
            if '.' in str_val:
                decimals = len(str_val.split('.')[1])
                max_decimals = max(max_decimals, decimals)
    return max_decimals

try:
    # Read the Excel file, skipping the first 3 rows (headers), data starts from row 4
    df = pd.read_excel('result_no_text_filled_more_than_80.xlsx', skiprows=3)

    # List to collect results
    results = []

    # Iterate through each column
    for col_num, col_name in enumerate(df.columns):
        col_data = df[col_name].dropna()

        # Check if column is numerical
        is_numerical = pd.api.types.is_numeric_dtype(df[col_name])

        if is_numerical and not col_data.empty:
            mean = col_data.mean()
            std_dev = col_data.std()
            precision = calculate_precision(col_data)
        else:
            mean = 'N/A'
            std_dev = 'N/A'
            precision = 'N/A'

        # Always calculate unique values
        unique_count = len(col_data.unique())

        # Append to results
        results.append({
            'Column Number': col_num,
            'Mean Value': mean,
            'Value Dispersion': std_dev,
            'Precision': precision,
            'Number of Unique Values': unique_count
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Save to Excel file
    results_df.to_excel('column_analysis_results.xlsx', index=False)

except FileNotFoundError:
    print("Error: The file 'result_no_text_filled_more_than_80.xlsx' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

"""
This script analyzes the Excel file 'result_no_text_filled_more_than_80.xlsx' containing patient data starting from row 4.
It iterates through each column, calculating for numerical columns: mean, standard deviation, and maximum decimal places (precision).
For non-numerical columns, mean, standard deviation, and precision are set to 'N/A'.
It always calculates the number of unique values in each column.
Results are saved to an Excel file 'column_analysis_results.xlsx' with columns: Column Number, Mean Value, Value Dispersion, Precision, Number of Unique Values.
The script handles missing values by dropping them for calculations and includes error handling for file issues.
"""