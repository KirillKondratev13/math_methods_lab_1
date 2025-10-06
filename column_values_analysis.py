import pandas as pd
import random

# Load the Excel file with error handling
try:
    df = pd.read_excel('result.xlsx')
except FileNotFoundError:
    print("Error: 'result.xlsx' not found.")
    exit(1)

# Handle empty or small DataFrame (needs at least 4 rows for analysis)
if df.empty or len(df) < 4:
    print("Error: DataFrame is empty or has fewer than 4 rows.")
    exit(1)

# Initialize counters for summary
numeric_count = 0
text_count = 0

# Open file for writing analysis results
with open('column_analysis.txt', 'w', encoding='utf-8') as f:
    # Loop through each column (1-based indexing)
    for col in range(len(df.columns)):
        # Extract column name: check rows 0,1,2, use first non-na non-empty, else 'nan'
        extracted_name = 'nan'
        for i in range(3):
            if i < len(df):
                val = df.iloc[i, col]
                if not pd.isna(val) and str(val).strip() != '':
                    extracted_name = str(val)
                    break
        # Get data from row 3 onwards (0-based index 3), drop NaN values
        data_slice = df.iloc[3:, col].dropna()
        # Determine data type: numeric if all values can be converted to numeric, else text
        if pd.to_numeric(data_slice, errors='coerce').isna().sum() == 0:
            data_type = 'numeric'
            numeric_count += 1
        else:
            data_type = 'text'
            text_count += 1
        # Extract 3 random samples if available, else all available values
        if len(data_slice) >= 3:
            samples = tuple(random.sample(list(data_slice), 3))
        else:
            samples = tuple(data_slice)
        # Format the line for output
        line = f"Column {col+1}: Name='{extracted_name}', Type='{data_type}', Samples={', '.join(str(s) for s in samples)}"
        f.write(line + '\n')

# Print summary statistics
print(f"Total columns: {len(df.columns)}, Numeric: {numeric_count}, Text: {text_count}")
print("Анализ завершен, файл 'column_analysis.txt' создан.")
print("Changes made: Modified column name extraction to check first three rows for non-empty values, updated output format to 'Column {number}: Name='{extracted_name}', Type='{type}', Samples={sample1}, {sample2}, {sample3}', changed output file to 'column_analysis.txt'.")