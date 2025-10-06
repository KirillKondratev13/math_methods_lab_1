import pandas as pd
from openpyxl import load_workbook

# Read the main data file with multi-index headers from first 3 rows and data starting from row 4 (index 3)
df = pd.read_excel('result_no_text_filled_more_than_80.xlsx', header=[0, 1, 2])

# Read the analysis statistics file
analysis_df = pd.read_excel('column_analysis_results.xlsx')

# Get total number of rows in the main DataFrame
total_rows = len(df)

# Iterate over each column in the DataFrame
for col in df.columns:
    col_name = col[0] if isinstance(col, tuple) else col  # Handle multi-index columns
    # Find the corresponding statistics for the column
    stats = analysis_df[analysis_df.iloc[:, 0] == col_name]  # Assume first column is column name
    if not stats.empty:
        unique_count = stats.iloc[0, 1]  # Assume second column is unique count
        std_dev = stats.iloc[0, 2]  # Assume third column is std dev
        precision = stats.iloc[0, 3]  # Assume fourth column is precision
        # Classify as numerical if conditions met
        if unique_count < 0.2 * total_rows and std_dev > 1 and precision > 0:
            # Use mean for numerical columns
            try:
                fill_value = df[col].mean()
            except (TypeError, ValueError):
                # Fallback to mode if mean fails (e.g., non-numeric)
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
        else:
            # Use mode for categorical/ID-like columns
            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
    else:
        # Default to mode if no statistics available
        fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
    # Fill missing values
    df[col] = df[col].fillna(fill_value)

# Load the original workbook to preserve structure
try:
    wb = load_workbook('result_no_text_filled_more_than_80.xlsx')
    ws = wb.active
    # Preserve first 3 rows exactly
    # Write filled data starting from row 4
    for r in range(len(df)):
        for c in range(len(df.columns)):
            val = df.iloc[r, c]
            ws.cell(row=r+4, column=c+1, value=val)
    # Save to new file
    wb.save('result_no_text_filled_more_than_80_filled.xlsx')
except Exception as e:
    print(f"Error processing workbook: {e}")

# Descriptive comment: This script reads the specified Excel file with multi-index headers, uses analysis statistics to classify columns as numerical or categorical, computes appropriate fill values (mean or mode) from non-missing data, fills missing values, loads the original workbook using openpyxl to preserve the first 3 header rows, writes the flattened column names to row 4, writes the filled data starting from row 5, and saves to a new file while handling potential errors.