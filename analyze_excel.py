import pandas as pd

# Load the Excel file
df = pd.read_excel('result.xlsx')

# For each column, extract first three rows and format output
lines = []
for i, col in enumerate(df.columns, 1):
    values = df[col].head(3).tolist()
    # Pad with empty strings if fewer than 3 rows
    while len(values) < 3:
        values.append('')
    # Format as specified
    line = f"Column {i}: Row1='{values[0]}', Row2='{values[1]}', Row3='{values[2]}'"
    print(line)
    lines.append(line)

# Write the list to 'column_names.txt'
with open('column_names.txt', 'w') as f:
    f.write('\n'.join(lines))

# Summary
print(f"Processed {len(df.columns)} columns.")