import pandas as pd
import re

# Read 'column_analysis.txt' line by line
text_nums = []
with open('column_analysis.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Parse the line
        match = re.match(r"Column (\d+): .* Type='(\w+)'", line)
        if match:
            num = int(match.group(1))
            typ = match.group(2)
            if typ == 'text':
                text_nums.append(num)

# Print the list
print("Text column numbers (1-based):", text_nums)

# Load 'result.xlsx'
df = pd.read_excel('result.xlsx')

# Drop the text columns
df_filtered = df.drop(df.columns[[num-1 for num in text_nums]], axis=1)

# Save to 'result_no_text.xlsx'
df_filtered.to_excel('result_no_text.xlsx', index=False)

# Print confirmation
print("Filtered DataFrame saved to 'result_no_text.xlsx'")