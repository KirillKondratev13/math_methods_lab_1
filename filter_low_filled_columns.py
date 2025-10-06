import pandas as pd
import re

with open('missing_stats.txt', 'r') as f:
    lines = f.readlines()

kept_numbers = []

for line in lines:
    match = re.search(r'Column (\d+): Filled=\d+, Total=\d+, Percentage=([\d.]+)%', line)
    if match:
        number = int(match.group(1))
        percentage = float(match.group(2))
        if percentage > 30.0:
            kept_numbers.append(number)

print(kept_numbers)

df = pd.read_excel('result_no_text.xlsx')

df_filtered = df.iloc[:, [num-1 for num in kept_numbers]]

df_filtered.to_excel('result_no_text_filled.xlsx', index=False)

print(f"Filtered to {len(kept_numbers)} columns with >30% filled data, saved to result_no_text_filled.xlsx")