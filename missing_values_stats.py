import pandas as pd

df = pd.read_excel('result_no_text.xlsx')
lines = []
for i, col in enumerate(df.columns, 1):
    sub_df = df.iloc[3:198, i-1]
    filled = sub_df.notna().sum()
    total = len(sub_df)
    percentage = (filled / total) * 100 if total > 0 else 0
    line = f"Column {i}: Filled={filled}, Total={total}, Percentage={percentage:.2f}%"
    print(line)
    lines.append(line)
with open('missing_stats.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f"Processed {len(df.columns)} columns, statistics saved to missing_stats.txt")