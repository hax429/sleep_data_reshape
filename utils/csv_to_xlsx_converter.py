import pandas as pd

# Read the CSV file
df = pd.read_csv('/template/definitions.csv')

# Convert to Excel format
output_path = '/template/definitions.xlsx'
df.to_excel(output_path, index=False)

print(f"Successfully converted definitions.csv to {output_path}")