import pandas as pd

# Load the Excel file
file_path = 'Book.xlsx'  # Replace with your actual file path
sheet_name = 'Sheet1'  # Set to a specific sheet name if needed, or None to use the first sheet

# Read the Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

# Print each column and its non-empty content
# print(df)

for column in df.columns:
    print(f"Column: {column}")
    non_empty_values = df[column].dropna()
    list = []
    for value in non_empty_values:
        list.append(value)
    print(list)
    print("-" * 40)
