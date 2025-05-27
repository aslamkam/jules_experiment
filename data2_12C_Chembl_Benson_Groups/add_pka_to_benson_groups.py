import pandas as pd

# Load the Excel and CSV files
excel_df = pd.read_excel("Amines_12C_CHEMBL_benson.xlsx")
csv_df = pd.read_csv("amines-pka-dataset.csv")

# Standardize the key column names for merging
csv_df.rename(columns={'inchi_key': 'InChI'}, inplace=True)

# Merge and keep only matching entries (inner join)
merged_df = pd.merge(excel_df, csv_df[['InChI', 'pka_value']], on='InChI', how='inner')

# Save the result
merged_df.to_excel("Amines_12C_CHEMBL_matched_with_pKa.xlsx", index=False)

print("Filtered entries with matching InChI keys saved to 'Amines_12C_CHEMBL_matched_with_pKa.xlsx'.")
