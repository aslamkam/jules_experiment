import pandas as pd

# Load the Excel file
file_path = 'Chembl_amines.xlsx'
df = pd.read_excel(file_path)

# Check for duplicate SMILES strings
duplicate_smiles = df[df.duplicated(subset='Smiles', keep=False)]

# Count unique duplicated SMILES and total duplicate rows
num_duplicates = duplicate_smiles['Smiles'].nunique()
total_duplicate_rows = duplicate_smiles.shape[0]

print(f"Number of unique duplicated SMILES: {num_duplicates}")
print(f"Total number of rows with duplicated SMILES: {total_duplicate_rows}")

# Optional: Show the duplicated SMILES and their corresponding ChEMBL IDs
if total_duplicate_rows > 0:
    print("\nDuplicated SMILES entries:")
    print(duplicate_smiles[['ChEMBL ID', 'Smiles']].sort_values(by='Smiles'))
