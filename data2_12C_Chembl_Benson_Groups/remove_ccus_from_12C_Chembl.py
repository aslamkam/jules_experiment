import pandas as pd

# File paths
DATA1_PATH = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\External Data Set Test\External Data Set\ccus_96_molecules_benson.xlsx"
DATA2_PATH = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa.xlsx"
OUTPUT_PATH = "Amines_12C_CHEMBL_benson_matched_with_pKa_and_removed_ccus.xlsx"

# Read the data
df1 = pd.read_excel(DATA1_PATH)
df2 = pd.read_excel(DATA2_PATH)

# Normalize InChIKey columns
df1['Inchi_Key'] = df1['Inchi Key'].astype(str).str.strip()
df2['InChI_Key'] = df2['InChI'].astype(str).str.strip()

# Get InChIKeys from Data1
inchi_keys_1 = set(df1['Inchi_Key'])

# Filter out entries from Data2 that are in Data1
original_count = len(df2)
filtered_df2 = df2[~df2['InChI_Key'].isin(inchi_keys_1)]
filtered_count = len(filtered_df2)
removed_count = original_count - filtered_count

# Save the result
filtered_df2.to_excel(OUTPUT_PATH, index=False)

# Print stats
print(f"Original entries in Data2: {original_count}")
print(f"Entries removed based on Data1 InChIKeys: {removed_count}")
print(f"Entries remaining after removal: {filtered_count}")
print(f"Filtered data saved to {OUTPUT_PATH}")
