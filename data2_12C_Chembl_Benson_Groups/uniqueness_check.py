import pandas as pd

# Paths to your Excel files
DATA1_PATH = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\External Data Set Test\External Data Set\ccus_96_molecules_benson.xlsx"
DATA2_PATH = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa.xlsx"

# Read the data
df1 = pd.read_excel(DATA1_PATH)
df2 = pd.read_excel(DATA2_PATH)

# Normalize InChIKey columns: drop any whitespace issues
df1['Inchi_Key'] = df1['Inchi Key'].astype(str).str.strip()
df2['InChI_Key'] = df2['InChI'].astype(str).str.strip()

# Create a set of InChIKeys from Data1
inchi_keys_1 = set(df1['Inchi_Key'])

# Mark matches in Data2
df2['InData1'] = df2['InChI_Key'].apply(lambda k: k in inchi_keys_1)

# Filter the matching rows
matches = df2[df2['InData1']]
unique_matched_inchi_keys = set(matches['InChI_Key'])

# Output results
if not matches.empty:
    print(f"Found {len(matches)} matching molecule(s) based on InChIKey.")
    print(f"Unique InChIKeys from Data1 found in Data2: {len(unique_matched_inchi_keys)}")
    print(matches[['Formula', 'Smiles', 'InChI_Key', 'pka_value']])
else:
    print("No molecules from Data1 were found in Data2 based on InChIKey.")

# Optionally, save the matching results to a new file
OUTPUT_PATH = r"matching_molecules_by_inchikey.xlsx"
matches.to_excel(OUTPUT_PATH, index=False)
print(f"Matching entries saved to {OUTPUT_PATH}")
