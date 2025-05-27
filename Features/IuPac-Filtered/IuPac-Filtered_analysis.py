import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For heatmaps

# Define the file path for the IuPac dataset
iupac_file_path = 'C:/Users/kamal/OneDrive - University of Guelph/My Research/Features/IuPac-Filtered/IuPac-Filtered.csv'

# Load the IuPac dataset
try:
    df_iupac = pd.read_csv(iupac_file_path)
except FileNotFoundError:
    print(f"Error: The file '{iupac_file_path}' was not found.")
    print("Please ensure the file path is correct.")
    exit()

print("--- IuPac Data Overview ---")

# 1. Count the number of rows
num_rows_iupac = df_iupac.shape[0]
print(f"Number of rows in the IuPac dataset: {num_rows_iupac}")

# 2. Count the number of unique IDs/InChI Keys
# Assuming 'InChI' is the most robust unique identifier if 'unique_ID' is not consistently unique
num_unique_inchi_iupac = df_iupac['InChI'].nunique()
print(f"Number of unique InChI Keys in IuPac: {num_unique_inchi_iupac}")

# 3. Range of pka_value
# Ensure 'pka_value' column is numeric, coercing errors will turn non-numeric into NaN
df_iupac['pka_value'] = pd.to_numeric(df_iupac['pka_value'], errors='coerce')
pka_min_iupac = df_iupac['pka_value'].min()
pka_max_iupac = df_iupac['pka_value'].max()
print(f"Range of pka_value: {pka_min_iupac:.2f} - {pka_max_iupac:.2f}")

print("\n--- IuPac Missing Values and Distributions ---")

# 4. Missing Values Analysis for key columns
print("\nMissing values per column in IuPac dataset:")
print(df_iupac[['pka_value', 'SMILES', 'T', 'solvent']].isnull().sum())

# 5. Histogram of pka_value values
plt.figure(figsize=(10, 6))
plt.hist(df_iupac['pka_value'].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of pka_value in IuPac Dataset')
plt.xlabel('pka_value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# 6. Distribution of Temperature (T)
# Ensure 'T' column is numeric
df_iupac['T'] = pd.to_numeric(df_iupac['T'], errors='coerce')
plt.figure(figsize=(8, 5))
df_iupac['T'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Temperature (T) in IuPac Dataset')
plt.xlabel('Temperature (°C)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\nValue counts for Temperature (T):")
print(df_iupac['T'].value_counts())


# 7. Length of SMILES Strings Distribution
df_iupac['SMILES_Length'] = df_iupac['SMILES'].astype(str).apply(len)
plt.figure(figsize=(10, 6))
plt.hist(df_iupac['SMILES_Length'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of SMILES String Lengths in IuPac Dataset')
plt.xlabel('SMILES String Length')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

print("\n--- IuPac Categorical Feature Analysis ---")

# 8. Unique pka_type values
print("\nUnique values and counts for 'pka_type' in IuPac:")
print(df_iupac['pka_type'].value_counts())

# 9. Unique solvent values
print("\nUnique values and counts for 'solvent' in IuPac:")
print(df_iupac['solvent'].value_counts())