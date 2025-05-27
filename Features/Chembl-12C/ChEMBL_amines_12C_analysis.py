import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For heatmaps

# Load the dataset
try:
    df = pd.read_csv('C:/Users/kamal/OneDrive - University of Guelph/My Research/Features/Chembl-12C/ChEMBL_amines_12C.csv')
except FileNotFoundError:
    print("Error: The file 'ChEMBL_amines_12C.csv' was not found at the specified path.")
    print("Please ensure the file path is correct: C:/Users/kamal/OneDrive - University of Guelph/My Research/Features/Chembl-12C/ChEMBL_amines_12C.csv")
    exit()

print("--- Basic Data Overview ---")
# 1. Count the number of rows
num_rows = df.shape[0]
print(f"Number of rows in the dataset: {num_rows}")

# 2. Count the number of unique Inchi Keys
num_unique_inchi_keys = df['Inchi Key'].nunique()
print(f"Number of unique Inchi Keys: {num_unique_inchi_keys}")

# 3. Range of CX Basic pKa
df['CX Basic pKa'] = pd.to_numeric(df['CX Basic pKa'], errors='coerce')
pka_min = df['CX Basic pKa'].min()
pka_max = df['CX Basic pKa'].max()
print(f"Range of CX Basic pKa: {pka_min:.2f} - {pka_max:.2f}")

# 4. Distribution of Amine class (primary, secondary, tertiary)
amine_class_distribution = df['Amine Class'].value_counts(normalize=True) * 100
print("\nPercentage distribution of Amine Class:")
print(amine_class_distribution.round(2))

# 5. Histogram of CX Basic pKa values
plt.figure(figsize=(10, 6))
plt.hist(df['CX Basic pKa'].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of CX Basic pKa Values')
plt.xlabel('CX Basic pKa')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

print("\n--- Advanced Data Exploration ---")

# 6. Missing Values Analysis
print("\nMissing values per column:")
print(df.isnull().sum())

# 7. Correlation Matrix for numerical features
# Identify numerical columns for correlation
numerical_cols = ['Molecular Weight', 'Aromatic Rings', 'CX Basic pKa']
# Ensure these columns are numeric, coercing errors will turn non-numeric into NaN
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values in relevant numerical columns for correlation calculation
# Or use .corr(min_periods=1) to include NaNs but it might be less informative
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# 8. Visualization of other Feature Distributions (Histograms)
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Molecular Weight'].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Molecular Weight')
plt.xlabel('Molecular Weight')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.subplot(1, 2, 2)
plt.hist(df['Aromatic Rings'].dropna(), bins=10, edgecolor='black', alpha=0.7)
plt.title('Distribution of Aromatic Rings')
plt.xlabel('Number of Aromatic Rings')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.show()

# 9. Length of SMILES Strings Distribution
df['Smiles_Length'] = df['Smiles'].astype(str).apply(len)
plt.figure(figsize=(10, 6))
plt.hist(df['Smiles_Length'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of SMILES String Lengths')
plt.xlabel('SMILES String Length')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# 10. Unique Values for Categorical Features
print("\nUnique values and counts for 'Molecular Species':")
print(df['Molecular Species'].value_counts())

print("\nUnique values and counts for 'Amine Class':")
print(df['Amine Class'].value_counts())