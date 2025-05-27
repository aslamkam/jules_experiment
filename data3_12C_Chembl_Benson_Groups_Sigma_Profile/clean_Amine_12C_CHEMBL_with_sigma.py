import pandas as pd
import ast

# Load the Excel file
df = pd.read_excel("Amines_12C_CHEMBL_with_sigma.xlsx")

# Optional: convert sigma_profile strings to actual lists (if read as strings)
def parse_sigma(val):
    try:
        # Check if already a list
        if isinstance(val, list):
            return val
        return ast.literal_eval(val)
    except:
        return []

# Apply parsing
df['sigma_profile'] = df['sigma_profile'].apply(parse_sigma)

# Remove rows where sigma_profile is empty
df_cleaned = df[df['sigma_profile'].apply(lambda x: len(x) > 0)]

# Save the cleaned file
df_cleaned.to_excel("Amines_12C_CHEMBL_with_sigma_cleaned.xlsx", index=False)
