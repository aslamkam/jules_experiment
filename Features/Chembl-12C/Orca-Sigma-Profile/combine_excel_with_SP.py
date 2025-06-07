import os
import pandas as pd

def populate_sigma_profile(file1_path, dir1_path, output_path=None):
    # Read the CSV
    df = pd.read_csv(file1_path)
    # Initialize the new column
    df['Sigma Profile'] = ''
    populated_count = 0

    # Iterate through rows
    for idx, row in df.iterrows():
        inchi_key = row.get('Inchi Key', '').strip()
        if not inchi_key:
            continue
        txt_file = os.path.join(dir1_path, f"{inchi_key}.txt")
        if os.path.isfile(txt_file):
            # Read the sigma profile text
            with open(txt_file, 'r') as f:
                content = f.read().strip()
            # Optionally, transform content to a single-line string
            # e.g., replace newlines with semicolons
            profile_str = content.replace("\n", ";")
            df.at[idx, 'Sigma Profile'] = profile_str
            populated_count += 1

    # Filter out rows without Sigma Profile
    df = df[df['Sigma Profile'] != '']

    # Optionally save the updated DataFrame
    if output_path:
        df.to_csv(output_path, index=False)
    return df, populated_count

if __name__ == '__main__':
    # Paths (update as needed)
    file1 = r"C:\Users\kamal\jules_experiment\Features\Chembl-12C\ChEMBL_amines_12C.csv"
    dir1 = r"C:\Users\kamal\jules_experiment\Features\Chembl-12C\Orca-Sigma-Profile\ChEMBL_12C_SigmaProfiles_Orca-5899"
    output_file = r"C:\Users\kamal\jules_experiment\Features\Chembl-12C\Orca-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"

    updated_df, count = populate_sigma_profile(file1, dir1, output_file)
    print(f"Number of rows populated with Sigma Profile: {count}")
