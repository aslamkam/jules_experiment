import os
import pandas as pd

def populate_sigma_profile(file1_path, dir1_path, output_path=None):
    """
    Populates a 'Sigma Profile' column in a CSV file by reading data from corresponding text files.

    Args:
        file1_path (str): The path to the input CSV file.
        dir1_path (str): The path to the directory containing the sigma profile .txt files.
        output_path (str, optional): The path to save the updated CSV file. Defaults to None.

    Returns:
        tuple: A tuple containing the updated DataFrame and the count of populated rows.
    """
    # Read the CSV
    df = pd.read_csv(file1_path)
    # Initialize the new column
    if 'Sigma Profile' not in df.columns:
        df['Sigma Profile'] = ''
    df['Sigma Profile'] = df['Sigma Profile'].astype(object)

    populated_count = 0

    # Iterate through rows
    for idx, row in df.iterrows():
        inchi_key = row.get('Inchi Key', '').strip()
        if not inchi_key:
            continue
        
        txt_file = os.path.join(dir1_path, f"{inchi_key}.txt")
        if os.path.isfile(txt_file):
            profile_list = []
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        # Skip empty or whitespace-only lines
                        if not line.strip():
                            continue
                        
                        # THE CRITICAL CHANGE IS HERE.
                        # We are splitting on the tab character '\t'.
                        parts = line.strip().split('\t')
                        
                        # We only care about the second column
                        if len(parts) >= 2:
                            profile_list.append(float(parts[1]))
                
                if profile_list:
                    df.at[idx, 'Sigma Profile'] = str(profile_list)
                    populated_count += 1
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not process file {inchi_key}.txt. Error on line: '{line.strip()}'. Details: {e}. Skipping.")


    # Filter out rows without a populated Sigma Profile
    # Ensure the column exists and is not empty list string '[]'
    df = df[df['Sigma Profile'].astype(bool) & (df['Sigma Profile'] != '[]')]

    # Optionally save the updated DataFrame
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"File saved to {output_path}")

    return df, populated_count

if __name__ == '__main__':
    # I'm not touching these paths again. Make sure they are correct.
    file1 = r"C:\\Users\\kamal\\jules_experiment\\Features\\Chembl-12C\\ChEMBL_amines_12C.csv"
    dir1 = r"C:\\Users\\kamal\\jules_experiment\\Features\\Chembl-12C\\Orca-Sigma-Profile\\ChEMBL_12C_SigmaProfiles_Orca-5899"
    output_file = r"C:\\Users\\kamal\\jules_experiment\\Features\\Chembl-12C\\Orca-Sigma-Profile\\ChEMBL_amines_12C_with_sigma.csv"

    print("Processing file... again.")
    _, count = populate_sigma_profile(file1, dir1, output_path=output_file)
    print(f"Processing complete. Populated {count} rows. Let's hope it's right this time.")