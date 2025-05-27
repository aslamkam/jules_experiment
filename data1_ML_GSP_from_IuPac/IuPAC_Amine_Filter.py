import pandas as pd
from rdkit import Chem
import os, sys

def set_cwd_to_script_location():
    """
    Changes the working directory to the directory where the script is located.
    """
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    print(f"Changed working directory to: {os.getcwd()}")

set_cwd_to_script_location()

# Load the CSV file into a pandas DataFrame
csv_file = 'iupac_high-confidence_v2_2.csv'
data = pd.read_csv(csv_file, sep=',', encoding='utf-8')  # Correct delimiter

# Function to check if a molecule contains sulfur or silicon
def contains_sulfur_or_silicon(molecule):
    return molecule.HasSubstructMatch(Chem.MolFromSmarts('[#16]')) or \
           molecule.HasSubstructMatch(Chem.MolFromSmarts('[#14]'))

# Function to check if a molecule contains only allowed atoms: Carbon, Nitrogen, Oxygen, and Hydrogen
def has_only_allowed_atoms(molecule):
    allowed_atomic_numbers = {1, 6, 7, 8}  # H, C, N, O
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() not in allowed_atomic_numbers:
            return False
    return True

# Function to check if a molecule is an amine
def is_amine(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        # Only allow molecules with Carbon, Nitrogen, Oxygen, and Hydrogen
        if not has_only_allowed_atoms(molecule):
            return False
        # Substructure patterns for amines
        primary_amine = Chem.MolFromSmarts('[NX3;H2;!$(NC=O)]')
        secondary_amine = Chem.MolFromSmarts('[NX3;H1;!$(NC=O)]')
        tertiary_amine = Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]')
        return (
            not contains_sulfur_or_silicon(molecule) and
            (molecule.HasSubstructMatch(primary_amine) or
             molecule.HasSubstructMatch(secondary_amine) or
             molecule.HasSubstructMatch(tertiary_amine))
        )
    except:
        return False

# Filter rows where the SMILES string corresponds to an amine and meets temperature/pka conditions
data['Is_Amine'] = data['SMILES'].apply(is_amine)
amine_data = data[
    (data['Is_Amine']) &
    (data['T'] == '25') &
    (data['pka_type'] == 'pKaH1')
].drop(columns=['Is_Amine'])  # Exclude the helper column

# Group by InChI and select the row with the highest pka_value
unique_amine_data = amine_data.loc[amine_data.groupby('InChI')['pka_value'].idxmax()]

# Save the filtered DataFrame to a CSV file
output_file = 'output.csv'
unique_amine_data.to_csv(output_file, index=False)

# Output the count of amine molecules and file location
print(f'The number of unique amine molecules is: {len(unique_amine_data)}')
print(f'The list of unique amines has been saved to: {output_file}')
