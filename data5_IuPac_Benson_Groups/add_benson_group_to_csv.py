import pandas as pd
import re
from collections import defaultdict
from rmgpy.molecule import Molecule
from rmgpy.data.thermo import ThermoDatabase
from tqdm import tqdm  # <-- Progress bar

def load_thermo_db():
    thermo_db = ThermoDatabase()
    thermo_db.load(
        path='/rmg/RMG-database/input/thermo/',
        depository=False,
    )
    return thermo_db

def get_benson_vector(smiles, thermo_db):
    mol = Molecule().from_smiles(smiles)
    mol.generate_resonance_structures()
    thermo_data = thermo_db.estimate_thermo_via_group_additivity(mol)

    groups = re.findall(r'group\((.*?)\)', thermo_data.comment)
    group_counts = defaultdict(int)
    for group in groups:
        group_counts[group] += 1
    return group_counts

def safe_get_benson(smiles, thermo_db):
    try:
        group_counts = get_benson_vector(smiles, thermo_db)
        return str(group_counts)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

def main():
    thermo_db = load_thermo_db()
    input_excel = 'Filtered_IuPac.xlsx'
    df = pd.read_excel(input_excel)

    if 'Smiles' not in df.columns:
        raise KeyError("The Excel file must contain a 'Smiles' column.")

    tqdm.pandas(desc="Processing SMILES")
    df['benson_groups'] = df['Smiles'].progress_apply(lambda s: safe_get_benson(s, thermo_db))

    problematic = df['benson_groups'].isnull().sum()
    if problematic > 0:
        print(f"Removing {problematic} molecule(s) due to errors.")
        df = df[df['benson_groups'].notnull()]

    output_excel = 'Filtered_IuPac_benson.xlsx'
    df.to_excel(output_excel, index=False)
    print(f"Updated Excel file saved as {output_excel}")

if __name__ == '__main__':
    main()
