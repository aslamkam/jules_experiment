import pandas as pd
import re
import signal
import json
from collections import defaultdict
from rmgpy.molecule import Molecule
from rmgpy.data.thermo import ThermoDatabase
from tqdm import tqdm

# Custom exception for timeouts
class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()

def load_thermo_db():
    thermo_db = ThermoDatabase()
    thermo_db.load(path='/rmg/RMG-database/input/thermo/', depository=False)
    return thermo_db

def safe_get_benson(smiles, thermo_db, timeout=60):
    if not hasattr(signal, 'SIGALRM'):
        return _compute_benson(smiles, thermo_db)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = _compute_benson(smiles, thermo_db)
        signal.alarm(0)
        return result
    except TimeoutException:
        raise TimeoutException(f"Timeout after {timeout}s for SMILES: {smiles}")
    except Exception as e:
        raise e

def _compute_benson(smiles, thermo_db):
    mol = Molecule().from_smiles(smiles)
    mol.generate_resonance_structures()
    thermo_data = thermo_db.estimate_thermo_via_group_additivity(mol)
    groups = re.findall(r'group\((.*?)\)', thermo_data.comment)
    group_counts = defaultdict(int)
    for group in groups:
        group_counts[group] += 1
    return dict(group_counts)

def main():
    thermo_db = load_thermo_db()
    input_file = 'ChEMBL_amines.xlsx'
    df = pd.read_excel(input_file)

    if 'Smiles' not in df.columns:
        raise KeyError("The Excel file must contain a 'Smiles' column.")

    error_log = {}
    timeout_60_log = []
    timeout_1800_log = []
    successful = {}

    print("Starting first pass with 60s timeout...")
    for smiles in tqdm(df['Smiles'], desc="First Pass"):
        try:
            result = safe_get_benson(smiles, thermo_db, timeout=60)
            successful[smiles] = result
        except TimeoutException:
            timeout_60_log.append(smiles)
        except Exception as e:
            error_log[smiles] = str(e)

    print("Starting second pass for 60s timeouts with 30-minute timeout...")
    for smiles in tqdm(timeout_60_log[:], desc="Second Pass"):
        try:
            result = safe_get_benson(smiles, thermo_db, timeout=1800)
            successful[smiles] = result
            timeout_60_log.remove(smiles)
        except TimeoutException:
            timeout_1800_log.append(smiles)
        except Exception as e:
            error_log[smiles] = str(e)

    # Create result DataFrame
    df['benson_groups'] = df['Smiles'].map(lambda s: successful.get(s))

    output_file = 'Amines_12C_CHEMBL_benson.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Save logs
    with open('error_log.json', 'w') as f:
        json.dump(error_log, f, indent=2)
    with open('timeout_60s_log.txt', 'w') as f:
        for s in timeout_60_log:
            f.write(s + '\n')
    with open('timeout_30min_log.txt', 'w') as f:
        for s in timeout_1800_log:
            f.write(s + '\n')

    print("Logs saved for errors and timeouts.")

if __name__ == '__main__':
    main()
