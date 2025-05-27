import pandas as pd
import hashlib
import argparse
from pathlib import Path
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


def generate_uid_from_inchi(inchi):
    """
    Generate a unique identifier from an InChI string using SHA-256 hash.
    Returns the first 8 characters of the hash.
    """
    if pd.isna(inchi):
        return None
    
    # Convert InChI to bytes and generate hash
    hash_object = hashlib.sha256(str(inchi).encode())
    # Get first 8 characters of the hash
    return hash_object.hexdigest()[:8]

def add_inchi_uid(input_file, output_file=None):
    """
    Add InChI-based UID column to the CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, will modify the input filename
    
    Returns:
        str: Path to the output file
    """
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")
    
    # Check if InChI column exists
    if 'InChI' not in df.columns:
        raise ValueError("CSV file must contain an 'InChI' column")
    
    # Generate new column name
    new_col_name = 'InChI_UID'
    counter = 1
    while new_col_name in df.columns:
        new_col_name = f'InChI_UID_{counter}'
        counter += 1
    
    # Generate UIDs
    df[new_col_name] = df['InChI'].apply(generate_uid_from_inchi)
    
    # Determine output file path
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_uid{input_path.suffix}")
    
    # Save the modified DataFrame
    df.to_csv(output_file, index=False)
    return output_file

def main():
    
    try:
        output_path = add_inchi_uid('amine_molecules_full.csv', 'amine_molecules_full_with_UID.csv')
        print(f"Successfully processed file. Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()