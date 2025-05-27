import os
import csv
import re

def get_second_column(file_path):
    """
    Extract the entire second column from a file as floats
    
    Args:
    file_path (str): Path to the file
    
    Returns:
    list: List of float values in the second column
    """
    second_column = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Split the line by tab and get the second column
                parts = line.strip().split('\t')
                if len(parts) > 1:
                    try:
                        # Convert to float
                        second_column.append(float(parts[1]))
                    except ValueError:
                        print(f"Error converting {parts[1]} to float in {file_path}")
                        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return second_column

def find_matching_files(original_dir, molecules_dir):
    """
    Find files in molecules_dir that have identical second column to files in original_dir
    
    Args:
    original_dir (str): Path to the directory with original files
    molecules_dir (str): Path to the directory with molecule files
    
    Returns:
    list: List of tuples containing matching (molecule_filename, original_filename)
    """
    # Get list of files in both directories
    # Filter original_files to only include numerical filenames
    original_files = [f for f in os.listdir(original_dir) 
                      if re.match(r'^\d+\.txt$', f)]
    molecule_files = os.listdir(molecules_dir)
    
    # List to store matching files
    matching_files = []
    
    # Iterate through molecule files
    for mol_file in molecule_files:
        mol_path = os.path.join(molecules_dir, mol_file)
        mol_second_column = get_second_column(mol_path)
        
        # Check against original files
        for orig_file in original_files:
            orig_path = os.path.join(original_dir, orig_file)
            orig_second_column = get_second_column(orig_path)
            
            # If entire second columns match (as floats), add to matching files
            if mol_second_column == orig_second_column:
                matching_files.append((mol_file, orig_file))
                break  # Stop checking once a match is found
    
    return matching_files

def write_matching_files_to_csv(matching_files, output_file='matching_files.csv'):
    """
    Write matching files to a CSV
    
    Args:
    matching_files (list): List of tuples with matching filenames
    output_file (str): Path to output CSV file
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['MoleculeFileName', 'OriginalFileName'])
        # Write matching files
        csv_writer.writerows(matching_files)
    
    print(f"Matching files written to {output_file}")

def main():
    # Specify the directories
    original_dir = r'.\SigmaProfileData\SigmaProfileData'
    molecules_dir = r'.\SigmaProfileData\96Molecules'
    
    # Find matching files
    matching_files = find_matching_files(original_dir, molecules_dir)
    
    # Print matching files
    print("Matching Files:")
    for mol_file, orig_file in matching_files:
        print(f"{mol_file} matches {orig_file}")
    
    # Write to CSV
    write_matching_files_to_csv(matching_files)

if __name__ == '__main__':
    main()