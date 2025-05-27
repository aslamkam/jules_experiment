import pandas as pd
import os
from pathlib import Path

def create_filtered_dataset(input_csv_path, sigma_profile_dir, output_path):
    """
    Create a filtered dataset containing only the amine entries that:
    1. Have corresponding sigma profiles (using ID column to match)
    2. Have pka_value >= 6.0
    """
    # Read the amine dataset
    print("Reading amine dataset...")
    print(os.getcwd())
    df = pd.read_csv(input_csv_path)
    
    # Get list of available sigma profile files
    print("Getting available sigma profiles...")
    sigma_files = set()
    for f in Path(sigma_profile_dir).glob("*.txt"):
        try:
            # Remove leading zeros and .txt extension to get the ID number
            sigma_files.add(int(f.stem.lstrip('0')))
        except Exception:
            print(f"Skipping file with invalid format: {f.name}")
    
    print(f"Found {len(sigma_files)} sigma profile files")
    print(f"Sample of sigma file numbers: {sorted(list(sigma_files))[:5]}")
    
    # Create mask for rows where:
    # 1. The ID matches a sigma profile
    # 2. pka_value is >= 6.0
    sigma_mask = df['ID'].isin(sigma_files)
    pka_mask = df['pka_value'] <= 7.0
    combined_mask = sigma_mask & pka_mask
    
    print(f"\nNumber of matching rows (with sigma profiles): {sigma_mask.sum()}")
    print(f"Number of rows with pKa >= 6.0: {pka_mask.sum()}")
    print(f"Number of rows meeting both criteria: {combined_mask.sum()}")
    
    # Filter the dataset
    print("\nFiltering dataset...")
    filtered_df = df[combined_mask].copy()
    
    # Save the filtered dataset if it's not empty
    if len(filtered_df) > 0:
        print("Saving filtered dataset...")
        filtered_df.to_csv(output_path, index=False)  # No need to keep the index
        print(f"Filtered dataset saved to: {output_path}")
    else:
        print("WARNING: Filtered dataset is empty!")
    
    print(f"\nSummary:")
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Rows excluded due to pKa < 6.0: {len(df[sigma_mask]) - len(filtered_df)}")
    
    # Print a few examples of matches for verification
    if len(filtered_df) > 0:
        print("\nSample of matches (first 5):")
        for _, row in filtered_df.head().iterrows():
            print(f"ID {row['ID']} matches sigma profile: {row['ID']:06d}.txt (pKa: {row['pka_value']:.2f})")

if __name__ == "__main__":
    # Define paths
    input_csv = "./amines-pka-dataset.csv"
    sigma_dir = "./SigmaProfileData/SigmaProfileData"
    output_file = "./available-amine-pka-dataset.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the dataset
    create_filtered_dataset(input_csv, sigma_dir, output_file)
