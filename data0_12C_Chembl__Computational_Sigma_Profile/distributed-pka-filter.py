import pandas as pd
import numpy as np
from pathlib import Path

def create_equally_distributed_dataset(input_csv_path, sigma_profile_dir, output_equal_path, output_remaining_path):
    """
    Create filtered datasets containing amine entries with:
    1. Corresponding sigma profiles (using ID column to match)
    2. pka_value >= 6.0
    3. Equal distribution of pKa values across the range
    
    Args:
        input_csv_path: Path to input CSV file
        sigma_profile_dir: Directory containing sigma profile files
        output_equal_path: Path for equally distributed dataset
        output_remaining_path: Path for remaining entries
    """
    # Read the amine dataset
    print("Reading amine dataset...")
    df = pd.read_csv(input_csv_path)
    
    # Get list of available sigma profile files
    print("Getting available sigma profiles...")
    sigma_files = set()
    for f in Path(sigma_profile_dir).glob("*.txt"):
        try:
            sigma_files.add(int(f.stem.lstrip('0')))
        except Exception:
            print(f"Skipping file with invalid format: {f.name}")
    
    print(f"Found {len(sigma_files)} sigma profile files")
    
    # Create initial masks
    sigma_mask = df['ID'].isin(sigma_files)
    pka_mask = df['pka_value'] >= 0.0
    combined_mask = sigma_mask & pka_mask
    
    # Get initially filtered dataset
    filtered_df = df[combined_mask].copy()
    
    if len(filtered_df) == 0:
        print("ERROR: No entries match the initial criteria!")
        return
    
    # Create equally distributed subset
    # Number of bins - adjust this value to control distribution granularity
    n_bins = 2
    
    # Calculate histogram
    hist, bin_edges = np.histogram(filtered_df['pka_value'], bins=n_bins)
    
    # Find the target number of samples per bin
    # We'll use the minimum number of samples in any bin to ensure even distribution
    target_per_bin = 500 # max(1, min(hist[hist > 0]) // 1)  # Using integer division
    print(f"\nTarget samples per pKa bin: {target_per_bin}")
    
    # Initialize empty DataFrame for equally distributed samples
    equal_dist_rows = []
    
    # For each bin, randomly select the target number of samples
    for i in range(len(bin_edges) - 1):
        bin_mask = (filtered_df['pka_value'] >= bin_edges[i]) & (filtered_df['pka_value'] < bin_edges[i+1])
        bin_data = filtered_df[bin_mask]
        
        if len(bin_data) > 0:
            # Randomly select samples from this bin
            selected = bin_data.sample(n=min(target_per_bin, len(bin_data)), random_state=42)
            equal_dist_rows.append(selected)
    
    # Combine all selected rows
    equal_dist_df = pd.concat(equal_dist_rows) if equal_dist_rows else pd.DataFrame()
    
    # Create remaining dataset (entries that didn't make it into the equal distribution)
    remaining_df = filtered_df[~filtered_df.index.isin(equal_dist_df.index)]
    
    # Save datasets
    print("\nSaving datasets...")
    equal_dist_df.to_csv(output_equal_path, index=False)
    remaining_df.to_csv(output_remaining_path, index=False)
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Original dataset size: {len(df)}")
    print(f"Initially filtered dataset size: {len(filtered_df)}")
    print(f"Equally distributed dataset size: {len(equal_dist_df)}")
    print(f"Remaining dataset size: {len(remaining_df)}")
    
    # Print pKa distribution statistics
    print("\npKa distribution statistics:")
    print("\nEqually distributed dataset:")
    print(equal_dist_df['pka_value'].describe())
    print("\nRemaining dataset:")
    print(remaining_df['pka_value'].describe())
    
    # Verify even distribution by printing bin counts
    print("\nBin counts in equally distributed dataset:")
    hist_equal, _ = np.histogram(equal_dist_df['pka_value'], bins=bin_edges)
    for i in range(len(hist_equal)):
        if hist_equal[i] > 0:
            print(f"Bin {i+1} ({bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}): {hist_equal[i]} samples")

if __name__ == "__main__":
    # Define paths
    input_csv = "./available-amine-pka-dataset.csv"
    sigma_dir = "./SigmaProfileData/SigmaProfileData"
    output_equal_file = "./available-amine-pka-equal-distribution-dataset.csv"
    output_remaining_file = "./available-amine-pka-remaining-dataset.csv"
    
    # Process the dataset
    create_equally_distributed_dataset(input_csv, sigma_dir, output_equal_file, output_remaining_file)
