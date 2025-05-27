import pandas as pd

def match_molecules_smiles():
    # Read the first CSV file (96Molecules.csv)
    # Use no header and specify column names
    df1 = pd.read_csv('96Molecules.csv', header=None, names=['Name', 'Formula', 'ThirdColumn', 'ID'])

    # Read the second CSV file (truncated_amines-pka-dataset.csv)
    df2 = pd.read_csv('amines-pka-dataset.csv')

    # Merge the dataframes based on the ThirdColumn from df1 and smiles from df2
    merged_df = pd.merge(df1, df2, left_on='ThirdColumn', right_on='smiles', how='inner')

    # Include the 'Name' column from df1 in the output
    output_columns = list(set(df2.columns).union({'Name'}))
    final_df = merged_df[output_columns]

    final_df = final_df.loc[~final_df.eq('chembl').any(axis=1)]

    # Save the matched results to a new CSV file
    final_df.to_csv('matched_molecules_smiles_full.csv', index=False)

    # Print some information about the matching
    print(f"Total rows in 96Molecules.csv: {len(df1)}")
    print(f"Total rows in truncated_amines-pka-dataset.csv: {len(df2)}")
    print(f"Number of matched rows: {len(final_df)}")

    # Display the matched rows
    print("\nMatched Rows:")
    print(final_df)

# Run the matching function
match_molecules_smiles()
