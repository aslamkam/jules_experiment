from rdkit import Chem

smiles = "C[n+]1c2ccccc2c(N)c2ccccc21.[Br-]"  # Ethanol as an example
mol = Chem.MolFromSmiles(smiles)

# Get unique atomic symbols
unique_atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}

print(unique_atoms)  # Output: {'C', 'O', 'H'}