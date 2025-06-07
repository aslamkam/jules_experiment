import os
import pickle
import logging
import pandas as pd
import numpy as np
import shutil
from typing import List, Tuple, Optional

import tensorflow as tf
from spektral import transforms
from spektral.data import Dataset, Graph, BatchLoader
from spektral.layers import GCNConv, GlobalSumPool
from rdkit import Chem
from rdkit.Chem import AllChem

# Constants
MODEL_PATH = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Papers\McGinn-GSP\GSP-main\GSP-main\Main\Models\MMFF_GCN.pkl"
SIGMA_RANGE = (-0.025, 0.025, 51)
UNIQUE_ATOM_TYPES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35,
    37, 38, 39, 40, 42, 43, 44, 45, 59, 61, 63, 64, 65, 66, 70,
    71, 72, 74
]

# Configure logging
log_file = "error_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class GCNModel(tf.keras.Model):
    def __init__(self, conv_channels: List[int], l2_coeff: float = 0.01):
        super().__init__()
        self.layers_stack = []
        reg = tf.keras.regularizers.L2(l2_coeff)
        initializer = 'he_uniform'

        for channels in filter(None, conv_channels):
            self.layers_stack.append(
                GCNConv(channels, activation='relu',
                        kernel_initializer=initializer,
                        kernel_regularizer=reg,
                        use_bias=False)
            )

        self.layers_stack.extend([
            tf.keras.layers.Dense(51, activation='relu',
                                  kernel_initializer=initializer,
                                  kernel_regularizer=reg,
                                  use_bias=False),
            GlobalSumPool()
        ])

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x, a = inputs
        for layer in self.layers_stack[:-2]:
            x = layer([x, a])
        x = self.layers_stack[-2](x)
        return self.layers_stack[-1](x)


class MoleculeDataset(Dataset):
    def __init__(self, graph: Graph, **kwargs):
        self.graph = graph
        super().__init__(**kwargs)

    def read(self):
        return [self.graph]


def load_gcn_model() -> GCNModel:
    with open(MODEL_PATH, 'rb') as f:
        weights = pickle.load(f)

    architecture = [w.shape[1] for w in weights[:3]]
    model = GCNModel(architecture)

    input_x_shape = (None, weights[0].shape[0])
    input_a_shape = (None, None)
    model.build([input_x_shape, input_a_shape])
    model.set_weights(weights)

    return model


def process_molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) == -1:
        raise RuntimeError("Failed to generate 3D conformation")
    return mol


def create_graph_features(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prop = AllChem.MMFFGetMoleculeProperties(mol)
    if prop is None:
        raise ValueError("Molecule not compatible with MMFF force field")

    atom_types = [prop.GetMMFFAtomType(atom.GetIdx()) for atom in mol.GetAtoms()]
    x = np.zeros((len(atom_types), len(UNIQUE_ATOM_TYPES)))
    for i, a_type in enumerate(atom_types):
        if a_type not in UNIQUE_ATOM_TYPES:
            raise ValueError(f"Unsupported atom type: {a_type}")
        x[i, UNIQUE_ATOM_TYPES.index(a_type)] = 1

    adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
    y = np.zeros((len(atom_types), 1))
    return x, adj, y


def predict_sigma_profile(model: GCNModel, graph: Graph) -> np.ndarray:
    ds = MoleculeDataset(graph)
    ds.apply(transforms.GCNFilter())
    loader = BatchLoader(ds, batch_size=1, shuffle=False)
    return model.predict(loader.load(), steps=loader.steps_per_epoch)


def main():
    # Input file and copy
    original_path = r"C:\Users\kamal\Dropbox\0000-ML_Databases\ChEMBL Data (Do NOT Edit)\ChEMBL_amines_12C.csv"
    copy_path = os.path.join(os.getcwd(), "ChEMBL_amines_12C_copy.csv")
    shutil.copyfile(original_path, copy_path)

    # Load model
    model = load_gcn_model()

    # Read data copy
    df = pd.read_csv(copy_path)
    df["Sigma Profile"] = None
    charge_values = np.linspace(-0.025, 0.025, 51)

    sigma_profiles_output_dir = "./ChEMBL_12C_SigmaProfiles_Mcginn-5899"

    for idx, row in df.iterrows():
        try:
            mol = process_molecule(row['Smiles'])
            x, adj, y = create_graph_features(mol)
            graph = Graph(x=x, a=adj, y=y)
            prediction = predict_sigma_profile(model, graph)[0].flatten()

            # Ensure the output directory exists
            os.makedirs(sigma_profiles_output_dir, exist_ok=True)

            # Get InChI Key for filename
            inchi_key = row['Inchi Key']
            # It's good practice to sanitize filenames, though InChIKeys are generally safe.
            # For this task, we'll assume they are safe.
            filename = f"{inchi_key}.txt"

            file_path = os.path.join(sigma_profiles_output_dir, filename)

            # Combine charge values and sigma profile prediction
            output_data = np.column_stack((charge_values, prediction))

            # Write the combined data to the text file, space-delimited
            np.savetxt(file_path, output_data, fmt='%s', delimiter=' ')

            df.at[idx, "Sigma Profile"] = prediction.tolist()
        except Exception as e:
            logger.error(f"Error at index {idx}, SMILES={row['Smiles']}: {e}")
            df.at[idx, "Sigma Profile"] = np.nan

    # Save updated dataframe
    output_path = os.path.join(os.getcwd(), "ChEMBL_amines_12C_with_sigma.csv")
    df.to_csv(output_path, index=False)
    print(f"Updated data saved to {output_path}")

if __name__ == "__main__":
    main()
