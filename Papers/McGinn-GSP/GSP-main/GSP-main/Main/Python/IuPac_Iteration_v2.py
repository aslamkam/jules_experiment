import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

import tensorflow as tf
from spektral import transforms
from spektral.data import Dataset, Graph, BatchLoader
from spektral.layers import GCNConv, GlobalSumPool
from rdkit import Chem
from rdkit.Chem import AllChem

# Constants
MODEL_PATH = "../Models/MMFF_GCN.pkl"
CSV_PATH = "./amine_molecules_full_with_UID.csv"
OUTPUT_DIR = "sigma_profiles"
SIGMA_RANGE = (-0.025, 0.025, 51)
UNIQUE_ATOM_TYPES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45,
    59, 61, 63, 64, 65, 66, 70, 71, 72, 74
]

# Configure logging
log_file = "error_log.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


class GCNModel(tf.keras.Model):
    """Graph Convolutional Network model for sigma profile prediction."""

    def __init__(self, conv_channels: List[int], l2_coeff: float = 0.01):
        super().__init__()
        self.layers_stack = []
        reg = tf.keras.regularizers.L2(l2_coeff)
        initializer = 'he_uniform'

        # Add GCN layers
        for channels in filter(None, conv_channels):
            self.layers_stack.append(
                GCNConv(channels, activation='relu',
                        kernel_initializer=initializer,
                        kernel_regularizer=reg,
                        use_bias=False)
            )

        # Add final dense and pooling layers
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
    """Custom dataset for handling single molecule graphs."""

    def __init__(self, graph: Graph, **kwargs):
        self.graph = graph
        super().__init__(**kwargs)

    def read(self) -> List[Graph]:
        return [self.graph]


def load_gcn_model() -> GCNModel:
    """Load and initialize the GCN model with pretrained weights."""
    with open(MODEL_PATH, 'rb') as f:
        weights = pickle.load(f)

    architecture = [w.shape[1] for w in weights[:3]]  # First 3 weights are GCN layers
    model = GCNModel(architecture)

    # Build model with dynamic input shapes
    input_x_shape = (None, weights[0].shape[0])  # (nodes, features)
    input_a_shape = (None, None)  # (nodes, nodes)
    model.build([input_x_shape, input_a_shape])  # Correct input shapes
    model.set_weights(weights)

    return model


def process_molecule(smiles: str) -> Optional[Chem.Mol]:
    """Process SMILES string into a 3D molecular structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) == -1:
        raise RuntimeError("Failed to generate 3D conformation")

    return mol


def create_graph_features(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create node features, adjacency matrix, and dummy y tensor."""
    prop = AllChem.MMFFGetMoleculeProperties(mol)
    if prop is None:
        raise ValueError("Molecule not compatible with MMFF force field")

    atom_types = [prop.GetMMFFAtomType(atom.GetIdx()) for atom in mol.GetAtoms()]

    # Create one-hot encoded features
    x = np.zeros((len(atom_types), len(UNIQUE_ATOM_TYPES)))
    for i, a_type in enumerate(atom_types):
        if a_type not in UNIQUE_ATOM_TYPES:
            raise ValueError(f"Unsupported atom type: {a_type}")
        x[i, UNIQUE_ATOM_TYPES.index(a_type)] = 1

    # Create adjacency matrix
    adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)

    # Create dummy y tensor
    y = np.zeros((len(atom_types), 1))  # Dummy y tensor for compatibility

    return x, adj, y


def predict_sigma_profile(model: GCNModel, graph: Graph) -> np.ndarray:
    """Predict sigma profile for a molecular graph."""
    dataset = MoleculeDataset(graph)
    dataset.apply(transforms.GCNFilter())

    loader = BatchLoader(dataset, batch_size=1, shuffle=False)
    return model.predict(loader.load(), steps=loader.steps_per_epoch)


def save_sigma_profile(prediction: np.ndarray, InChI_UID: str) -> None:
    """Save sigma profile to text file."""
    sigma_values = np.linspace(*SIGMA_RANGE)
    filename = os.path.join(OUTPUT_DIR, f"{InChI_UID}.txt")

    np.savetxt(filename,
               np.column_stack([sigma_values, prediction.flatten()]),
               delimiter='\t',
               fmt='%.6f')
    print(f"Saved sigma profile: {filename}")


def main():
    # Initialize model and output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = load_gcn_model()
    data = pd.read_csv(CSV_PATH)

    for index, row in data.iterrows():
        try:
            # Process molecule
            mol = process_molecule(row['SMILES'])

            # Create graph representation
            x, adj, y = create_graph_features(mol)
            graph = Graph(x=x, a=adj, y=y)  # Pass the dummy y tensor

            # Make prediction
            prediction = predict_sigma_profile(model, graph)

            # Save results
            save_sigma_profile(prediction[0], row['InChI_UID'])

        except Exception as e:
            logger.error(f"Error processing row {index} ({row['InChI_UID']}): {str(e)}", exc_info=True)
            continue


if __name__ == "__main__":
    main()