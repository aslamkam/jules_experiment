import os
import pickle
import pandas as pd
import numpy as np
import spektral
from spektral import transforms
from rdkit import Chem
from rdkit.Chem import AllChem
from spektral.data import BatchLoader
from spektral.data.graph import Graph
import tensorflow as tf

# Define the GCN model
class GCN_Model_SP(tf.keras.models.Model):
    def __init__(self, architecture):
        super().__init__()
        conv1_channels = architecture.get('conv1_channels')
        conv2_channels = architecture.get('conv2_channels')
        conv3_channels = architecture.get('conv3_channels')
        reg = tf.keras.regularizers.L2(architecture.get('L2 coeff.', 0.01))
        ki = 'he_uniform'

        self.userLayers = []
        if conv1_channels > 0:
            self.userLayers.append(spektral.layers.GCNConv(conv1_channels, activation='relu', kernel_initializer=ki, kernel_regularizer=reg, use_bias=False))
        if conv2_channels > 0:
            self.userLayers.append(spektral.layers.GCNConv(conv2_channels, activation='relu', kernel_initializer=ki, kernel_regularizer=reg, use_bias=False))
        if conv3_channels > 0:
            self.userLayers.append(spektral.layers.GCNConv(conv3_channels, activation='relu', kernel_initializer=ki, kernel_regularizer=reg, use_bias=False))
        self.userLayers.append(tf.keras.layers.Dense(51, activation='relu', kernel_initializer=ki, kernel_regularizer=reg, use_bias=False))
        self.userLayers.append(spektral.layers.GlobalSumPool())

    def call(self, inputs):
        X, A = inputs
        for layer in self.userLayers[:-2]:
            X = layer([X, A])
        X = self.userLayers[-2](X)
        X = self.userLayers[-1](X)
        return X

# Load the model weights
model_path = r"../Models/MMFF_GCN.pkl"
with open(model_path, 'rb') as f:
    weights = pickle.load(f)
architecture = {
    'conv1_channels': weights[0].shape[1],
    'conv2_channels': weights[1].shape[1],
    'conv3_channels': weights[2].shape[1]
}

GCN = GCN_Model_SP(architecture)
GCN.compile()

# Build the model by calling it with some input data
dummy_X = np.zeros((1, len(weights[0])))
dummy_A = np.zeros((1, 1))
GCN([dummy_X, dummy_A])

GCN.set_weights(weights)

# Create the output directory
output_dir = "sigma_profiles"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
csv_path = "./amine_molecules.csv"
data = pd.read_csv(csv_path)

# Process each SMILES string
for index, row in data.iterrows():
    smiles = row['SMILES']
    inchi = row['InChI']
    
    try:
        molecule = Chem.MolFromSmiles(smiles)
        molecule = AllChem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)

        prop = AllChem.MMFFGetMoleculeProperties(molecule)
        if prop is None:
            raise ValueError(f"Cannot describe molecule {smiles} using MMFF.")

        atomTypes = [prop.GetMMFFAtomType(atom.GetIdx()) for atom in molecule.GetAtoms()]
        uniqueAtomTypes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,20,21,22,23,24,25,
                           26,27,28,29,30,31,32,33,35,37,38,39,40,42,43,44,45,59,61,63,
                           64,65,66,70,71,72,74]

        X = np.zeros((len(atomTypes), len(uniqueAtomTypes)))
        for n, atomType in enumerate(atomTypes):
            if atomType not in uniqueAtomTypes:
                raise ValueError(f"Molecule {smiles} contains unsupported atom types.")
            X[n, :] = spektral.utils.one_hot(uniqueAtomTypes.index(atomType), len(uniqueAtomTypes))

        adjacencyMatrix = Chem.GetAdjacencyMatrix(molecule).astype('float32')
        graph = Graph(x=X, a=adjacencyMatrix, e=None, y=np.ones((51,)))
        class MyDataset(spektral.data.Dataset):
            def read(self):
                return [graph]

        graphSet = MyDataset()
        graphSet.apply(transforms.GCNFilter())

        loader = BatchLoader(graphSet, shuffle=False)
        predSP = GCN.predict(loader.load(), steps=loader.steps_per_epoch)

        sigma = np.linspace(-0.025, 0.025, 51)
        output_file_path = os.path.join(output_dir, f"{inchi.replace('/', '_')}_sigma_profile.txt")

        with open(output_file_path, 'w') as file:
            for sigma_value, profile_value in zip(sigma, predSP[0, :].flatten()):
                file.write(f"{sigma_value:.6f}\t{float(profile_value):.6f}\n")

        print(f"Sigma profile for {inchi} saved to {output_file_path}")

    except Exception as e:
        import traceback
        print(f"Error processing molecule {smiles}: {e}")
        traceback.print_exc()
