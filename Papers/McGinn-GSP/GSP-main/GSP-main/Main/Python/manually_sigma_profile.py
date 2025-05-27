# General
import pickle

# Specific
import numpy
import tensorflow
import spektral
from spektral import transforms
import networkx
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers as rdFF
from matplotlib import pyplot as plt

SMILES='c1ccc(cc1)C[C@@H](C(=O)O)N'

# Obtain RDKit mol object
molecule=Chem.MolFromSmiles(SMILES)
# Make hydrogens explicit
molecule=AllChem.AddHs(molecule)
# Generate initial 3D structure of the molecule
AllChem.EmbedMolecule(molecule)
# Initialize MMFF props object
prop=rdFF.MMFFGetMoleculeProperties(molecule)
# Check if MMFF exists for entry
if prop is None:
    raise ValueError('Cannot describe requested molecule using MMFF.')
else:
    # Initialize container
    atomTypes=[]
    # Retrieve node-level features
    for atom in molecule.GetAtoms():
        # Get MMFF atom type
        atomType=prop.GetMMFFAtomType(atom.GetIdx())
        # Append
        atomTypes.append(atomType)

# Define available unique MMFF atom types (F)
uniqueAtomTypes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,20,21,22,23,24,25,
                 26,27,28,29,30,31,32,33,35,37,38,39,40,42,43,44,45,59,61,63,
                 64,65,66,70,71,72,74]
# Initialize node-level feature matrix (X) of size NxF
X=numpy.zeros((len(atomTypes),len(uniqueAtomTypes)))
# Iterate over atomTypes
for n in range(len(atomTypes)):
    # Check that atom type it is available
    if atomTypes[n] not in uniqueAtomTypes:
        raise ValueError('Molecule contains atoms not trained on the GCN.')
    # Map force field atom type to index in uniqueAtomTypes
    oneHotIndex=uniqueAtomTypes.index(atomTypes[n])
    # One-hot encode atom n
    oneHot=spektral.utils.one_hot(oneHotIndex,len(uniqueAtomTypes))
    # Add entry to X
    X[n,:]=oneHot

class graphDataset(spektral.data.Dataset):
    """
    spektral.data.Dataset object containing the spektral graph objects of the
    molecules in any given dataset.
    """
    def __init__(self,graph):
        self.graph=graph
        super().__init__()
    def read(self):
        dataset=[self.graph]
        return dataset
# Get adjacency matrix
adjacencyMatrix=Chem.GetAdjacencyMatrix(molecule).astype('float32')
# Builg graph
graph=spektral.data.graph.Graph(x=X,a=adjacencyMatrix,e=None,y=numpy.ones((51,)))
# Convert to dataset
graphSet=graphDataset(graph)

#---- Define GCN model
class GCN_Model_SP(tensorflow.keras.models.Model):
    """
    tensorflow.keras.models.Model object containing the architecture of the
    graph neural network model for sigma profile regression.
    """
    def __init__(self,architecture):
        """
        __init__() constrcuts the architecture of the model.

        Parameters
        ----------
        architecture : dict
            See configuration section.

        Returns
        -------
        None.

        """
        super().__init__()
        # Unpack architecture
        conv1_channels=architecture.get('conv1_channels')
        conv2_channels=architecture.get('conv2_channels')
        conv3_channels=architecture.get('conv3_channels')
        reg=tensorflow.keras.regularizers.L2(architecture.get('L2 coeff.'))
        ki='he_uniform'
        # Define userLayers list
        self.userLayers=[]
        # First conv layer
        if conv1_channels>0:
            conv1Layer=spektral.layers.GCNConv(conv1_channels,
                                               activation='relu',
                                               kernel_initializer=ki,
                                               kernel_regularizer=reg,
                                               use_bias=False)
            self.userLayers.append(conv1Layer)
        # Second conv layer
        if conv2_channels>0:
            conv2Layer=spektral.layers.GCNConv(conv2_channels,
                                               activation='relu',
                                               kernel_initializer=ki,
                                               kernel_regularizer=reg,
                                               use_bias=False)
            self.userLayers.append(conv2Layer)
        # Third conv layer
        if conv3_channels>0:
            conv3Layer=spektral.layers.GCNConv(conv3_channels,
                                               activation='relu',
                                               kernel_initializer=ki,
                                               kernel_regularizer=reg,
                                               use_bias=False)
            self.userLayers.append(conv3Layer)
        # Dense layer (X*W)
        dense=tensorflow.keras.layers.Dense(51,
                                            activation='relu',
                                            kernel_initializer=ki,
                                            kernel_regularizer=reg,
                                            use_bias=False)
        self.userLayers.append(dense)
        # Pooling layer
        poolLayer=spektral.layers.GlobalSumPool()
        self.userLayers.append(poolLayer)
    def call(self, inputs):
        """
        call() propagates "inputs" through GCN model.

        Parameters
        ----------
        inputs : tuple
            Tuple containing the adjacency tensor (batch,N,N) and the feature
            tensor (batch,N,F).

        Returns
        -------
        x : tf.Tensor
            Tensor containg the predicted sigma profiles (batch,1,51).

        """
        # Extract node feature vector (x) and adjacency matrix (a) from inputs
        X,A=inputs
        # Conv layers:
        for n in range(len(self.userLayers)-2):
            X=self.userLayers[n]([X,A])
        # Dense layer (X*W)
        X=self.userLayers[-2](X)
        # Pooling layer
        X=self.userLayers[-1](X)
        # Output
        return X

# Path to Model
modelPath=r'C:\Users\kamal\OneDrive - University of Guelph\My Research\IUPAC\GSP-main\GSP-main\Main\Models\MMFF_GCN.pkl'
# Load weights
with open(modelPath,'rb') as f:
      weights=pickle.load(f)
# Define architecture
architecture={'conv1_channels': weights[0].shape[1],
              'conv2_channels': weights[1].shape[1],
              'conv3_channels': weights[2].shape[1]}
# Build model
GCN=GCN_Model_SP(architecture)
# Compile model
GCN.compile()
# Create loader
loader=spektral.data.BatchLoader(graphSet,shuffle=False)
# Set model shape
GCN.fit(loader.load(),steps_per_epoch=loader.steps_per_epoch,epochs=1)
# Set weights
GCN.set_weights(weights)

# Apply filters to adjacency matrices
graphSet.apply(transforms.GCNFilter())
# Predict sigma profile
loader=spektral.data.BatchLoader(graphSet,shuffle=False)
predSP=GCN.predict(loader.load(),steps=loader.steps_per_epoch)

sigma=numpy.linspace(-0.025,0.025,51)

# Define the file path for saving the sigma profile
output_file_path = r'.\sigma_profile.txt'

# Save the sigma profile to the text file
with open(output_file_path, 'w') as file:
    for sigma_value, profile_value in zip(sigma, predSP[0, :]):
        file.write(f"{sigma_value:.6f}\t{profile_value:.6f}\n")

print(f"Sigma profile saved to {output_file_path}")