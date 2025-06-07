import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

from rdkit import Chem
from rdkit.Chem import rdchem

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_sigma_profile(file_path):
    """Load sigma profile from file."""
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        # Return only the sigma values (i.e., column index 1)
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def preprocess_data(dataset_path, sigma_profiles_path):
    """Load and preprocess dataset and sigma profile files."""
    # Load amines dataset
    amines_df = pd.read_csv(dataset_path)
    # Keep only the columns used in the RandomForests file
    columns_to_keep = ['ChEMBL ID', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'Smiles', 'Inchi Key']
    amines_df = amines_df[columns_to_keep]
    
    # Aggregate sigma profiles using the 'ID' column.
    sigma_profiles = []
    ids_with_profiles = []
    for molecule_chembl_id, inchi_key in amines_df[['ChEMBL ID', 'Inchi Key']].values:
        # Use the same file naming format as in the RandomForests file.
        file_path = os.path.join(sigma_profiles_path, f'{inchi_key}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            ids_with_profiles.append(molecule_chembl_id)
    
    if len(sigma_profiles) == 0:
        raise ValueError("No valid sigma profiles were loaded.")
    
    # Create a dataframe for sigma profiles
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), columns=column_names)
    sigma_profiles_df['ChEMBL ID'] = ids_with_profiles

    # Merge the sigma profile data with the amines dataset (using 'ID' as the key)
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='ChEMBL ID')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Rename the 'smiles' column to 'SMILES' for compatibility with the graph generation functions.
    merged_df.rename(columns={'Smiles': 'SMILES'}, inplace=True)
    
    return merged_df, column_names

# ------------------------------------
# Molecular Graph Generation Functions
# ------------------------------------

def atom_to_feature_vector(atom):
    hybridization_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4
    }
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        hybridization_map.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetMass(),
    ]

def molecule_to_graph(smiles):
    """Convert a SMILES string into a PyTorch Geometric graph."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        print(f"Error converting SMILES to graph: {e}")
        return None

def prepare_molecular_graphs(smiles_list):
    """Convert a list of SMILES strings to molecular graphs."""
    graphs = []
    valid_indices = []
    for idx, smiles in enumerate(smiles_list):
        graph = molecule_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
            valid_indices.append(idx)
    return graphs, valid_indices

# -------------------------------
# Fusion Model Definition
# -------------------------------
class FusionModel(nn.Module):
    def __init__(self, node_feature_dim, sigma_input_dim, hidden_dim=64):
        super(FusionModel, self).__init__()
        # GCN branch remains unchanged
        self.gcn_conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.gcn_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        # Updated Feedforward branch for sigma profiles with increased complexity
        self.ff_fc1 = nn.Linear(sigma_input_dim, hidden_dim * 4)
        self.ff_bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.ff_fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.ff_bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.ff_fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Fusion layer - combining both branch embeddings (concatenated)
        self.fusion_fc = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward_gcn(self, data):
        x, edge_index = data.x, data.edge_index
        # Determine batch indices if using a batched graph
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GCN layers with dropout and residual connection
        x = self.gcn_conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x_res = x
        x = self.gcn_conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x + x_res)
        x = self.dropout(x)
        
        x = self.gcn_conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        return x
    
    def forward_ff(self, sigma_input):
        # Feedforward branch with three layers, batch normalization, and dropout
        x = self.ff_fc1(sigma_input)
        x = self.ff_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.ff_fc2(x)
        x = self.ff_bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.ff_fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x
    
    def forward(self, graph_data, sigma_data):
        # Obtain embeddings from both branches
        gcn_embedding = self.forward_gcn(graph_data)         # shape: (batch_size, hidden_dim)
        ff_embedding = self.forward_ff(sigma_data)             # shape: (batch_size, hidden_dim)
        # Concatenate features from both branches
        fused = torch.cat([gcn_embedding, ff_embedding], dim=1)  # shape: (batch_size, 2*hidden_dim)
        out = self.fusion_fc(fused)  # final regression output
        return out.squeeze()

# -------------------------------
# Training and Evaluation Functions
# -------------------------------

def train_epoch(model, optimizer, criterion, graph_batch, sigma_batch, targets, device):
    model.train()
    optimizer.zero_grad()
    preds = model(graph_batch, sigma_batch)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item(), preds

def evaluate(model, criterion, graph_batch, sigma_batch, targets, device):
    model.eval()
    with torch.no_grad():
        preds = model(graph_batch, sigma_batch)
        loss = criterion(preds, targets)
    return loss.item(), preds

# -------------------------------
# Main Function for Fusion Model
# -------------------------------

def main():
    # Update file paths to use the same files as in the RandomForests file.
    dataset_path = r"C:\Users\kamal\jules_experiment\Features\Chembl-12C\ChEMBL_amines_12C.csv"
    sigma_profiles_path = r"C:\Users\kamal\jules_experiment\Features\Chembl-12C\Orca-Sigma-Profile\ChEMBL_12C_SigmaProfiles_Orca-5899"
    
    print("Loading and preprocessing data for Fusion Model...")
    merged_df, sigma_columns = preprocess_data(dataset_path, sigma_profiles_path)
    
    # Prepare graph data from SMILES.
    # Now using the renamed 'SMILES' column (converted from 'smiles').
    smiles_list = merged_df['SMILES'].values
    graphs, valid_indices = prepare_molecular_graphs(smiles_list)
    if not graphs:
        print("No valid molecular graphs were generated.")
        return
    
    # Filter targets and sigma profiles using valid indices.
    y = merged_df['CX Basic pKa'].values[valid_indices]
    sigma_data = merged_df[sigma_columns].values[valid_indices]
    
    # Standardize sigma profile features.
    scaler = StandardScaler()
    sigma_data_scaled = scaler.fit_transform(sigma_data)
    
    # Create train/test split.
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    train_graph_batch = Batch.from_data_list(train_graphs)
    test_graph_batch = Batch.from_data_list(test_graphs)
    
    sigma_train = torch.tensor(sigma_data_scaled[train_idx], dtype=torch.float)
    sigma_test = torch.tensor(sigma_data_scaled[test_idx], dtype=torch.float)
    
    y_train = torch.tensor(y[train_idx], dtype=torch.float)
    y_test = torch.tensor(y[test_idx], dtype=torch.float)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_graph_batch = train_graph_batch.to(device)
    test_graph_batch = test_graph_batch.to(device)
    sigma_train = sigma_train.to(device)
    sigma_test = sigma_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    # Set dimensions for the two branches.
    node_feature_dim = train_graphs[0].x.shape[1]
    sigma_input_dim = sigma_train.shape[1]
    
    model = FusionModel(node_feature_dim, sigma_input_dim, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    num_epochs = 10
    train_losses = []
    test_losses = []
    
    print("Starting training for Fusion Model...")
    for epoch in range(num_epochs):
        train_loss, _ = train_epoch(model, optimizer, criterion, train_graph_batch, sigma_train, y_train, device)
        test_loss, preds = evaluate(model, criterion, test_graph_batch, sigma_test, y_test, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Calculate metrics on test set.
    model.eval()
    with torch.no_grad():
        preds = model(test_graph_batch, sigma_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    
    print("Evaluation Metrics for Fusion Model:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    
    # Plot loss curves and true vs predicted scatter plot.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curve - Fusion Model")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, preds, alpha=0.6)
    plt.xlabel("True pKa")
    plt.ylabel("Predicted pKa")
    plt.title("True vs Predicted pKa - Fusion Model")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
