import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.nn import GINEConv, global_mean_pool

from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem


# -------------------------------
# Data Loading and Preprocessing
# -------------------------------


def load_sigma_profile(file_path):
    """Load sigma profile from file."""
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_pka_value(pka_str):
    """Convert pKa string to float; if a range is provided, use its average."""
    try:
        return float(pka_str)
    except (ValueError, TypeError):
        if isinstance(pka_str, str) and 'to' in pka_str:
            try:
                values = [float(x.strip()) for x in pka_str.split('to')]
                return sum(values) / len(values)
            except Exception:
                return None
        return None

def preprocess_data(dataset_path, sigma_profiles_path):
    """Load and preprocess dataset and sigma profile files."""
    # Load amines dataset
    amines_df = pd.read_csv(dataset_path)
    columns_to_keep = ['InChI', 'SMILES', 'pka_value', 'InChI_UID']
    amines_df = amines_df[columns_to_keep]
    amines_df['pka_value'] = amines_df['pka_value'].apply(process_pka_value)
    amines_df = amines_df.dropna(subset=['pka_value'])
    
    # Aggregate sigma profiles (even if not used in the GCN-only model, they may be useful to filter out bad entries)
    sigma_profiles = []
    valid_inchis = []
    for inchi_uid in amines_df['InChI_UID']:
        file_path = os.path.join(sigma_profiles_path, f'{inchi_uid}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            valid_inchis.append(inchi_uid)
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), columns=column_names)
    sigma_profiles_df['InChI_UID'] = valid_inchis
    
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='InChI_UID')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df

# ------------------------------------
# Molecular Graph Generation Functions
# ------------------------------------


def atom_to_feature_vector(atom, mol=None):
    hybridization_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4
    }
    idx = atom.GetIdx()
    gasteiger_charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0

    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        hybridization_map.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetMass(),
        gasteiger_charge,  # NEW FEATURE
    ]

def molecule_to_graph(smiles):
    """Convert a SMILES string into a PyTorch Geometric graph with Gasteiger charges."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Compute Gasteiger charges
        mol = Chem.AddHs(mol)  # Gasteiger requires explicit Hs
        AllChem.ComputeGasteigerCharges(mol)

        atom_features = [atom_to_feature_vector(atom, mol) for atom in mol.GetAtoms()]
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

class MolecularGCNModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=64):
        super(MolecularGCNModel, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2 (Residual connection)
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x + x_res)  # Residual
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Pooling
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out.squeeze()


# -------------------------------
# Training and Evaluation Functions
# -------------------------------

def train_epoch(model, optimizer, criterion, sigma_tensor, graph_batch, targets, device):
    model.train()
    optimizer.zero_grad()
    preds = model(graph_batch)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    return loss.item(), preds

def evaluate(model, criterion, graph_batch, targets, device):
    model.eval()
    with torch.no_grad():
        preds = model(graph_batch)
        loss = criterion(preds, targets)
    return loss.item(), preds

# -------------------------------
# Main Function for GCN Model
# -------------------------------

def main():
    # Update these paths as needed.
    dataset_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\amine_molecules_full_with_UID.csv'
    sigma_profiles_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\sigma_profiles'
    
    print("Loading and preprocessing data for GCN model...")
    merged_df = preprocess_data(dataset_path, sigma_profiles_path)
    
    # Prepare molecular graph data
    smiles_list = merged_df['SMILES'].values
    graphs, valid_indices = prepare_molecular_graphs(smiles_list)
    if not graphs:
        print("No valid molecular graphs were generated.")
        return
    
    # Filter targets using valid indices
    y = merged_df['pka_value'].values[valid_indices]
    
    # Split data into training and testing sets
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    train_batch = Batch.from_data_list(train_graphs)
    test_batch = Batch.from_data_list(test_graphs)
    
    y_train = torch.tensor(y[train_idx], dtype=torch.float)
    y_test = torch.tensor(y[test_idx], dtype=torch.float)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch = train_batch.to(device)
    test_batch = test_batch.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    node_feature_dim = train_graphs[0].x.shape[1]
    model = MolecularGCNModel(node_feature_dim, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    num_epochs = 2200
    train_losses = []
    test_losses = []
    
    print("Starting training for GCN model...")
    for epoch in range(num_epochs):
        train_loss, _ = train_epoch(model, optimizer, criterion, None, train_batch, y_train, device)
        test_loss, preds = evaluate(model, criterion, test_batch, y_test, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Calculate metrics on test set
    model.eval()
    with torch.no_grad():
        preds = model(test_batch).cpu().numpy()
        y_true = y_test.cpu().numpy()
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    
    print("Evaluation Metrics for GCN Model:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    
    # Plot loss curves and true vs predicted.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curve - GCN Model")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, preds, alpha=0.6)
    plt.xlabel("True pKa")
    plt.ylabel("Predicted pKa")
    plt.title("True vs Predicted pKa - GCN Model")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
