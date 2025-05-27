import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


# --- Data Loading and Preprocessing Functions ---

def load_sigma_profile(file_path):
    """Load sigma profile from file."""
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        # Return only the sigma profile values (column index 1)
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_pka_value(pka_str):
    """Convert pKa string to float, handling ranges by taking their average."""
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
    """Load and preprocess data."""
    # Load the amines pKa dataset
    amines_df = pd.read_csv(dataset_path)
    
    # Keep only necessary columns
    columns_to_keep = ['InChI', 'SMILES', 'pka_value', 'InChI_UID']
    amines_df = amines_df[columns_to_keep]
    amines_df['pka_value'] = amines_df['pka_value'].apply(process_pka_value)
    amines_df = amines_df.dropna(subset=['pka_value'])
    
    # Aggregate sigma profile data
    sigma_profiles = []
    inchis_with_profiles = []
    
    for inchi_uid in amines_df['InChI_UID']:
        file_path = os.path.join(sigma_profiles_path, f'{inchi_uid}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            inchis_with_profiles.append(inchi_uid)
    
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), columns=column_names)
    sigma_profiles_df['InChI_UID'] = inchis_with_profiles
    
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='InChI_UID')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df, column_names


# --- Molecular Graph Generation Functions ---

def molecule_to_graph(smiles):
    """Convert a SMILES string to a molecular graph for PyTorch Geometric."""
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get atomic features for each atom in the molecule
        atom_features = []
        for atom in mol.GetAtoms():
            # Basic atom features: atomic number, formal charge, degree, implicit hydrogen count,
            # aromaticity, and a numerical representation for hybridization state.
            features = [
                atom.GetAtomicNum(),                  # Atomic number
                atom.GetFormalCharge(),               # Formal charge
                atom.GetDegree(),                     # Number of bonded neighbors
                atom.GetNumImplicitHs(),              # Implicit hydrogen count
                int(atom.GetIsAromatic()),            # Is atom aromatic (0 or 1)
                float(atom.GetHybridization())        # Hybridization state as a float value
            ]
            atom_features.append(features)
        
        # Convert list of features to a torch tensor
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Create bond (edge) connections
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # Add edges in both directions
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        # If there are no bonds, create an empty edge index
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    except Exception as e:
        print(f"Error converting SMILES to graph: {str(e)}")
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


# --- Model Definition ---

class SigmaProfileEncoder(nn.Module):
    """Neural network to encode sigma profiles."""
    def __init__(self, input_dim, hidden_dim=64):
        super(SigmaProfileEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class MolecularGCN(nn.Module):
    """GCN model for processing molecular graphs."""
    def __init__(self, node_feature_dim, hidden_dim=64):
        super(MolecularGCN, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling to get graph-level representation
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        x = global_mean_pool(x, batch)
        
        return x


class HybridModel(nn.Module):
    """Hybrid model combining sigma profile data and molecular structure data."""
    def __init__(self, sigma_feature_dim, node_feature_dim, hidden_dim=64):
        super(HybridModel, self).__init__()
        
        # Sigma profile branch
        self.sigma_encoder = SigmaProfileEncoder(sigma_feature_dim, hidden_dim)
        
        # Molecular structure branch
        self.mol_gcn = MolecularGCN(node_feature_dim, hidden_dim)
        
        # Fusion and prediction layers
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.predict = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, sigma_data, mol_data):
        # Process sigma profile data
        sigma_features = self.sigma_encoder(sigma_data)
        
        # Process molecular graph data
        mol_features = self.mol_gcn(mol_data)
        
        # Concatenate features from both branches
        combined = torch.cat([sigma_features, mol_features], dim=1)
        
        # Fusion layer
        combined = F.relu(self.fusion(combined))
        combined = self.dropout(combined)
        
        # Final prediction
        prediction = self.predict(combined)
        
        return prediction.squeeze()


# --- Training and Evaluation Functions ---

def train_epoch(model, sigma_data, mol_data, targets, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(sigma_data, mol_data)
    loss = criterion(predictions, targets)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, sigma_data, mol_data, targets, criterion, device):
    model.eval()
    with torch.no_grad():
        predictions = model(sigma_data, mol_data)
        loss = criterion(predictions, targets)
    
    return loss.item(), predictions


# --- Main Function ---

def main():
    # Define your dataset paths (update these with your actual paths)
    dataset_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\amine_molecules_full_with_UID.csv'
    sigma_profiles_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\sigma_profiles'
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    merged_df, column_names = preprocess_data(dataset_path, sigma_profiles_path)
    
    # Extract sigma profiles and scale them
    X_sigma = merged_df[column_names].values
    scaler = StandardScaler()
    X_sigma_scaled = scaler.fit_transform(X_sigma)
    
    # Prepare molecular graphs from SMILES
    print("Generating molecular graphs from SMILES...")
    smiles_list = merged_df['SMILES'].values
    molecular_graphs, valid_indices = prepare_molecular_graphs(smiles_list)
    
    # Keep only data for molecules where we successfully created graphs
    X_sigma_scaled = X_sigma_scaled[valid_indices]
    y = merged_df['pka_value'].values[valid_indices]
    
    # Create train/test split
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_sigma_train = torch.tensor(X_sigma_scaled[train_idx], dtype=torch.float)
    X_sigma_test = torch.tensor(X_sigma_scaled[test_idx], dtype=torch.float)
    y_train = torch.tensor(y[train_idx], dtype=torch.float)
    y_test = torch.tensor(y[test_idx], dtype=torch.float)
    
    # Prepare molecular graph data for training and testing
    train_graphs = [molecular_graphs[i] for i in train_idx]
    test_graphs = [molecular_graphs[i] for i in test_idx]
    train_batch = Batch.from_data_list(train_graphs)
    test_batch = Batch.from_data_list(test_graphs)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    X_sigma_train = X_sigma_train.to(device)
    X_sigma_test = X_sigma_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    train_batch = train_batch.to(device)
    test_batch = test_batch.to(device)
    
    # Initialize model
    node_feature_dim = train_graphs[0].x.shape[1]  # Dimension of node features
    sigma_feature_dim = X_sigma_scaled.shape[1]    # Dimension of sigma profile features
    model = HybridModel(sigma_feature_dim, node_feature_dim, hidden_dim=64).to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    num_epochs = 1000
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, X_sigma_train, train_batch, y_train, optimizer, criterion, device)
        
        # Evaluate: obtain test loss and predictions for later use
        test_loss, _ = evaluate(model, X_sigma_test, test_batch, y_test, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print progress every 10 epochs and at the final epoch
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_hybrid_model.pt")
    
    print(f"Training complete. Best Test Loss (MSE): {best_test_loss:.4f}")
    
    # --- Metrics and Graphs ---
    
    # Load best model weights
    model.load_state_dict(torch.load("best_hybrid_model.pt"))
    model.eval()
    with torch.no_grad():
        _, test_predictions = evaluate(model, X_sigma_test, test_batch, y_test, criterion, device)
    
    # Convert predictions and ground truth to numpy arrays for metrics calculation
    y_test_np = y_test.cpu().numpy()
    predictions_np = test_predictions.cpu().numpy()
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test_np, predictions_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_np, predictions_np)
    r2 = r2_score(y_test_np, predictions_np)
    
    print("\nPerformance Metrics on Test Set:")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    
    # Plot loss curves over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curves over Training Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curves.png")
    plt.show()
    
    # Scatter plot of predicted vs actual pKa values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_np, predictions_np, c='blue', alpha=0.6, edgecolors='k')
    plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
    plt.xlabel("Actual pKa")
    plt.ylabel("Predicted pKa")
    plt.title("Predicted vs Actual pKa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png")
    plt.show()


if __name__ == "__main__":
    main()
