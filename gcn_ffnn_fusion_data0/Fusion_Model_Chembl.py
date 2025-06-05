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

def preprocess_data(new_dataset_path, new_sigma_profile_csv_path):
    """Load and preprocess dataset and sigma profile CSV files."""
    # Load amines dataset
    amines_df_full = pd.read_csv(new_dataset_path)
    # Keep only the necessary columns, including 'Inchi Key' for merging
    # and original feature names before they are renamed later.
    columns_to_keep = ['Inchi Key', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'Smiles']
    if not all(col in amines_df_full.columns for col in columns_to_keep):
        # Attempt to find columns by case-insensitive matching for flexibility
        available_cols_lower = {col.lower(): col for col in amines_df_full.columns}
        actual_cols_to_keep = []
        missing_cols = []
        for col_k in columns_to_keep:
            if col_k.lower() in available_cols_lower:
                actual_cols_to_keep.append(available_cols_lower[col_k.lower()])
            else:
                missing_cols.append(col_k)
        if missing_cols:
            raise ValueError(f"Dataset CSV at {new_dataset_path} is missing one of the required columns: {missing_cols}. Available columns: {amines_df_full.columns.tolist()}")
        amines_df = amines_df_full[actual_cols_to_keep].copy()
        # Rename to standard casing for consistency within this function
        rename_map = {actual: desired for actual, desired in zip(actual_cols_to_keep, columns_to_keep)}
        amines_df.rename(columns=rename_map, inplace=True)
    else:
        amines_df = amines_df_full[columns_to_keep].copy()


    # Load sigma profile CSV
    sigma_source_df_full = pd.read_csv(new_sigma_profile_csv_path)
    # Case-insensitive check for 'Inchi Key' and 'Sigma Profile'
    inchi_key_col_sigma = next((col for col in sigma_source_df_full.columns if col.lower() == 'inchi key'), None)
    sigma_profile_col_sigma = next((col for col in sigma_source_df_full.columns if col.lower() == 'sigma profile'), None)

    if not inchi_key_col_sigma or not sigma_profile_col_sigma:
        raise ValueError(f"Sigma profile CSV at {new_sigma_profile_csv_path} is missing 'Inchi Key' or 'Sigma Profile' column. Available columns: {sigma_source_df_full.columns.tolist()}")
    
    sigma_source_df = sigma_source_df_full[[inchi_key_col_sigma, sigma_profile_col_sigma]].copy()
    # Rename to standard casing for consistency
    sigma_source_df.rename(columns={inchi_key_col_sigma: 'Inchi Key', sigma_profile_col_sigma: 'Sigma Profile'}, inplace=True)


    # Prepare Inchi Keys for merging (e.g., strip whitespace)
    amines_df['Inchi Key'] = amines_df['Inchi Key'].astype(str).str.strip()
    sigma_source_df['Inchi Key'] = sigma_source_df['Inchi Key'].astype(str).str.strip()

    # Merge the two dataframes using 'Inchi Key'
    merged_df_initial = pd.merge(amines_df, sigma_source_df, on='Inchi Key', how='inner')

    parsed_sigma_profiles_list = []
    valid_inchi_keys_for_final_df = []

    if 'Sigma Profile' not in merged_df_initial.columns: # Should be present due to rename
        raise ValueError("'Sigma Profile' column not found after merging. Check CSVs and merge logic.")

    for index, row in merged_df_initial.iterrows():
        sigma_str = row['Sigma Profile']
        inchi_key = row['Inchi Key']

        if pd.isna(sigma_str):
            continue

        try:
            profile_values = [float(val) for val in str(sigma_str).split(';')]
            if not np.all(np.isfinite(profile_values)):
                continue
            parsed_sigma_profiles_list.append(profile_values)
            valid_inchi_keys_for_final_df.append(inchi_key)
        except ValueError:
            continue
        except Exception:
            continue

    if not parsed_sigma_profiles_list:
        raise ValueError("No valid sigma profiles were successfully parsed from the 'Sigma Profile' column.")

    sigma_profiles_array = np.array(parsed_sigma_profiles_list, dtype=np.float32)
    
    num_sigma_features = sigma_profiles_array.shape[1]
    sigma_feature_column_names = [f'sigma_value_{i}' for i in range(num_sigma_features)]

    sigma_df_parsed = pd.DataFrame(sigma_profiles_array, columns=sigma_feature_column_names)
    sigma_df_parsed['Inchi Key'] = valid_inchi_keys_for_final_df

    merged_df_filtered = merged_df_initial[merged_df_initial['Inchi Key'].isin(valid_inchi_keys_for_final_df)].copy()
    
    # Ensure 'Inchi Key' in merged_df_filtered is also string for merging with sigma_df_parsed
    merged_df_filtered['Inchi Key'] = merged_df_filtered['Inchi Key'].astype(str).str.strip()
    sigma_df_parsed['Inchi Key'] = sigma_df_parsed['Inchi Key'].astype(str).str.strip()


    merged_df_final = pd.merge(merged_df_filtered.drop(columns=['Sigma Profile']), sigma_df_parsed, on='Inchi Key', how='inner')

    merged_df_final = merged_df_final.replace([np.inf, -np.inf], np.nan).dropna(subset=sigma_feature_column_names + ['CX Basic pKa'])


    if 'Smiles' in merged_df_final.columns:
        merged_df_final.rename(columns={'Smiles': 'SMILES'}, inplace=True)
    elif 'smiles' in merged_df_final.columns: # if it was renamed from actual_cols_to_keep
        merged_df_final.rename(columns={'smiles': 'SMILES'}, inplace=True)
    else:
        raise ValueError("Column 'Smiles' (or 'smiles') not found in merged_df_final. Check input CSV column names.")

    if 'CX Basic pKa' in merged_df_final.columns:
        merged_df_final.rename(columns={'CX Basic pKa': 'pka_value'}, inplace=True)
    elif 'cx basic pka' in {col.lower(): col for col in merged_df_final.columns}:
        actual_pkacol_name = {col.lower(): col for col in merged_df_final.columns}['cx basic pka']
        merged_df_final.rename(columns={actual_pkacol_name: 'pka_value'}, inplace=True)
    elif 'pka_value' not in merged_df_final.columns: # Check if it's already named 'pka_value'
             raise ValueError("Column 'CX Basic pKa' (or 'pka_value') not found in merged_df_final.")
    
    required_final_cols = ['SMILES', 'pka_value', 'Molecular Formula', 'Amine Class'] + sigma_feature_column_names
    missing_final_cols = [col for col in required_final_cols if col not in merged_df_final.columns]
    if missing_final_cols:
        raise ValueError(f"Final merged DataFrame is missing columns: {missing_final_cols}. Available: {merged_df_final.columns.tolist()}")

    return merged_df_final, sigma_feature_column_names

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
    # dataset_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data0_Computational_SP_Chembl\available-amine-pka-dataset-full.csv'
    # sigma_profiles_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data0_Computational_SP_Chembl\SigmaProfileData/SigmaProfileData'
    
    print("Loading and preprocessing data for Fusion Model...")
    # The actual paths will be passed as arguments in a future step.
    # For now, ensure preprocess_data can be called with two arguments.
    # merged_df, sigma_columns = preprocess_data(dataset_path, sigma_profiles_path)
    # Example of new call structure (actual arguments to be defined later):
    merged_df, sigma_columns = preprocess_data("Features/Chembl-12C/ChEMBL_amines_12C.csv", "Features/Chembl-12C/Orca-Sigma-Profile/ChEMBL_amines_12C_with_sigma.csv")
    
    # Prepare graph data from SMILES.
    # Now using the renamed 'SMILES' column (converted from 'smiles').
    smiles_list = merged_df['SMILES'].values
    graphs, valid_indices = prepare_molecular_graphs(smiles_list)
    if not graphs:
        print("No valid molecular graphs were generated.")
        return
    
    # Filter targets and sigma profiles using valid indices.
    y = merged_df['pka_value'].values[valid_indices]
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
