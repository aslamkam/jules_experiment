import os
import sys
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
    
    # Create train/validation/test split.
    indices = np.arange(len(y))
    
    # First split: (train+validation) and test
    # For example, 80% for train+validation, 20% for test
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Second split: train and validation from the (train+validation) set
    # For example, 80% of the (train+validation) for train, 20% for validation
    # This means validation will be 0.2 * 0.8 = 0.16 of the total dataset.
    # Training will be 0.8 * 0.8 = 0.64 of the total dataset.
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42) # test_size=0.2 means 20% of train_val_idx for validation

    # Prepare graph data for validation set
    val_graphs = [graphs[i] for i in val_idx]
    val_graph_batch = Batch.from_data_list(val_graphs)

    # Prepare sigma data for validation set
    sigma_val = torch.tensor(sigma_data_scaled[val_idx], dtype=torch.float)

    # Prepare target data for validation set
    y_val = torch.tensor(y[val_idx], dtype=torch.float)

    # Existing train and test set preparations remain largely the same, just using the new indices.
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
    val_graph_batch = val_graph_batch.to(device)
    sigma_val = sigma_val.to(device)
    y_val = y_val.to(device)
    
    # Set dimensions for the two branches.
    node_feature_dim = train_graphs[0].x.shape[1]
    sigma_input_dim = sigma_train.shape[1]
    
    model = FusionModel(node_feature_dim, sigma_input_dim, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    num_epochs = 2200
    train_losses = []
    val_losses = []
    
    print("Starting training for Fusion Model...")
    for epoch in range(num_epochs):
        train_loss, _ = train_epoch(model, optimizer, criterion, train_graph_batch, sigma_train, y_train, device)
        val_loss, _ = evaluate(model, criterion, val_graph_batch, sigma_val, y_val, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Final evaluation on Validation Set
    model.eval()
    with torch.no_grad():
        val_preds_tensor = model(val_graph_batch, sigma_val)
        val_preds = val_preds_tensor.cpu().numpy()
        y_val_true = y_val.cpu().numpy()

    print("\nFinal Validation Set Metrics:")
    val_mae = mean_absolute_error(y_val_true, val_preds)
    val_mse = mean_squared_error(y_val_true, val_preds)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val_true, val_preds)

    print(f"MAE (Validation): {val_mae:.4f}")
    print(f"MSE (Validation): {val_mse:.4f}")
    print(f"RMSE (Validation): {val_rmse:.4f}")
    print(f"R^2 (Validation): {val_r2:.4f}")

    validation_errors = val_preds - y_val_true
    abs_validation_errors = np.abs(validation_errors)
    print(f"Validation Set Errors (predicted - true): {validation_errors.tolist()}")


    if len(abs_validation_errors) > 0:
        max_abs_error_val = np.max(abs_validation_errors)
        pct_err_le_02_val = np.sum(abs_validation_errors <= 0.2) / len(abs_validation_errors) * 100
        pct_err_gt0_le_02_val = np.sum((validation_errors > 0) & (validation_errors <= 0.2)) / len(validation_errors) * 100
        pct_err_gt_neg02_lt0_val = np.sum((validation_errors > -0.2) & (validation_errors < 0)) / len(validation_errors) * 100
        pct_err_le_04_val = np.sum(abs_validation_errors <= 0.4) / len(abs_validation_errors) * 100
        pct_err_gt0_le_04_val = np.sum((validation_errors > 0) & (validation_errors <= 0.4)) / len(validation_errors) * 100
        pct_err_gt_neg04_lt0_val = np.sum((validation_errors > -0.4) & (validation_errors < 0)) / len(validation_errors) * 100
    else:
        max_abs_error_val = 0.0
        pct_err_le_02_val = 0.0
        pct_err_gt0_le_02_val = 0.0
        pct_err_gt_neg02_lt0_val = 0.0
        pct_err_le_04_val = 0.0
        pct_err_gt0_le_04_val = 0.0
        pct_err_gt_neg04_lt0_val = 0.0

    print(f"Max Abs Error (Validation): {max_abs_error_val:.4f}")
    print(f"% |Err|<=0.2 (Validation): {pct_err_le_02_val:.3f}%")
    print(f"% Err in (0,0.2] (Validation): {pct_err_gt0_le_02_val:.3f}%")
    print(f"% Err in (-0.2,0) (Validation): {pct_err_gt_neg02_lt0_val:.3f}%")
    print(f"% |Err|<=0.4 (Validation): {pct_err_le_04_val:.3f}%")
    print(f"% Err in (0,0.4] (Validation): {pct_err_gt0_le_04_val:.3f}%")
    print(f"% Err in (-0.4,0) (Validation): {pct_err_gt_neg04_lt0_val:.3f}%")
    
    # Calculate metrics on test set.
    model.eval()
    with torch.no_grad():
        preds_tensor = model(test_graph_batch, sigma_test) # Get tensor output
        preds = preds_tensor.cpu().numpy()
        y_true = y_test.cpu().numpy()

    # Existing metrics
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    
    print("\nFinal Test Set Metrics:") # Ensure title is clear
    print(f"MAE (Test): {mae:.4f}") # mae, mse, etc. are already calculated for test set
    print(f"MSE (Test): {mse:.4f}")
    print(f"RMSE (Test): {rmse:.4f}")
    print(f"R^2 (Test): {r2:.4f}")

    # Calculate additional metrics for the test set
    errors_test = preds - y_true
    print(f"Test Set Errors (predicted - true): {errors_test.tolist()}")
    abs_errors_test = np.abs(errors_test)

    if len(abs_errors_test) > 0: # Ensure there are errors to calculate
        max_abs_error_test = np.max(abs_errors_test)
        pct_err_le_02_test = np.sum(abs_errors_test <= 0.2) / len(abs_errors_test) * 100
        pct_err_gt0_le_02_test = np.sum((errors_test > 0) & (errors_test <= 0.2)) / len(errors_test) * 100
        pct_err_gt_neg02_lt0_test = np.sum((errors_test > -0.2) & (errors_test < 0)) / len(errors_test) * 100
        pct_err_le_04_test = np.sum(abs_errors_test <= 0.4) / len(abs_errors_test) * 100
        pct_err_gt0_le_04_test = np.sum((errors_test > 0) & (errors_test <= 0.4)) / len(errors_test) * 100
        pct_err_gt_neg04_lt0_test = np.sum((errors_test > -0.4) & (errors_test < 0)) / len(errors_test) * 100
    else:
        max_abs_error_test = 0.0
        pct_err_le_02_test = 0.0
        pct_err_gt0_le_02_test = 0.0
        pct_err_gt_neg02_lt0_test = 0.0
        pct_err_le_04_test = 0.0
        pct_err_gt0_le_04_test = 0.0
        pct_err_gt_neg04_lt0_test = 0.0

    print(f"Max Abs Error (Test): {max_abs_error_test:.4f}")
    print(f"% |Err|<=0.2 (Test): {pct_err_le_02_test:.3f}%")
    print(f"% |Err|<=0.4 (Test): {pct_err_le_04_test:.3f}%")
    print(f"% Err in (0,0.2] (Test): {pct_err_gt0_le_02_test:.3f}%")
    print(f"% Err in (-0.2,0) (Test): {pct_err_gt_neg02_lt0_test:.3f}%")
    print(f"% Err in (0,0.4] (Test): {pct_err_gt0_le_04_test:.3f}%")
    print(f"% Err in (-0.4,0) (Test): {pct_err_gt_neg04_lt0_test:.3f}%")
    # Optional detailed error percentages can be uncommented by the user if needed.
    
    # Plotting section
    # Loss Curve Plot
    plt.figure(figsize=(8, 6)) # Can be a single plot figure
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curve - Fusion Model")
    plt.legend()
    plt.savefig("loss_curve_fusion_model.png")
    plt.close()

    # Validation Set Parity Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_val_true, val_preds, alpha=0.6, label='Validation Data')
    plt.xlabel("True pKa (Validation Set)")
    plt.ylabel("Predicted pKa (Validation Set)")
    plt.title("Validation Set: True vs Predicted pKa")
    if len(y_val_true) > 0 and len(val_preds) > 0:
        min_val_v = min(np.min(y_val_true), np.min(val_preds))
        max_val_v = max(np.max(y_val_true), np.max(val_preds))
        plt.plot([min_val_v, max_val_v], [min_val_v, max_val_v], 'r--', label='y=x')
    plt.legend()
    plt.savefig("parity_plot_validation_set.png")
    plt.close()

    # Validation Set Error Distribution Plot
    plt.figure(figsize=(6, 5))
    plt.hist(validation_errors, bins=20)
    plt.xlabel('Error (Predicted - True) on Validation Set')
    plt.ylabel('Count')
    plt.title('Validation Set: Error Distribution')
    plt.savefig("error_dist_validation_set.png")
    plt.close()

    # Test Set Parity Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, preds, alpha=0.6, label='Test Data') # y_true and preds are from test set
    plt.xlabel("True pKa (Test Set)")
    plt.ylabel("Predicted pKa (Test Set)")
    plt.title("Test Set: True vs Predicted pKa")
    if len(y_true) > 0 and len(preds) > 0:
        min_val_t = min(np.min(y_true), np.min(preds))
        max_val_t = max(np.max(y_true), np.max(preds))
        plt.plot([min_val_t, max_val_t], [min_val_t, max_val_t], 'r--', label='y=x')
    plt.legend()
    plt.savefig("parity_plot_test_set.png")
    plt.close()

    # Test Set Error Distribution Plot
    plt.figure(figsize=(6, 5))
    plt.hist(errors_test, bins=20) # errors_test is from test set
    plt.xlabel('Error (Predicted - True) on Test Set')
    plt.ylabel('Count')
    plt.title('Test Set: Error Distribution')
    plt.savefig("error_dist_test_set.png")
    plt.close()
    
    # Remove plt.show() as it's not needed for automated scripts and can cause issues.
    # The existing plt.show() at the end of the original script should be removed or commented out.
    # plt.tight_layout() # This can also be kept or removed, less critical than show()
    # plt.show() # REMOVE OR COMMENT OUT THIS LINE

if __name__ == "__main__":
    script_dir_for_output = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir_for_output, "output.txt")
    original_stdout = sys.stdout
    output_file_handle = None # Keep this for the finally block

    try:
        # Redirect stdout to file
        output_file_handle = open(output_file_path, 'w')
        sys.stdout = output_file_handle
        main()
    finally:
        if sys.stdout.name == output_file_path: # Check if stdout is our file
             if output_file_handle: # Ensure it's not None
                output_file_handle.close()
        sys.stdout = original_stdout # Restore original stdout
