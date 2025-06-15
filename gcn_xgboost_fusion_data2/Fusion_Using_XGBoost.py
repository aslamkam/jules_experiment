#!/usr/bin/env python3
"""
Combine GCN-derived molecular embeddings with Benson group count features,
then train an XGBoost regressor on concatenated features to predict pKa values.
"""
import os
import re
import ast
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
import matplotlib.pyplot as plt

# -------------------------------
# Benson group parsing & vectorization
# -------------------------------

def parse_benson_groups(text):
    """Parses the string representation of a defaultdict into a dictionary."""
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    try:
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return {} # Return an empty dict if parsing fails

# -------------------------------
# Graph conversion & embedding model
# -------------------------------

def atom_to_feature_vector(atom):
    """Converts an RDKit atom object to a feature vector."""
    hybrid_map = {
        rdchem.HybridizationType.SP: 0, rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2, rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4
    }
    # Gasteiger charge is computed once for the whole molecule in molecule_to_graph
    charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        hybrid_map.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetMass(),
        charge
    ]

def molecule_to_graph(smiles):
    """Converts a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol) # Compute charges once per molecule

    feats = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(feats, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]]) # Add edges in both directions

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class MolecularGCNModel(nn.Module):
    """Graph Convolutional Network model for molecular embedding."""
    def __init__(self, in_dim, hid_dim=64, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.bn1 = BatchNorm(hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.bn2 = BatchNorm(hid_dim)
        self.conv3 = GCNConv(hid_dim, out_dim)
        self.bn3 = BatchNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(out_dim, 1) # Final layer for pKa prediction

    def forward(self, data):
        """Forward pass for training the GCN model directly."""
        x = self.embed(data)
        return self.fc(x).squeeze()

    def embed(self, data):
        """Generates molecular embeddings from graph data."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN layers with batch norm, ReLU, and residual connection
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x_res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + x_res)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # Global pooling to get a graph-level embedding
        return global_mean_pool(x, batch)

# -------------------------------
# Evaluation and Utility Functions
# -------------------------------

def print_metrics(y_true, y_pred, set_name):
    """Calculates and prints regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {set_name} Set Metrics ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("--------------------------")

def plot_results(y_true, y_pred, set_name):
    """Generates and saves parity and error distribution plots."""
    # Parity Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, label=f'{set_name} Data')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', alpha=0.75, label='Ideal Fit')
    plt.xlabel(f"Actual pKa ({set_name} Set)")
    plt.ylabel(f"Predicted pKa ({set_name} Set)")
    plt.title(f"{set_name} Set: True vs. Predicted pKa")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"parity_plot_{set_name.lower()}_set.png")
    plt.close()

    # Error Distribution Plot
    errors = y_pred - y_true
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, alpha=0.75)
    plt.xlabel(f"Error (Predicted - True) on {set_name} Set")
    plt.ylabel("Frequency")
    plt.title(f"{set_name} Set: Error Distribution")
    plt.grid(True)
    plt.savefig(f"error_dist_{set_name.lower()}_set.png")
    plt.close()

# -------------------------------
# Main Pipeline
# -------------------------------

def main():
    """Main function to run the entire data processing, training, and evaluation pipeline."""
    output_path = "output.txt"
    original_stdout = sys.stdout
    output_handle = open(output_path, 'w')
    sys.stdout = output_handle

    try:
        # --- Load Data ---
        benson_file_path = os.path.join("..", "Features", "Chembl-12C", "Benson-Groups", "ChEMBL_amines_12C.xlsx")
        csv_file_path = os.path.join("..", "Features", "Chembl-12C", "ChEMBL_amines_12C.csv")

        df_benson = pd.read_excel(benson_file_path)
        df_csv = pd.read_csv(csv_file_path)

        # --- Preprocess Data ---
        df_benson['features'] = df_benson['benson_groups'].apply(parse_benson_groups)
        df_csv.rename(columns={'CX Basic pKa': 'pka_value'}, inplace=True)
        df = pd.merge(df_csv, df_benson[['Smiles', 'features']], on='Smiles', how='inner')
        df.dropna(subset=['Smiles', 'pka_value', 'features'], inplace=True)
        df = df.reset_index(drop=True)

        # --- Convert SMILES to Graphs and Filter Invalid Entries ---
        graphs, valid_indices = [], []
        for i, smi in enumerate(df['Smiles']):
            g = molecule_to_graph(smi)
            if g is not None:
                graphs.append(g)
                valid_indices.append(i)
            else:
                print(f"Warning: Could not convert SMILES '{smi}' to graph. Skipping.")
        
        if not graphs:
            raise ValueError("No valid molecular graphs could be created from the input data.")

        # Filter all dataframes and arrays based on valid graphs
        df_filtered = df.iloc[valid_indices].reset_index(drop=True)
        y = df_filtered['pka_value'].values
        metadata_df = df_filtered[['Smiles', 'Molecular Formula', 'Amine Class', 'Inchi Key']]

        # Vectorize tabular Benson group features
        vec = DictVectorizer(sparse=False)
        X_tab = vec.fit_transform(df_filtered['features'])

        # --- Data Splitting ---
        # Stratify to ensure target distribution is similar across splits
        full_indices = np.arange(len(y))
        try:
            train_val_idx, test_idx = train_test_split(full_indices, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=10, labels=False, duplicates='drop'))
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42, stratify=pd.qcut(y[train_val_idx], q=8, labels=False, duplicates='drop'))
        except ValueError: # Fallback for small datasets where stratification fails
            print("Warning: Stratification failed, falling back to standard split.")
            train_val_idx, test_idx = train_test_split(full_indices, test_size=0.2, random_state=42)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)

        X_tab_train, X_tab_val, X_tab_test = X_tab[train_idx], X_tab[val_idx], X_tab[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        metadata_test = metadata_df.iloc[test_idx]

        # --- Scale Tabular Features ---
        scaler = StandardScaler()
        X_tab_train = scaler.fit_transform(X_tab_train)
        X_tab_val = scaler.transform(X_tab_val)
        X_tab_test = scaler.transform(X_tab_test)

        # --- GCN Training and Embedding Extraction ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Create graph batches
        train_batch = Batch.from_data_list([graphs[i] for i in train_idx]).to(device)
        val_batch = Batch.from_data_list([graphs[i] for i in val_idx]).to(device)
        test_batch = Batch.from_data_list([graphs[i] for i in test_idx]).to(device)
        
        y_train_t = torch.tensor(y_train, dtype=torch.float, device=device)
        
        # Initialize and train GCN
        in_dim = graphs[0].x.shape[1]
        gcn = MolecularGCNModel(in_dim).to(device)
        opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        
        print("\n--- Training GCN Model ---")
        epochs = 100
        for ep in range(epochs):
            gcn.train()
            opt.zero_grad()
            pred = gcn(train_batch)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            opt.step()
            if (ep + 1) % 20 == 0:
                print(f"GCN Epoch {ep+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Extract embeddings after training is complete
        print("--- Extracting GCN Embeddings ---")
        gcn.eval()
        with torch.no_grad():
            emb_train = gcn.embed(train_batch).cpu().numpy()
            emb_val = gcn.embed(val_batch).cpu().numpy()
            emb_test = gcn.embed(test_batch).cpu().numpy()

        # --- Combine Features for XGBoost ---
        X_train_comb = np.hstack([X_tab_train, emb_train])
        X_val_comb = np.hstack([X_tab_val, emb_val])
        X_test_comb = np.hstack([X_tab_test, emb_test])

        # --- Train XGBoost on Combined Features ---
        print("\n--- Training XGBoost Regressor ---")
        xgb = XGBRegressor(
            colsample_bytree=0.8, gamma=0, learning_rate=0.1,
            max_depth=7, n_estimators=200,
            reg_alpha=0.1, reg_lambda=1, subsample=0.7, random_state=42, early_stopping_rounds=10
        )
        xgb.fit(X_train_comb, y_train, eval_set=[(X_val_comb, y_val)], verbose=False)
        print("XGBoost training complete.")

        # --- Evaluate Final Model ---
        pred_train = xgb.predict(X_train_comb)
        pred_val = xgb.predict(X_val_comb)
        pred_test = xgb.predict(X_test_comb)
        
        print_metrics(y_train, pred_train, 'Train')
        print_metrics(y_val, pred_val, 'Validation')
        print_metrics(y_test, pred_test, 'Test')

        # --- Generate and Save Plots ---
        plot_results(y_val, pred_val, 'Validation')
        plot_results(y_test, pred_test, 'Test')
        print("\nGenerated and saved parity and error plots.")

        # --- Save Test Predictions to CSV ---
        predictions_df = pd.DataFrame({
            'SMILES': metadata_test['Smiles'],
            'Molecular Formula': metadata_test['Molecular Formula'],
            'Amine Class': metadata_test['Amine Class'],
            'Inchi Key': metadata_test['Inchi Key'],
            'Actual_pKa': y_test,
            'Predicted_pKa': pred_test
        })
        predictions_df.to_csv('pka_predictions_xgboost_fusion.csv', index=False)
        print("Test predictions saved to 'pka_predictions_xgboost_fusion.csv'")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        # Also print to the output file if it's open
        print(f"An error occurred: {e}")
    finally:
        if 'output_handle' in locals() and not output_handle.closed:
            output_handle.close()
        sys.stdout = original_stdout
        print(f"\nFinished. Stdout restored. Full log saved to {output_path}")

if __name__ == '__main__':
    main()