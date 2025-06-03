#!/usr/bin/env python3
"""
Combine GCN-derived molecular embeddings with Benson group count features
and sigma profile features, then train an XGBoost regressor on concatenated features
 to predict pKa values.
"""
import os
import sys
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

# -------------------------------
# Benson group parsing & vectorization
# -------------------------------

def parse_benson_groups(text):
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

# -------------------------------
# Graph conversion & embedding model
# -------------------------------

def atom_to_feature_vector(atom):
    hybrid_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4
    }
    # Gasteiger charges
    AllChem.ComputeGasteigerCharges(atom.GetOwningMol())
    charge = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
    return [
        atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetDegree(),
        atom.GetTotalNumHs(), int(atom.GetIsAromatic()),
        hybrid_map.get(atom.GetHybridization(), -1), int(atom.IsInRing()),
        atom.GetMass(), charge
    ]


def molecule_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)
    feats = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(feats, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


class MolecularGCNModel(nn.Module):
    def __init__(self, in_dim, hid_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.bn1 = BatchNorm(hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.bn2 = BatchNorm(hid_dim)
        self.conv3 = GCNConv(hid_dim, hid_dim)
        self.bn3 = BatchNorm(hid_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x_res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + x_res)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.fc(x).squeeze()

    def embed(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x_res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + x_res)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)

# -------------------------------
# Pipeline: load, preprocess, train & combine
# -------------------------------

def main():
    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    # --- Load combined Benson & sigma data ---
    data_file = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data3_12C_Chembl_Benson_Groups_Sigma_Profile\Amines_12C_CHEMBL_with_sigma_cleaned.xlsx"
    df = pd.read_excel(data_file)

    # Parse Benson group dicts
    df['benson_dict'] = df['benson_groups'].apply(parse_benson_groups)
    # Parse sigma profiles (assumes a single nested list per entry)
    df['sigma_list'] = df['sigma_profile'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else (x[0] if isinstance(x, list) and len(x)==1 else x))

    # Vectorize Benson features
    vec = DictVectorizer(sparse=False)
    X_benson = vec.fit_transform(df['benson_dict'])
    # Build sigma feature matrix
    sigma_matrix = np.vstack(df['sigma_list'].apply(lambda arr: np.array(arr)).values)

    # Combine tabular features
    X_tab = np.hstack([X_benson, sigma_matrix])
    smiles = df['Smiles'].values
    y = df['pka_value'].values

    # --- Convert SMILES to graph objects ---
    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles):
        g = molecule_to_graph(smi)
        if g is not None:
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            graphs.append(g)
            valid_idx.append(i)
    # Filter tabular & labels to valid graphs
    X_tab = X_tab[valid_idx]
    y = y[valid_idx]

    # Train/validation/test split
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

    X_tab_train, X_tab_val, X_tab_test = X_tab[train_idx], X_tab[val_idx], X_tab[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_batch = Batch.from_data_list(train_graphs)
    val_batch = Batch.from_data_list(val_graphs)
    test_batch = Batch.from_data_list(test_graphs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch, val_batch, test_batch = train_batch.to(device), val_batch.to(device), test_batch.to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float, device=device)

    # --- Train GCN model ---
    in_dim = train_batch.x.shape[1]
    gcn = MolecularGCNModel(in_dim).to(device)
    opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    epochs = 1000
    for ep in range(epochs):
        gcn.train()
        opt.zero_grad()
        pred = gcn(train_batch)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            print(f"GCN Epoch {ep}/{epochs}, Loss: {loss.item():.4f}")

    # --- Extract embeddings ---
    gcn.eval()
    with torch.no_grad():
        emb_train = gcn.embed(train_batch).cpu().numpy()
        emb_val = gcn.embed(val_batch).cpu().numpy()
        emb_test = gcn.embed(test_batch).cpu().numpy()

    # --- Combine all features ---
    X_train_comb = np.hstack([X_tab_train, emb_train])
    X_val_comb = np.hstack([X_tab_val, emb_val])
    X_test_comb = np.hstack([X_tab_test, emb_test])

    # --- Train XGBoost on combined features ---
    xgb = XGBRegressor(
        colsample_bytree=0.8, gamma=0, learning_rate=0.1,
        max_depth=7, n_estimators=200,
        reg_alpha=0.1, reg_lambda=1, subsample=0.7, random_state=42
    )
    xgb.fit(X_train_comb, y_train)

    # --- Evaluate performance ---
    for name, X_, y_ in [('Train', X_train_comb, y_train), ('Validation', X_val_comb, y_val), ('Test', X_test_comb, y_test)]:
        pred = xgb.predict(X_)

        mae = mean_absolute_error(y_, pred)
        mse = mean_squared_error(y_, pred)
        rmse = mean_squared_error(y_, pred, squared=False)
        r2 = r2_score(y_, pred)

        if name == 'Train':
            print(f"\n{name} Set Metrics:")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE:  {mae:.3f}")
            print(f"  R2:   {r2:.3f}")
        elif name in ['Validation', 'Test']:
            print(f"\n{name} Set Metrics:")
            print(f"  MAE:  {mae:.3f}")
            print(f"  MSE:  {mse:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R2:   {r2:.3f}")

            errors = [p - t for p, t in zip(pred, y_)]
            abs_errors = [abs(e) for e in errors]

            max_abs_err = max(abs_errors) if abs_errors else 0.0

            pct_abs_err_le_02 = (sum(e <= 0.2 for e in abs_errors) / len(abs_errors) * 100) if abs_errors else 0.0
            pct_err_in_0_02 = (sum(0 < e <= 0.2 for e in errors) / len(errors) * 100) if errors else 0.0
            pct_err_in_neg02_0 = (sum(-0.2 < e < 0 for e in errors) / len(errors) * 100) if errors else 0.0

            pct_abs_err_le_04 = (sum(e <= 0.4 for e in abs_errors) / len(abs_errors) * 100) if abs_errors else 0.0
            pct_err_in_0_04 = (sum(0 < e <= 0.4 for e in errors) / len(errors) * 100) if errors else 0.0
            pct_err_in_neg04_0 = (sum(-0.4 < e < 0 for e in errors) / len(errors) * 100) if errors else 0.0

            print(f"  Max Abs Error:      {max_abs_err:.3f}")
            print(f"  % |Err| <= 0.2:     {pct_abs_err_le_02:.3f}%")
            print(f"  % Err in (0,0.2]:   {pct_err_in_0_02:.3f}%")
            print(f"  % Err in (-0.2,0):  {pct_err_in_neg02_0:.3f}%")
            print(f"  % |Err| <= 0.4:     {pct_abs_err_le_04:.3f}%")
            print(f"  % Err in (0,0.4]:   {pct_err_in_0_04:.3f}%")
            print(f"  % Err in (-0.4,0):  {pct_err_in_neg04_0:.3f}%")

        # --- Generate and Save Plots ---
        # Parity Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_, pred, alpha=0.5, label=f'{name} Data')
        min_val = min(min(y_), min(pred)) if len(y_) > 0 and len(pred) > 0 else 0
        max_val = max(max(y_), max(pred)) if len(y_) > 0 and len(pred) > 0 else 1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x')
        plt.xlabel('True pKa')
        plt.ylabel('Predicted pKa')
        plt.title(f'{name} Set: Predicted vs True pKa')
        plt.legend()
        plt.grid(True)
        parity_plot_path = os.path.join(script_dir_main, f'parity_plot_{name.lower()}.png')
        plt.savefig(parity_plot_path)
        plt.close()
        print(f"  Saved parity plot to {parity_plot_path}")

        # Error Distribution Plot
        # errors var was calculated above if name is Validation or Test, recalculate for Train or if not available
        current_errors = pred - y_ # Both pred and y_ are numpy arrays
        plt.figure(figsize=(8, 6))
        plt.hist(current_errors, bins=20, alpha=0.7, label=f'{name} Errors')
        plt.xlabel('Error (Predicted - True pKa)')
        plt.ylabel('Count')
        plt.title(f'{name} Set: Error Distribution')
        plt.legend()
        plt.grid(True)
        error_dist_plot_path = os.path.join(script_dir_main, f'error_dist_{name.lower()}.png')
        plt.savefig(error_dist_plot_path)
        plt.close()
        print(f"  Saved error distribution plot to {error_dist_plot_path}")


if __name__ == '__main__':
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the output file path
    output_file_path = os.path.join(script_dir, "output.txt")

    # Store the original stdout
    original_stdout = sys.stdout

    try:
        # Redirect stdout to the output file
        sys.stdout = open(output_file_path, 'w')
        print(f"Outputting to: {output_file_path}") # Optional: confirm redirection
        main()
    except Exception as e:
        # If any error occurs, print it to original stdout (console)
        # and also to the file if redirection was successful.
        sys.stdout = original_stdout # Restore for console print
        print(f"An error occurred: {e}", file=sys.stderr) # Print error to stderr
        # Attempt to log to file as well, if it was opened
        try:
            with open(output_file_path, 'a') as f_err:
                f_err.write(f"\nAn error occurred during execution: {e}\n")
                import traceback
                traceback.print_exc(file=f_err)
        except Exception: # If logging to file fails
            pass # Avoid further errors during error handling
        raise # Re-raise the exception
    finally:
        # Ensure stdout is restored
        if sys.stdout.name == output_file_path:
            sys.stdout.close()
        sys.stdout = original_stdout
        print("Output redirection finished. Results are in output.txt") # Optional: confirm restoration
