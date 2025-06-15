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
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

# -------------------------------
# Graph conversion & embedding model
# -------------------------------

def atom_to_feature_vector(atom):
    hybrid_map = {rdchem.HybridizationType.SP:0, rdchem.HybridizationType.SP2:1,
                  rdchem.HybridizationType.SP3:2, rdchem.HybridizationType.SP3D:3,
                  rdchem.HybridizationType.SP3D2:4}
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
        # conv layers
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x_res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + x_res)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        # global pooling
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
    output_path = "output.txt"
    original_stdout = sys.stdout
    output_handle = open(output_path, 'w')
    sys.stdout = output_handle

    try:
        # --- Load Data ---
        # Define relative paths
    benson_file_path = os.path.join("Features", "Chembl-12C", "Benson-Groups", "ChEMBL_amines_12C.xlsx")
    csv_file_path = os.path.join("Features", "Chembl-12C", "ChEMBL_amines_12C.csv")

    # Load data
    df_benson = pd.read_excel(benson_file_path)
    df_csv = pd.read_csv(csv_file_path)

    # Process Benson data
    df_benson['features'] = df_benson['benson_groups'].apply(parse_benson_groups)
    df_benson = df_benson[['Smiles', 'features']]

    # Process CSV data
    df_csv.rename(columns={'CX Basic pKa': 'pka_value'}, inplace=True)
    df_csv = df_csv[['Smiles', 'pka_value', 'Molecular Formula', 'Amine Class', 'Inchi Key']]
    df_csv.dropna(subset=['Smiles', 'pka_value'], inplace=True)

    # Merge dataframes
    df = pd.merge(df_csv, df_benson, on='Smiles', how='inner')
    df.dropna(subset=['pka_value', 'features'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Feature and Target Extraction ---
    vec = DictVectorizer(sparse=False)
    # X_tab will be created after graph conversion and filtering
    y = df['pka_value'].values
    metadata_df = df[['Smiles', 'Molecular Formula', 'Amine Class', 'Inchi Key']]
    smiles_list = df['Smiles'].tolist()

    # --- Convert to graphs and filter invalid SMILES ---
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = molecule_to_graph(smi)
        if g is not None:
            # Store graph with a placeholder for batch attribute, to be assigned later
            graphs.append(g)
            valid_indices.append(i)
        else:
            print(f"Warning: Could not convert SMILES '{smi}' to graph. Skipping.")

    # Filter y, metadata, and df based on valid graphs
    y = y[valid_indices]
    metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)

    # Now create X_tab from the filtered df
    X_tab = vec.fit_transform(df_filtered['features'])


    # --- Data Splitting ---
    full_indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(full_indices, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=10, labels=False, duplicates='drop'))
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42, stratify=pd.qcut(y[train_val_idx], q=8, labels=False, duplicates='drop')) # 0.25 * 0.8 = 0.2

    # Tabular splits
    X_tab_train, X_tab_val, X_tab_test = X_tab[train_idx], X_tab[val_idx], X_tab[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    metadata_train, metadata_val, metadata_test = metadata_df.iloc[train_idx], metadata_df.iloc[val_idx], metadata_df.iloc[test_idx]

    # Graph splits
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # Batch.from_data_list will create the batch attribute correctly.
    # No need to manually assign g.batch here if creating Batch objects for the whole split.
    train_batch = Batch.from_data_list(train_graphs) if train_graphs else Batch() # Handle empty list
    val_batch = Batch.from_data_list(val_graphs) if val_graphs else Batch()
    test_batch = Batch.from_data_list(test_graphs) if test_graphs else Batch()

    # --- Scale Benson Features ---
    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab_train)
    X_tab_val = scaler.transform(X_tab_val)
    X_tab_test = scaler.transform(X_tab_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch, val_batch, test_batch = train_batch.to(device), val_batch.to(device), test_batch.to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float, device=device)
    # y_val_t needed for GCN validation if we implement it
    y_test_t = torch.tensor(y_test, dtype=torch.float, device=device)

    # --- Train GCN ---
    if not graphs: # No valid graphs were created
        print("No valid molecular graphs to process. Exiting GCN training.")
        # Set up dummy variables to allow pipeline to proceed if desired, or exit
        emb_train, emb_val, emb_test = [np.array([]).reshape(X_tab_train.shape[0] if i==0 else (X_tab_val.shape[0] if i==1 else X_tab_test.shape[0]), 0) for i in range(3)]
    else:
        in_dim = graphs[0].x.shape[1] # Get in_dim from the first valid graph
        gcn = MolecularGCNModel(in_dim).to(device)
        opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
    epochs = 11 # Updated epochs
    for ep in range(epochs):
        gcn.train(); opt.zero_grad()
        # Ensure train_batch is not empty and has nodes
        if train_batch.x is not None and train_batch.x.size(0) > 0:
            pred = gcn(train_batch)
            loss = loss_fn(pred, y_train_t)
            loss.backward(); opt.step()
            if ep % 20 == 0 or ep == epochs -1: # Print more frequently or at least the last one
                print(f"GCN Epoch {ep+1}/{epochs}, Loss: {loss.item():.4f}")
        else:
            print(f"Skipping GCN training epoch {ep+1} due to empty train batch.")
            break # or continue, depending on desired behavior

        # --- Extract embeddings ---
        gcn.eval()
        with torch.no_grad():
            emb_train = gcn.embed(train_batch).cpu().numpy() if train_batch.x is not None and train_batch.x.size(0) > 0 else np.array([]).reshape(X_tab_train.shape[0], 0)
            emb_val = gcn.embed(val_batch).cpu().numpy() if val_batch.x is not None and val_batch.x.size(0) > 0 else np.array([]).reshape(X_tab_val.shape[0], 0)
            emb_test = gcn.embed(test_batch).cpu().numpy() if test_batch.x is not None and test_batch.x.size(0) > 0 else np.array([]).reshape(X_tab_test.shape[0], 0)

    # --- Combine features ---
    # Ensure correct dimensions if some embeddings are empty
    # If emb_*.size is 0 (because it's reshaped to (N,0)), hstack still works as expected.
    X_train_comb = np.hstack([X_tab_train, emb_train])
    X_val_comb = np.hstack([X_tab_val, emb_val])
    X_test_comb = np.hstack([X_tab_test, emb_test])

# -------------------------------
# Pipeline: load, preprocess, train & combine
# -------------------------------


    # --- Train XGBoost on combined ---
    xgb = XGBRegressor(
        colsample_bytree=0.8, gamma=0, learning_rate=0.1,
        max_depth=7, n_estimators=20, # Updated n_estimators
        reg_alpha=0.1, reg_lambda=1, subsample=0.7, random_state=42
    )
    # Check if X_train_comb is not empty before fitting
    if X_train_comb.shape[0] > 0:
        xgb.fit(X_train_comb, y_train, eval_set=[(X_val_comb, y_val)], early_stopping_rounds=10, verbose=False)
    else:
        print("Skipping XGBoost training due to empty training data.")


    # --- Evaluate ---
    # Ensure that data exists before trying to predict and evaluate
    evaluation_results = {} # Store predictions for plotting

    if X_train_comb.shape[0] > 0 and y_train.shape[0] > 0:
        if xgb.get_booster().num_boosted_rounds() > 0:
            pred_train = xgb.predict(X_train_comb)
            print_metrics(y_train, pred_train, 'Train')
            evaluation_results['Train'] = {'true': y_train, 'pred': pred_train}
        else:
            print(f"\nSkipping evaluation for Train as XGBoost model was not trained.")

    if X_val_comb.shape[0] > 0 and y_val.shape[0] > 0:
        if xgb.get_booster().num_boosted_rounds() > 0:
            pred_val = xgb.predict(X_val_comb)
            print_metrics(y_val, pred_val, 'Validation')
            evaluation_results['Validation'] = {'true': y_val, 'pred': pred_val}
        else:
            print(f"\nSkipping evaluation for Validation as XGBoost model was not trained.")

    if X_test_comb.shape[0] > 0 and y_test.shape[0] > 0:
        if xgb.get_booster().num_boosted_rounds() > 0:
            pred_test = xgb.predict(X_test_comb)
            print_metrics(y_test, pred_test, 'Test')
            evaluation_results['Test'] = {'true': y_test, 'pred': pred_test}
        else:
            print(f"\nSkipping evaluation for Test as XGBoost model was not trained.")

    # --- Generate and Save Plots ---
    if 'Validation' in evaluation_results:
        y_val_true = evaluation_results['Validation']['true']
        y_val_pred = evaluation_results['Validation']['pred']

        plt.figure(figsize=(8, 8))
        plt.scatter(y_val_true, y_val_pred, alpha=0.5)
        plt.plot([min(y_val_true.min(),y_val_pred.min()), max(y_val_true.max(),y_val_pred.max())], [min(y_val_true.min(),y_val_pred.min()), max(y_val_true.max(),y_val_pred.max())], 'r--')
        plt.xlabel("Actual pKa (Validation Set)")
        plt.ylabel("Predicted pKa (Validation Set)")
        plt.title("Validation Set: True vs Predicted pKa")
        plt.savefig("parity_plot_validation_set.png")
        plt.close()

        val_errors = y_val_pred - y_val_true
        plt.figure(figsize=(8, 6))
        plt.hist(val_errors, bins=30, alpha=0.7)
        plt.xlabel("Error (Predicted - True) on Validation Set")
        plt.ylabel("Count")
        plt.title("Validation Set: Error Distribution")
        plt.savefig("error_dist_validation_set.png")
        plt.close()

    if 'Test' in evaluation_results:
        y_test_true = evaluation_results['Test']['true']
        y_test_pred = evaluation_results['Test']['pred']

        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_true, y_test_pred, alpha=0.5)
        plt.plot([min(y_test_true.min(),y_test_pred.min()), max(y_test_true.max(),y_test_pred.max())], [min(y_test_true.min(),y_test_pred.min()), max(y_test_true.max(),y_test_pred.max())], 'r--')
        plt.xlabel("Actual pKa (Test Set)")
        plt.ylabel("Predicted pKa (Test Set)")
        plt.title("Test Set: True vs Predicted pKa")
        plt.savefig("parity_plot_test_set.png")
        plt.close()

        test_errors = y_test_pred - y_test_true
        plt.figure(figsize=(8, 6))
        plt.hist(test_errors, bins=30, alpha=0.7)
        plt.xlabel("Error (Predicted - True) on Test Set")
        plt.ylabel("Count")
        plt.title("Test Set: Error Distribution")
        plt.savefig("error_dist_test_set.png")
        plt.close()

    # --- Save Test Predictions to CSV ---
    if 'Test' in evaluation_results and metadata_test is not None:
        y_test_true = evaluation_results['Test']['true']
        y_test_pred = evaluation_results['Test']['pred']

        # Ensure metadata_test is aligned with y_test_true/pred.
        # The splitting logic should ensure this, but an explicit check might be added if issues arise.
        # Assuming metadata_test is already correctly filtered and ordered.

        predictions_df = pd.DataFrame({
            'SMILES': metadata_test['Smiles'].values,
            'Molecular Formula': metadata_test['Molecular Formula'].values,
            'Amine Class': metadata_test['Amine Class'].values,
            'Inchi Key': metadata_test['Inchi Key'].values,
            'Actual_pKa': y_test_true,
            'Predicted_pKa': y_test_pred
        })
        predictions_df.to_csv('pka_predictions_xgboost_fusion.csv', index=False)
        print("\nTest predictions saved to 'pka_predictions_xgboost_fusion.csv'")
    else:
        print("\nSkipping saving of test predictions as test data or predictions were not available.")

    finally:
        if 'output_handle' in locals() and output_handle and not output_handle.closed:
            output_handle.close()
        sys.stdout = original_stdout
        print(f"Finished. Stdout restored. Output saved to {output_path}")


if __name__ == '__main__':
    main()
