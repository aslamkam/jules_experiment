#!/usr/bin/env python3
"""
Combine GCN-derived molecular embeddings with Benson group count features,
then train an XGBoost regressor on concatenated features to predict pKa values.
"""
import os
import re
import ast
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
    # --- Load Benson data ---
    benson_file = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa.xlsx"
    df = pd.read_excel(benson_file)
    df['features'] = df['benson_groups'].apply(parse_benson_groups)

    # Vectorize Benson features
    vec = DictVectorizer(sparse=False)
    X_tab = vec.fit_transform(df['features'])
    smiles = df['Smiles'].values
    y = df['pka_value'].values

    # --- Convert to graphs ---
    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles):
        g = molecule_to_graph(smi)
        if g is not None:
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            graphs.append(g)
            valid_idx.append(i)
    X_tab = X_tab[valid_idx]
    y = y[valid_idx]

    # Train/test split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    # Tabular splits
    X_tab_train, X_tab_test = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Graph splits
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    train_batch = Batch.from_data_list(train_graphs)
    test_batch = Batch.from_data_list(test_graphs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch, test_batch = train_batch.to(device), test_batch.to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float, device=device)

    # --- Train GCN ---
    in_dim = train_batch.x.shape[1]
    gcn = MolecularGCNModel(in_dim).to(device)
    opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    epochs = 500
    for ep in range(epochs):
        gcn.train(); opt.zero_grad()
        pred = gcn(train_batch)
        loss = loss_fn(pred, y_train_t)
        loss.backward(); opt.step()
        if ep % 20 == 0:
            print(f"GCN Epoch {ep}/{epochs}, Loss: {loss.item():.4f}")

    # --- Extract embeddings ---
    gcn.eval()
    with torch.no_grad():
        emb_train = gcn.embed(train_batch).cpu().numpy()
        emb_test = gcn.embed(test_batch).cpu().numpy()

    # --- Combine features ---
    X_train_comb = np.hstack([X_tab_train, emb_train])
    X_test_comb = np.hstack([X_tab_test, emb_test])

    # --- Train XGBoost on combined ---
    xgb = XGBRegressor(
        colsample_bytree=0.8, gamma=0, learning_rate=0.1,
        max_depth=7, n_estimators=200,
        reg_alpha=0.1, reg_lambda=1, subsample=0.7, random_state=42
    )
    xgb.fit(X_train_comb, y_train)

    # --- Evaluate ---
    for name, X_, y_ in [('Train', X_train_comb, y_train), ('Test', X_test_comb, y_test)]:
        pred = xgb.predict(X_)
        rmse = mean_squared_error(y_, pred, squared=False)
        mae = mean_absolute_error(y_, pred)
        r2 = r2_score(y_, pred)
        print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

if __name__ == '__main__':
    main()
