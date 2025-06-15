import os
import ast
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def parse_benson_groups(text):
    """Convert the string representation of a defaultdict(int) into a dict."""
    match = re.match(r".*defaultdict\(<class 'int'>,\s*(.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

def atom_to_feature_vector(atom):
    """Build a per-atom feature vector, including a Gasteiger charge."""
    hybrid_map = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4,
    }
    mol = atom.GetOwningMol()
    AllChem.ComputeGasteigerCharges(mol)
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
        charge,
    ]

def smiles_to_graph(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        return None
    # Precompute charges for all atoms once
    AllChem.ComputeGasteigerCharges(mol)
    feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(feats, dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

class BensonMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GraphGCN(nn.Module):
    def __init__(self, node_dim, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, out_dim)
        self.bn1 = BatchNorm(out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.bn2 = BatchNorm(out_dim)
        self.conv3 = GCNConv(out_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, 0.2)
        res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + res)
        x = F.dropout(x, 0.2)
        x = F.relu(self.conv3(x, edge_index))
        return global_mean_pool(x, batch)

class FusionModel(nn.Module):
    def __init__(self, node_feat_dim, benson_dim):
        super().__init__()
        self.gcn = GraphGCN(node_feat_dim, out_dim=64)
        self.benson = BensonMLP(benson_dim, hidden_dims=(64,))
        self.fc1 = nn.Linear(64 + 64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data, benson_feats):
        g_emb = self.gcn(data)
        b_emb = self.benson(benson_feats)
        x = torch.cat([g_emb, b_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze()

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def print_metrics(y_true, y_pred, set_name: str):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {set_name} Metrics ---")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    benson_path = os.path.join(base_dir, "Features", "Chembl-12C", "Benson-Groups", "ChEMBL_amines_12C.xlsx")
    csv_path    = os.path.join(base_dir, "Features", "Chembl-12C", "ChEMBL_amines_12C.csv")
    output_path = os.path.join(script_dir, "output.txt")

    original_stdout = sys.stdout
    output_handle = open(output_path, 'w')
    try:
        sys.stdout = output_handle

        # --- Load data ---
        df_benson = pd.read_excel(benson_path)
        df_csv    = pd.read_csv(csv_path)

        df_benson['features'] = df_benson['benson_groups'].map(parse_benson_groups)
        df_benson = df_benson[['Smiles', 'features']]

        df_csv = df_csv[['Smiles', 'CX Basic pKa']].rename(columns={'CX Basic pKa': 'pka_value'})

        df = pd.merge(df_csv, df_benson, on='Smiles', how='inner').dropna().reset_index(drop=True)

        # Benson vectors
        dict_vec = DictVectorizer(sparse=False)
        X_b      = dict_vec.fit_transform(df['features'])
        y_vals   = df['pka_value'].astype(np.float32).values

        X_b_scaled, scaler = scale_features(X_b)

        # Build graphs
        graphs, benson_feats, targets = [], [], []
        for i, row in df.iterrows():
            g = smiles_to_graph(row['Smiles'])
            if g is None:
                continue
            graphs.append(g)
            benson_feats.append(X_b_scaled[i])
            targets.append(row['pka_value'])

        targets = torch.tensor(targets, dtype=torch.float)
        benson_feats = torch.tensor(np.array(benson_feats), dtype=torch.float)

        idx_all = np.arange(len(graphs))
        train_val_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=42)
        train_idx, val_idx     = train_test_split(train_val_idx, test_size=0.2, random_state=42)

        # Prepare loaders
        def subset(lst, idxs): return [lst[i] for i in idxs]
        train_loader = DataLoader(
            list(zip(subset(graphs, train_idx),
                     benson_feats[train_idx],
                     targets[train_idx])),
            batch_size=32, shuffle=True)
        val_loader = DataLoader(list(zip(subset(graphs, val_idx),
                                         benson_feats[val_idx],
                                         targets[val_idx])), batch_size=32)
        test_loader = DataLoader(list(zip(subset(graphs, test_idx),
                                          benson_feats[test_idx],
                                          targets[test_idx])), batch_size=32)

        # Model, optimizer, loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FusionModel(
            node_feat_dim=graphs[0].x.shape[1],
            benson_dim=benson_feats.shape[1]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        print("Using device:", device)
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Training loop
        train_losses, val_losses = [], []
        for epoch in range(1, 11):
            model.train()
            total_loss = 0.0
            for g, b, y in train_loader:
                g, b, y = g.to(device), b.to(device), y.to(device)
                optimizer.zero_grad()
                preds = model(g, b)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
            avg_train = total_loss / len(train_loader.dataset)
            train_losses.append(avg_train)

            if epoch % 10 == 0 or epoch == 1000:
                model.eval()
                with torch.no_grad():
                    yv, yv_pred = [], []
                    for g, b, y in val_loader:
                        g, b = g.to(device), b.to(device)
                        yv_pred.append(model(g, b).cpu())
                        yv.append(y)
                    yv = torch.cat(yv).numpy()
                    yv_pred = torch.cat(yv_pred).numpy()
                val_loss = mean_squared_error(yv, yv_pred)
                val_losses.append(val_loss)
                print(f"Epoch {epoch:4d} | Train MSE: {avg_train:.4f} | Val MSE: {val_loss:.4f}")

        # Final metrics
        def get_preds(loader):
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for g, b, y in loader:
                    g, b = g.to(device), b.to(device)
                    preds.append(model(g, b).cpu())
                    truths.append(y)
            return (torch.cat(truths).numpy(), torch.cat(preds).numpy())

        y_val, y_val_pred = get_preds(val_loader)
        print_metrics(y_val, y_val_pred, "Validation")
        y_test, y_test_pred = get_preds(test_loader)
        print_metrics(y_test, y_test_pred, "Test")

        # Plotting
        epochs = np.arange(1, len(train_losses) + 1)
        val_epochs = np.arange(10, 11, 10).tolist()
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, label="Train MSE")
        plt.plot(val_epochs, val_losses, label="Val MSE", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig("loss_curve_fusion_model.png")
        plt.close()

        print(f"\nAll outputs to {output_path} and loss_curve_fusion_model.png")

    except Exception as e:
        # Always log to both file and console
        print(f"Error during execution: {e}", file=original_stdout)
        raise
    finally:
        if not output_handle.closed:
            output_handle.close()
        sys.stdout = original_stdout
        print("Finished. Stdout restored.")

if __name__ == "__main__":
    main()
