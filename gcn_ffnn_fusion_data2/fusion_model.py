import os
import ast
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def parse_benson_groups(text):
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

# Load Benson group data
benson_path = r"C:\Users\kaslam\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa.xlsx"
df = pd.read_excel(benson_path)
df['features'] = df['benson_groups'].apply(parse_benson_groups)
df = df.dropna(subset=['features', 'pka_value', 'Smiles']).reset_index(drop=True)

dict_vec = DictVectorizer(sparse=False)
X_b = dict_vec.fit_transform(df['features'])  # Benson group counts
y = df['pka_value'].values.astype(np.float32)

# Scale Benson features
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

X_b_scaled, scaler = scale_features(X_b)

# ------------------------------------
# Molecular Graph Generation Functions
# ------------------------------------

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

def smiles_to_graph(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None: return None
    AllChem.ComputeGasteigerCharges(mol)
    feats = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(feats, dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i,j],[j,i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0),dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# Build dataset
graphs = []
groups = []
targets = []
for i,row in df.iterrows():
    g = smiles_to_graph(row['Smiles'])
    if g is None: continue
    graphs.append(g)
    groups.append(X_b_scaled[i])
    targets.append(row['pka_value'])

targets = torch.tensor(targets, dtype=torch.float)
groups = torch.tensor(groups, dtype=torch.float)

# Train-test split indices
train_idx, test_idx = train_test_split(np.arange(len(graphs)), test_size=0.2, random_state=42)

train_graphs = [graphs[i] for i in train_idx]
train_groups = groups[train_idx]
train_y = targets[train_idx]

test_graphs = [graphs[i] for i in test_idx]
test_groups = groups[test_idx]
test_y = targets[test_idx]

train_loader = DataLoader(list(zip(train_graphs, train_groups, train_y)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(test_graphs, test_groups, test_y)), batch_size=32)

# -------------------------------
# Model Definitions
# -------------------------------

class BensonMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

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
        x = F.dropout(x,0.2)
        res = x
        x = F.relu(self.bn2(self.conv2(x, edge_index)) + res)
        x = F.dropout(x,0.2)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

class FusionModel(nn.Module):
    def __init__(self, node_feat_dim, benson_dim):
        super().__init__()
        self.gcn = GraphGCN(node_feat_dim, out_dim=64)
        self.benson = BensonMLP(benson_dim, hidden_dims=[64])
        self.fc1 = nn.Linear(64+64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32,1)
    def forward(self, data, benson_feats):
        g_emb = self.gcn(data)
        b_emb = self.benson(benson_feats)
        x = torch.cat([g_emb, b_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze()

# -------------------------------
# Training Loop
# -------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FusionModel(node_feat_dim=train_graphs[0].x.shape[1], benson_dim=train_groups.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

def train():
    model.train()
    total_loss = 0
    for g,b,y in train_loader:
        g = g.to(device)
        b = b.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(g,b)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*y.size(0)
    return total_loss/len(train_loader.dataset)

def evaluate(loader):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for g,b,y in loader:
            g = g.to(device)
            b = b.to(device)
            preds = model(g,b).cpu()
            all_preds.append(preds)
            all_y.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_y = torch.cat(all_y).numpy()
    return all_y, all_preds

print("Using device:", device)
if device.type == 'cuda':
    print(f"  --> GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  --> Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  --> Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Run training
epochs=1000
for ep in range(1, epochs+1):
    loss = train()
    if ep % 10 == 0:
        y_true, y_pred = evaluate(test_loader)
        print(f"Epoch {ep}: Train Loss={loss:.4f}, Test MAE={mean_absolute_error(y_true, y_pred):.4f}")

# Final evaluation
y_true, y_pred = evaluate(test_loader)
print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R2:", r2_score(y_true, y_pred))
