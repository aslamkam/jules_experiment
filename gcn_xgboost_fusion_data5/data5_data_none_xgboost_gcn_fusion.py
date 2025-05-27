import re
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ----------------------------
# Helper: clean Benson groups
# ----------------------------

def clean_benson_group(val):
    try:
        if isinstance(val, str) and val.startswith("defaultdict("):
            match = re.search(r"defaultdict\(.*?,\s*(\{.*\})\)", val)
            if match:
                return ast.literal_eval(match.group(1))
        elif isinstance(val, dict):
            return val
        return ast.literal_eval(val)
    except:
        return {}

# ----------------------------
# GCN encoder
# ----------------------------
class MolecularGCNEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        return global_mean_pool(x, batch)

# ----------------------------
# SMILES to PyG graph
# ----------------------------
def atom_to_feature_vector(atom):
    hybrid_map = {rdchem.HybridizationType.SP:0, rdchem.HybridizationType.SP2:1,
                  rdchem.HybridizationType.SP3:2, rdchem.HybridizationType.SP3D:3,
                  rdchem.HybridizationType.SP3D2:4}
    try:
        charge = float(atom.GetProp('_GasteigerCharge'))
    except:
        charge = 0.0
    return [
        atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetDegree(),
        atom.GetTotalNumHs(), int(atom.GetIsAromatic()),
        hybrid_map.get(atom.GetHybridization(),-1), int(atom.IsInRing()),
        atom.GetMass(), charge
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.ComputeGasteigerCharges(mol)
    features = [atom_to_feature_vector(a) for a in mol.GetAtoms()]
    x = torch.tensor(features, dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ----------------------------
# Load and prepare data
# ----------------------------
# Assumes Excel file has columns: 'benson_groups', 'pka_value', 'SMILES'
excel_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data5_IuPac_Benson_Groups\Filtered_IuPac_benson.xlsx'
df = pd.read_excel(excel_path)
df = df.dropna(subset=['benson_groups', 'pka_value', 'Smiles'])
# clean and parse
df['benson_groups'] = df['benson_groups'].apply(clean_benson_group)
# average if range
def parse_pka(val):
    if isinstance(val, str) and 'to' in val:
        parts = [float(x) for x in val.split('to')]
        return sum(parts)/len(parts)
    return float(val)
df['pka_value'] = df['pka_value'].apply(parse_pka)

# vectorize Benson
vec = DictVectorizer(sparse=False)
X_benson = vec.fit_transform(df['benson_groups'])
y = df['pka_value'].values

# build graphs
graphs, valid_idx = [], []
for idx, smi in enumerate(df['Smiles']):
    g = smiles_to_graph(smi)
    if g is not None:
        graphs.append(g)
        valid_idx.append(idx)
# align features
y = y[valid_idx]
X_benson = X_benson[valid_idx]

# split indices
eids = np.arange(len(y))
train_idx, test_idx = train_test_split(eids, test_size=0.2, random_state=42)

# batch graphs
batch_train = Batch.from_data_list([graphs[i] for i in train_idx])
batch_test  = Batch.from_data_list([graphs[i] for i in test_idx])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_train, batch_test = batch_train.to(device), batch_test.to(device)

# encode
encoder = MolecularGCNEncoder(node_dim=graphs[0].x.shape[1]).to(device)
# TODO: pretrain encoder if desired
encoder.eval()
with torch.no_grad():
    emb_train = encoder(batch_train).cpu().numpy()
    emb_test  = encoder(batch_test).cpu().numpy()

# combine features
X_train = np.hstack([X_benson[train_idx], emb_train])
X_test  = np.hstack([X_benson[test_idx], emb_test])

y_train, y_test = y[train_idx], y[test_idx]

# ----------------------------
# Train XGBoost
# ----------------------------
xgb = XGBRegressor(objective='reg:squarederror', subsample=0.8,
                   n_estimators=200, max_depth=5, learning_rate=0.1,
                   random_state=42)
xgb.fit(X_train, y_train)

# Evaluate
for name, (X_, y_, emb_) in [('Train', (X_train, y_train, emb_train)), ('Test', (X_test, y_test, emb_test))]:
    preds = xgb.predict(X_)
    rmse = np.sqrt(mean_squared_error(y_, preds))
    mae  = mean_absolute_error(y_, preds)
    r2   = r2_score(y_, preds)
    print(f"{name} | RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
