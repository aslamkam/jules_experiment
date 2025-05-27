import os
import re
import ast
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdchem
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Paths
dir_itp = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Partial-Charges\chembl_AM1BCC_charges\chembl_AM1BCC_charges"
file_csv = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\ChEMBL_amines_12C.csv"
file_benson = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Benson-Groups\ChEMBL_amines_12C.xlsx"
file_sigma = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Mcginn-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"

# 1. Load & merge

df_pka = pd.read_csv(file_csv).dropna(subset=['CX Basic pKa'])
df_benson = pd.read_excel(file_benson)
df_sigma = pd.read_csv(file_sigma)
# Inner joins
df = df_pka.merge(df_benson[['ChEMBL ID','benson_groups']], on='ChEMBL ID')
df = df.merge(df_sigma[['ChEMBL ID','Sigma Profile']], on='ChEMBL ID')

# 2. Build samples list

samples = []
for _, row in df.iterrows():
    cid = row['ChEMBL ID']
    itp = os.path.join(dir_itp, f"{cid}_GMX.itp")
    if not os.path.isfile(itp):
        continue
    # parse Benson groups
    bg = row['benson_groups']
    if isinstance(bg, str) and 'defaultdict' in bg:
        try:
            dict_str = bg[bg.find('{'):bg.rfind('}')+1]
            benson = ast.literal_eval(dict_str)
        except:
            continue
    else:
        continue
    # parse Sigma
    sp = row['Sigma Profile']
    if isinstance(sp, str):
        try:
            sigma = ast.literal_eval(sp)
        except:
            continue
    else:
        continue
    samples.append({'id':cid, 'smiles':row['Smiles'], 'pka':float(row['CX Basic pKa']),
                    'itp':itp, 'benson':benson, 'sigma':sigma})

print(f"Loaded {len(samples)} samples.")

# Gather keys and lengths
all_benson_keys = sorted({k for s in samples for k in s['benson'].keys()})
sigma_len = len(samples[0]['sigma']) if samples else 0

# 3. Helpers

def atom_feature_vector(atom: rdchem.Atom):
    hyb_map = {rdchem.HybridizationType.SP:0, rdchem.HybridizationType.SP2:1,
               rdchem.HybridizationType.SP3:2, rdchem.HybridizationType.SP3D:3,
               rdchem.HybridizationType.SP3D2:4}
    return [atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetDegree(),
            atom.GetTotalNumHs(), int(atom.GetIsAromatic()),
            hyb_map.get(atom.GetHybridization(), -1), int(atom.IsInRing()), atom.GetMass()]

def parse_charges(itp_path: str):
    charges = []
    with open(itp_path) as f:
        in_atoms = False
        for line in f:
            if line.strip().startswith('[ atoms ]'):
                in_atoms = True
                next(f)
                continue
            if in_atoms:
                if line.strip().startswith('['):
                    break
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 7:
                    charges.append(float(parts[6]))
    return charges

# 4. Dataset class

class PkaGraphDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        # Molecular graph
        mol = Chem.MolFromSmiles(s['smiles'])
        charges = parse_charges(s['itp'])
        feats = []
        for atom, ch in zip(mol.GetAtoms(), charges):
            fv = atom_feature_vector(atom)
            fv.append(ch)
            feats.append(fv)
        x = torch.tensor(feats, dtype=torch.float)
        # edges
        edge_list = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edge_list.extend([(i, j), (j, i)])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # target and extras
        y = torch.tensor([s['pka']], dtype=torch.float)
        extra = torch.tensor([ [ s['benson'].get(k,0) for k in all_benson_keys ] + s['sigma'] ], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y, extra=extra)

# 5. Split & loaders

dataset = PkaGraphDataset(samples)

# Split data into train, validation, and test sets
train_val_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_val_set, test_size=0.25, random_state=42) # 0.25 of 0.8 is 0.2

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

# 6. Combined model

class CombinedModel(torch.nn.Module):
    def __init__(self, gcn_in=9, gcn_hidden=64, extra_dim=len(all_benson_keys)+sigma_len,
                 mlp_sizes=[128, 64, 32], dropout=0.2):
        super().__init__()
        # GCN layers (reduced to 3 layers)
        self.conv1 = GCNConv(gcn_in, gcn_hidden)
        self.conv2 = GCNConv(gcn_hidden, gcn_hidden)
        self.conv3 = GCNConv(gcn_hidden, gcn_hidden)
        # MLP head
        dims = [gcn_hidden + extra_dim] + mlp_sizes + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(torch.nn.BatchNorm1d(dims[i+1]))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        emb = global_mean_pool(x, batch)
        extra = data.extra.to(emb.device)
        if extra.dim() == 3:
            extra = extra.squeeze(1)
        out = self.mlp(torch.cat([emb, extra], dim=1))
        return out.view(-1)

# 7. Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(1, 501):
    model.train()
    for data in train_loader:
        data = data.to(device)
        pred = model(data)
        loss = F.mse_loss(pred, data.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                p = model(data)
                all_pred += p.cpu().tolist()
                all_true += data.y.view(-1).cpu().tolist()
        mae = mean_absolute_error(all_true, all_pred)
        rmse = mean_squared_error(all_true, all_pred)**0.5
        r2 = r2_score(all_true, all_pred)
        print(f"Epoch {epoch}: Val MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# 8. Final metrics & plots on validation set

model.eval()
val_pred, val_true = [], []
with torch.no_grad():
    for data in val_loader:
        data = data.to(device)
        p = model(data)
        val_pred += p.cpu().tolist()
        val_true += data.y.view(-1).cpu().tolist()

err = [p - t for p, t in zip(val_pred, val_true)]
abs_err = [abs(e) for e in err]
metrics_val = {
    'MAE': mean_absolute_error(val_true, val_pred),
    'MSE': mean_squared_error(val_true, val_pred),
    'RMSE': mean_squared_error(val_true, val_pred)**0.5,
    'R2': r2_score(val_true, val_pred),
    'Max Abs Error': max(abs_err),
    '%|Err|<=0.2': sum(e<=0.2 for e in abs_err)/len(abs_err)*100,
    '%Err(0,0.2]': sum(0<e<=0.2 for e in err)/len(err)*100,
    '%Err(-0.2,0)': sum(-0.2<e<0 for e in err)/len(err)*100,
    '%|Err|<=0.4': sum(e<=0.4 for e in abs_err)/len(abs_err)*100,
    '%Err(0,0.4]': sum(0<e<=0.4 for e in err)/len(err)*100,
    '%Err(-0.4,0)': sum(-0.4<e<0 for e in err)/len(err)*100
}
print("Final metrics on validation set:")
for k, v in metrics_val.items():
    print(f"Val {k}: {v:.3f}")

plt.figure()
plt.scatter(val_true, val_pred)
mn, mx = min(val_true), max(val_true)
plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
plt.xlabel('True pKa (Validation)')
plt.ylabel('Predicted pKa (Validation)')
plt.title('Parity Plot (Validation)')
plt.show()

# 9. Test set evaluation

model.eval()
test_pred, test_true = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        p = model(data)
        test_pred += p.cpu().tolist()
        test_true += data.y.view(-1).cpu().tolist()

err_test = [p - t for p, t in zip(test_pred, test_true)]
abs_err_test = [abs(e) for e in err_test]
metrics_test = {
    'MAE': mean_absolute_error(test_true, test_pred),
    'MSE': mean_squared_error(test_true, test_pred),
    'RMSE': mean_squared_error(test_true, test_pred)**0.5,
    'R2': r2_score(test_true, test_pred),
    'Max Abs Error': max(abs_err_test),
    '%|Err|<=0.2': sum(e<=0.2 for e in abs_err_test)/len(abs_err_test)*100,
    '%Err(0,0.2]': sum(0<e<=0.2 for e in err_test)/len(err_test)*100,
    '%Err(-0.2,0)': sum(-0.2<e<0 for e in err_test)/len(err_test)*100,
    '%|Err|<=0.4': sum(e<=0.4 for e in abs_err_test)/len(abs_err_test)*100,
    '%Err(0,0.4]': sum(0<e<=0.4 for e in err_test)/len(err_test)*100,
    '%Err(-0.4,0)': sum(-0.4<e<0 for e in err_test)/len(err_test)*100
}
print("Final metrics on test set:")
for k, v in metrics_test.items():
    print(f"Test {k}: {v:.3f}")

plt.figure()
plt.scatter(test_true, test_pred)
mn, mx = min(test_true), max(test_true)
plt.plot([mn, mx], [mn, mx], 'k--', lw=2)
plt.xlabel('True pKa (Test)')
plt.ylabel('Predicted pKa (Test)')
plt.title('Parity Plot (Test)')
plt.show()