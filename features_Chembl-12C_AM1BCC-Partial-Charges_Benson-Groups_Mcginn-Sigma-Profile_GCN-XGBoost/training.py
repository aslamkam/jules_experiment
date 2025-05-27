import os
import re
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdchem
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import ast
from collections import defaultdict

# Paths
DIR_ITP = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Partial-Charges\chembl_AM1BCC_charges\chembl_AM1BCC_charges"
FILE_PKA = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\ChEMBL_amines_12C.csv"
FILE_BENSON = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Benson-Groups\ChEMBL_amines_12C.xlsx"
FILE_SIGMA = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Mcginn-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"

# 1. Load and merge metadata
df_pka = pd.read_csv(FILE_PKA).dropna(subset=['CX Basic pKa'])
df_benson = pd.read_excel(FILE_BENSON)
df_sigma = pd.read_csv(FILE_SIGMA)

# Inner join to ensure all three have data
df = df_pka.merge(df_benson[['ChEMBL ID', 'benson_groups']], on='ChEMBL ID', how='inner')
df = df.merge(df_sigma[['ChEMBL ID', 'Sigma Profile']], on='ChEMBL ID', how='inner')

# 2. Prepare samples list
samples = []
for _, row in df.iterrows():
    cid = row['ChEMBL ID']
    itp_path = os.path.join(DIR_ITP, f"{cid}_GMX.itp")
    if not os.path.isfile(itp_path):
        continue

    # parse benson groups (string repr of defaultdict)
    bens_val = row['benson_groups']
    benson = {}
    if isinstance(bens_val, str) and 'defaultdict' in bens_val:
        # extract inner dict
        try:
            s = bens_val[bens_val.find('{'):bens_val.rfind('}') + 1]
            benson = ast.literal_eval(s)
        except Exception:
            continue
    elif isinstance(bens_val, dict):
        benson = bens_val
    else:
        continue

    # parse sigma profile
    sigma_val = row['Sigma Profile']
    sigma = []
    if isinstance(sigma_val, str):
        try:
            sigma = ast.literal_eval(sigma_val)
        except Exception:
            continue
    elif isinstance(sigma_val, list):
        sigma = sigma_val
    else:
        continue

    samples.append({
        'id': cid,
        'smiles': row['Smiles'],
        'pka': float(row['CX Basic pKa']),
        'itp': itp_path,
        'benson': benson,
        'sigma': sigma
    })

print(f"Loaded {len(samples)} samples with all required data.")

# 3. Feature helpers
def atom_feature_vector(atom: rdchem.Atom):
    hyb_map = {
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
        hyb_map.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetMass()
    ]

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

# 4. PyG Dataset
class PkaGraphDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        s = self.samples[idx]
        mol = Chem.MolFromSmiles(s['smiles'])
        charges = parse_charges(s['itp'])
        features = []
        for atom, ch in zip(mol.GetAtoms(), charges):
            fv = atom_feature_vector(atom)
            fv.append(ch)
            features.append(fv)
        x = torch.tensor(features, dtype=torch.float)
        edges = [[], []]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges[0] += [i, j]
            edges[1] += [j, i]
        edge_index = torch.tensor(edges, dtype=torch.long)
        y = torch.tensor([s['pka']], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

# 5. Split data and loaders
dataset = PkaGraphDataset(samples)
train_val_set, test_set = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_val_dataset = torch.utils.data.Subset(dataset, train_val_set)
test_dataset = torch.utils.data.Subset(dataset, test_set)

train_n = int(0.8 * len(train_val_dataset))
val_n = len(train_val_dataset) - train_n
train_set_indices, val_set_indices = torch.utils.data.random_split(train_val_set, [train_n, val_n])
train_set = torch.utils.data.Subset(dataset, train_set_indices)
val_set = torch.utils.data.Subset(dataset, val_set_indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        emb = global_mean_pool(x, batch)
        return self.fc(emb), emb

# 7. Train GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = GCNModel().to(device)
optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-3)

for epoch in range(1, 1001):
    gcn.train()
    for batch in train_loader:
        batch = batch.to(device)
        pred, _ = gcn(batch)
        loss = F.mse_loss(pred.view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        gcn.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pv, _ = gcn(batch)
                preds += pv.view(-1).cpu().tolist()
                trues += batch.y.view(-1).cpu().tolist()
        mae = mean_absolute_error(trues, preds)
        rmse = mean_squared_error(trues, preds) ** 0.5
        r2 = r2_score(trues, preds)
        print(f"Epoch {epoch}: Val MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# 8. Extract embeddings for test set
embeddings_test, target_pkas_test = [], []
gcn.eval()
test_loader_emb = DataLoader(test_dataset, batch_size=32)
with torch.no_grad():
    for batch in test_loader_emb:
        batch = batch.to(device)
        _, emb = gcn(batch)
        embeddings_test.extend(emb.cpu().tolist())
        target_pkas_test.extend(batch.y.view(-1).cpu().tolist())

# 9. Aggregate Benson keys
all_keys = sorted({k for s in samples for k in s['benson'].keys()})

# 10. Prepare XGBoost test dataset
X_test_xgb, y_test_xgb = [], []
test_indices = test_dataset.indices
test_samples_correct_order = [dataset.samples[i] for i in test_indices]

for idx, s in enumerate(test_samples_correct_order):
    row_emb = embeddings_test[idx]
    bens = s['benson']
    sigma = s['sigma']
    features = row_emb + [bens.get(k, 0) for k in all_keys] + sigma
    X_test_xgb.append(features)
    y_test_xgb.append(s['pka'])

columns_emb = [f'emb_{i}' for i in range(len(embeddings_test[0]))]
columns_ben = [f'ben_{k}' for k in all_keys]
columns_sig = [f'sig_{i}' for i in range(len(test_samples_correct_order[0]['sigma'])) if 'sigma' in test_samples_correct_order[0]]
columns = columns_emb + columns_ben + columns_sig
df_xgb_test = pd.DataFrame(X_test_xgb, columns=columns)

X = []
y = []
for idx, s in enumerate(samples):
    row_emb = embeddings_test[test_indices.index(idx)] if idx in test_indices else [0] * len(embeddings_test[0]) # Handle cases where a sample might not be in the test set
    bens = s['benson']
    sigma = s['sigma']
    features = row_emb + [bens.get(k, 0) for k in all_keys] + sigma
    X.append(features)
    y.append(s['pka'])
df_xgb_full = pd.DataFrame(X, columns=columns)
df_xgb_full['pka'] = y

X_train_xgb, X_test_xgb_final, y_train_xgb, y_test_xgb_final = train_test_split(df_xgb_full.drop('pka', axis=1), df_xgb_full['pka'], test_size=0.2, random_state=42)

# 11. Train XGBoost
xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train_xgb, y_train_xgb)

# Evaluate on training set
pred_train_xgb = xgb.predict(X_train_xgb)
true_train_xgb = y_train_xgb.values
errors_train = pred_train_xgb - true_train_xgb
abs_errors_train = abs(errors_train)
metrics_train = {
    'MAE': mean_absolute_error(true_train_xgb, pred_train_xgb),
    'MSE': mean_squared_error(true_train_xgb, pred_train_xgb),
    'RMSE': mean_squared_error(true_train_xgb, pred_train_xgb) ** 0.5,
    'R2': r2_score(true_train_xgb, pred_train_xgb),
    'Max Abs Error': max(abs_errors_train),
    '%|Err|<=0.2': sum(abs_errors_train <= 0.2) / len(abs_errors_train) * 100,
    '%Err(0,0.2]': sum((errors_train > 0) & (errors_train <= 0.2)) / len(errors_train) * 100,
    '%Err(-0.2,0)': sum((errors_train < 0) & (errors_train > -0.2)) / len(errors_train) * 100,
    '%|Err|<=0.4': sum(abs_errors_train <= 0.4) / len(abs_errors_train) * 100,
    '%Err(0,0.4]': sum((errors_train > 0) & (errors_train <= 0.4)) / len(errors_train) * 100,
    '%Err(-0.4,0)': sum((errors_train < 0) & (errors_train > -0.4)) / len(errors_train) * 100
}
print("\n--- XGBoost Train Set Results ---")
for key, value in metrics_train.items():
    print(f"{key}: {value:.4f}")


pred_test_xgb = xgb.predict(X_test_xgb_final)
true_test_xgb = y_test_xgb_final.values

# 12. Metrics on test set
errors_test = pred_test_xgb - true_test_xgb
abs_errors_test = abs(errors_test)
metrics_test = {
    'MAE': mean_absolute_error(true_test_xgb, pred_test_xgb),
    'MSE': mean_squared_error(true_test_xgb, pred_test_xgb),
    'RMSE': mean_squared_error(true_test_xgb, pred_test_xgb) ** 0.5,
    'R2': r2_score(true_test_xgb, pred_test_xgb),
    'Max Abs Error': max(abs_errors_test),
    '%|Err|<=0.2': sum(abs_errors_test <= 0.2) / len(abs_errors_test) * 100,
    '%Err(0,0.2]': sum((errors_test > 0) & (errors_test <= 0.2)) / len(errors_test) * 100,
    '%Err(-0.2,0)': sum((errors_test < 0) & (errors_test > -0.2)) / len(errors_test) * 100,
    '%|Err|<=0.4': sum(abs_errors_test <= 0.4) / len(abs_errors_test) * 100,
    '%Err(0,0.4]': sum((errors_test > 0) & (errors_test <= 0.4)) / len(errors_test) * 100,
    '%Err(-0.4,0)': sum((errors_test < 0) & (errors_test > -0.4)) / len(errors_test) * 100
}
print("XGBoost Test Set Metrics:")
print(metrics_test)

# 13. Parity plot for test set
plt.figure()
plt.scatter(true_test_xgb, pred_test_xgb)
plt.plot([min(true_test_xgb), max(true_test_xgb)], [min(true_test_xgb), max(true_test_xgb)], 'k--', lw=2)
plt.xlabel('True pKa (Test Set)')
plt.ylabel('Predicted pKa (Test Set)')
plt.title('Parity Plot (XGBoost on Test Set)')
plt.show()