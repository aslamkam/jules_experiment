import os
import re
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DIR_ITP = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Partial-Charges\chembl_AM1BCC_charges\chembl_AM1BCC_charges"
FILE_CSV = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\ChEMBL_amines_12C.csv"

# 1. Load metadata and filter

df = pd.read_csv(FILE_CSV)
df = df.dropna(subset=['CX Basic pKa'])
samples = []
for _, row in df.iterrows():
    cid = row['ChEMBL ID']
    itp_file = os.path.join(DIR_ITP, f"{cid}_GMX.itp")
    if os.path.isfile(itp_file):
        samples.append((cid, row['Smiles'], float(row['CX Basic pKa']), itp_file))
print(f"{len(samples)} samples with valid ITP.")

# 2. Parse partial charges
def parse_charges(itp_path):
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

# 3. Dataset definition
class PkaDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        cid, smiles, pka, itp = self.samples[idx]
        mol = Chem.MolFromSmiles(smiles)
        charges = parse_charges(itp)
        x = torch.tensor(charges, dtype=torch.float).unsqueeze(1)
        edges = [[], []]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges[0] += [i, j]
            edges[1] += [j, i]
        edge_index = torch.tensor(edges, dtype=torch.long)
        y = torch.tensor([pka], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

# 4. Prepare data loaders

dataset = PkaDataset(samples)
train_n = int(0.8 * len(dataset))
val_n = len(dataset) - train_n
train_set, val_set = torch.utils.data.random_split(dataset, [train_n, val_n])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# 5. Define GCN model with three hidden layers
class GCNModel(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# 6. Training and evaluation functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Single epoch runner

def run_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()
    preds, targets = [], []
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        loss = F.mse_loss(out, batch.y.view(-1))
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds.extend(out.cpu().tolist())
        targets.extend(batch.y.view(-1).cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return preds, targets, avg_loss

# 7. Training loop

for epoch in range(1, 501):
    run_epoch(train_loader, training=True)
    if epoch % 10 == 0:
        p_val, t_val, _ = run_epoch(val_loader, training=False)
        mae = mean_absolute_error(t_val, p_val)
        mse = mean_squared_error(t_val, p_val)
        rmse = mse ** 0.5
        r2 = r2_score(t_val, p_val)
        print(f"Epoch {epoch}: Val MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# 8. Final evaluation with extended metrics

p_val, t_val, _ = run_epoch(val_loader, training=False)
errors = [p - t for p, t in zip(p_val, t_val)]
abs_errors = [abs(e) for e in errors]

metrics = {
    'Max Abs Error': max(abs_errors),
    '% |Err|≤0.2': sum(e <= 0.2 for e in abs_errors) / len(abs_errors) * 100,
    '% Err in (0,0.2]': sum(0 < e <= 0.2 for e in errors) / len(errors) * 100,
    '% Err in (-0.2,0)': sum(-0.2 < e < 0 for e in errors) / len(errors) * 100,
    '% |Err|≤0.4': sum(e <= 0.4 for e in abs_errors) / len(abs_errors) * 100,
    '% Err in (0,0.4]': sum(0 < e <= 0.4 for e in errors) / len(errors) * 100,
    '% Err in (-0.4,0)': sum(-0.4 < e < 0 for e in errors) / len(errors) * 100,
    'MAE': mean_absolute_error(t_val, p_val),
    'MSE': mean_squared_error(t_val, p_val),
    'RMSE': (mean_squared_error(t_val, p_val)) ** 0.5,
    'R2': r2_score(t_val, p_val)
}
print("Final metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")

# 9. Visualization

plt.figure()
plt.scatter(t_val, p_val)
plt.xlabel('True pKa')
plt.ylabel('Predicted pKa')
plt.title('Predicted vs True pKa')
plt.show()

plt.figure()
plt.hist(errors, bins=20)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.show()
