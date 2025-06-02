import os
import re
import sys # Added for stdout redirection
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdchem
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Added for train/test split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Redirect print to output.txt
script_dir_for_output = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(script_dir_for_output, "output.txt")
# Store the original stdout
original_stdout = sys.stdout
sys.stdout = open(output_file_path, 'w') # Ensure 'w' mode to overwrite

# Paths
dir_itp = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Partial-Charges\chembl_AM1BCC_charges\chembl_AM1BCC_charges"
file_csv = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\ChEMBL_amines_12C.csv"

# 1. Load metadata and filter
df = pd.read_csv(file_csv)
df = df.dropna(subset=['CX Basic pKa'])
samples = []
for _, row in df.iterrows():
    cid = row['ChEMBL ID']
    itp_file = os.path.join(dir_itp, f"{cid}_GMX.itp")
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

# 3. Atom feature function (excluding Gasteiger)
def atom_feature_vector(atom: rdchem.Atom):
    hybridization_map = {
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
        hybridization_map.get(atom.GetHybridization(), -1),
        int(atom.IsInRing()),
        atom.GetMass()
    ]

# 4. Dataset definition
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
        # Build node features: atom features + partial charge
        feats = []
        for atom, ch in zip(mol.GetAtoms(), charges):
            fv = atom_feature_vector(atom)
            fv.append(ch)
            feats.append(fv)
        x = torch.tensor(feats, dtype=torch.float)
        # Build edges
        edges = [[], []]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges[0] += [i, j]
            edges[1] += [j, i]
        edge_index = torch.tensor(edges, dtype=torch.long)
        y = torch.tensor([pka], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

# 5. Prepare data loaders
dataset = PkaDataset(samples)
# First split: 80% for combined train/val, 20% for test
train_val_indices, test_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    random_state=42  # For reproducibility
)
# Second split: 80% of combined for train, 20% for val
train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=0.2, # 0.2 of the 0.8 of total dataset (which is 0.16 of total)
    random_state=42  # For reproducibility
)

train_set = torch.utils.data.Subset(dataset, train_indices)
val_set = torch.utils.data.Subset(dataset, val_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False) # Shuffle is False for val
test_loader = DataLoader(test_set, batch_size=32, shuffle=False) # Shuffle is False for test

# 6. Define GCN model (input_dim = 9 features)
class GCNModel(torch.nn.Module):
    def __init__(self, hidden_dim=64, input_dim=9):
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
        x = global_mean_pool(x, batch)
        return self.fc(x)

# 7. Training and evaluation setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def run_epoch(loader, training=False):
    model.train() if training else model.eval()
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
    return preds, targets, total_loss / len(loader.dataset)

# 8. Training loop
for epoch in range(1, 2):  # CRITICAL: Run for 1 epoch (range(1,2) means epoch will be 1)
    run_epoch(train_loader, training=True)
    # Print validation metrics for the single epoch
    p_val, t_val, loss_val = run_epoch(val_loader, training=False) # Renamed _ to loss_val for clarity
    mae = mean_absolute_error(t_val, p_val)
    mse = mean_squared_error(t_val, p_val)
    rmse = mse ** 0.5
    r2 = r2_score(t_val, p_val)
    print(f"Epoch {epoch}: Val MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Val Loss={loss_val:.3f}")

# 9. Final evaluation with extended metrics
p_val, t_val, _ = run_epoch(val_loader, training=False)
errors = [p - t for p, t in zip(p_val, t_val)]
abs_errors = [abs(e) for e in errors]
metrics = {
    'Max Abs Error': max(abs_errors),
    '% |Err|<=0.2': sum(e <= 0.2 for e in abs_errors) / len(abs_errors) * 100,
    '% Err in (0,0.2]': sum(0 < e <= 0.2 for e in errors) / len(errors) * 100,
    '% Err in (-0.2,0)': sum(-0.2 < e < 0 for e in errors) / len(errors) * 100,
    '% |Err|<=0.4': sum(e <= 0.4 for e in abs_errors) / len(abs_errors) * 100,
    '% Err in (0,0.4]': sum(0 < e <= 0.4 for e in errors) / len(errors) * 100,
    '% Err in (-0.4,0)': sum(-0.4 < e < 0 for e in errors) / len(errors) * 100,
    'MAE': mean_absolute_error(t_val, p_val),
    'MSE': mean_squared_error(t_val, p_val),
    'RMSE': mean_squared_error(t_val, p_val) ** 0.5,
    'R2': r2_score(t_val, p_val)
}
print("\nFinal Validation Metrics:") # Added newline for clarity
for k, v in metrics.items(): print(f"{k}: {v:.3f}")

# Evaluate on Test Set
p_test, t_test, _ = run_epoch(test_loader, training=False) # Use test_loader
test_errors = [p - t for p, t in zip(p_test, t_test)]
test_abs_errors = [abs(e) for e in test_errors]

# Handle cases where test set might be empty
test_metrics = {
    'Max Abs Error': max(test_abs_errors) if test_abs_errors else 0.0,
    '% |Err|<=0.2': (sum(e <= 0.2 for e in test_abs_errors) / len(test_abs_errors) * 100) if test_abs_errors else 0.0,
    '% Err in (0,0.2]': (sum(0 < e <= 0.2 for e in test_errors) / len(test_errors) * 100) if test_errors else 0.0,
    '% Err in (-0.2,0)': (sum(-0.2 < e < 0 for e in test_errors) / len(test_errors) * 100) if test_errors else 0.0,
    '% |Err|<=0.4': (sum(e <= 0.4 for e in test_abs_errors) / len(test_abs_errors) * 100) if test_abs_errors else 0.0,
    '% Err in (0,0.4]': (sum(0 < e <= 0.4 for e in test_errors) / len(test_errors) * 100) if test_errors else 0.0,
    '% Err in (-0.4,0)': (sum(-0.4 < e < 0 for e in test_errors) / len(test_errors) * 100) if test_errors else 0.0,
    'MAE': mean_absolute_error(t_test, p_test) if t_test else 0.0, # Check if t_test is non-empty
    'MSE': mean_squared_error(t_test, p_test) if t_test else 0.0,
    'RMSE': (mean_squared_error(t_test, p_test) ** 0.5) if t_test else 0.0,
    'R2': r2_score(t_test, p_test) if t_test else 0.0
}
print("\nTest Set Metrics:")
for k, v in test_metrics.items(): print(f"{k}: {v:.3f}")

# 10. Visualization
script_dir_for_plots = os.path.dirname(os.path.abspath(__file__))

# Save Validation Plots
plt.figure()
plt.scatter(t_val, p_val)
plt.xlabel('True pKa')
plt.ylabel('Predicted pKa')
plt.title('Validation: Predicted vs True pKa')
plt.savefig(os.path.join(script_dir_for_plots, "parity_plot_validation.png"))
plt.close() # Close plot to free memory

plt.figure()
plt.hist(errors, bins=20)
plt.xlabel('Error (Predicted - True)')
plt.ylabel('Count')
plt.title('Validation: Error Distribution')
plt.savefig(os.path.join(script_dir_for_plots, "error_dist_validation.png"))
plt.close()

# Save Test Plots
plt.figure()
plt.scatter(t_test, p_test)
plt.xlabel('True pKa')
plt.ylabel('Predicted pKa')
plt.title('Test: Predicted vs True pKa')
plt.savefig(os.path.join(script_dir_for_plots, "parity_plot_test.png"))
plt.close()

plt.figure()
plt.hist(test_errors, bins=20)
plt.xlabel('Error (Predicted - True)')
plt.ylabel('Count')
plt.title('Test: Error Distribution')
plt.savefig(os.path.join(script_dir_for_plots, "error_dist_test.png"))
plt.close()

# Close the output file and restore stdout
if sys.stdout.name == output_file_path: # Check if we still have our file handle
    sys.stdout.close()
sys.stdout = original_stdout # Restore original stdout
