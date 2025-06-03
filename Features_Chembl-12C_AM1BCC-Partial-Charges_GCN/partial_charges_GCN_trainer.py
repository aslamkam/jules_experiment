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
from sklearn.model_selection import train_test_split
import sys

# Redirect print to output.txt
script_dir_for_output = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(script_dir_for_output, "output.txt")
# Store the original stdout
original_stdout = sys.stdout
sys.stdout = open(output_file_path, 'w') # Ensure 'w' mode to overwrite

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

# First split: 80% for combined train/val, 20% for test
train_val_indices, test_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    random_state=42  # For reproducibility
)

# Second split: 80% of combined for train, 20% for val (which is 16% of total)
train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=0.2, # 0.2 of the 0.8 of total dataset
    random_state=42  # For reproducibility
)

train_set = torch.utils.data.Subset(dataset, train_indices)
val_set = torch.utils.data.Subset(dataset, val_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False) # Shuffle is False for val
test_loader = DataLoader(test_set, batch_size=32, shuffle=False) # Shuffle is False for test

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

def calculate_comprehensive_metrics(t_actual, p_pred, set_name=""):
    if not t_actual or not p_pred or len(t_actual) == 0 or len(p_pred) == 0: # Handle empty or mismatched lists
        print(f"Warning: Empty or invalid data provided for {set_name} set metrics calculation.")
        return {
            'Max Abs Error': 0.0, '% |Err|<=0.2': 0.0, '% Err in (0,0.2]': 0.0,
            '% Err in (-0.2,0)': 0.0, '% |Err|<=0.4': 0.0, '% Err in (0,0.4]': 0.0,
            '% Err in (-0.4,0)': 0.0, 'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'R2': 0.0
        }

    errors = [p - t for p, t in zip(p_pred, t_actual)]
    abs_errors = [abs(e) for e in errors]

    # Ensure no division by zero if errors/abs_errors lists are empty (though checked above)
    len_abs_errors = len(abs_errors) if abs_errors else 1
    len_errors = len(errors) if errors else 1


    metrics_dict = {
        'Max Abs Error': max(abs_errors) if abs_errors else 0.0,
        '% |Err|<=0.2': (sum(e <= 0.2 for e in abs_errors) / len_abs_errors * 100),
        '% Err in (0,0.2]': (sum(0 < e <= 0.2 for e in errors) / len_errors * 100),
        '% Err in (-0.2,0)': (sum(-0.2 < e < 0 for e in errors) / len_errors * 100),
        '% |Err|<=0.4': (sum(e <= 0.4 for e in abs_errors) / len_abs_errors * 100),
        '% Err in (0,0.4]': (sum(0 < e <= 0.4 for e in errors) / len_errors * 100),
        '% Err in (-0.4,0)': (sum(-0.4 < e < 0 for e in errors) / len_errors * 100),
        'MAE': mean_absolute_error(t_actual, p_pred),
        'MSE': mean_squared_error(t_actual, p_pred),
        'RMSE': (mean_squared_error(t_actual, p_pred) ** 0.5),
        'R2': r2_score(t_actual, p_pred)
    }
    return metrics_dict

# 7. Training loop

total_epochs = 501
for epoch in range(1, total_epochs + 1):
    run_epoch(train_loader, training=True)
    if epoch % 10 == 0 or epoch == total_epochs:
        p_val, t_val, loss_val = run_epoch(val_loader, training=False) # Capture loss_val
        mae = mean_absolute_error(t_val, p_val)
        mse = mean_squared_error(t_val, p_val)
        rmse = mse ** 0.5
        r2 = r2_score(t_val, p_val)
        # Updated print statement to include Val Loss and match example format
        print(f"Epoch {epoch}/{total_epochs}: Val MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Val Loss={loss_val:.3f}")

# 8. Final evaluation with extended metrics

# Evaluate on Training Set (Optional, but good practice)
p_train, t_train, _ = run_epoch(train_loader, training=False)
train_metrics = calculate_comprehensive_metrics(t_train, p_train, "Training")
print("\nFinal Training Metrics:")
for k, v in train_metrics.items(): print(f"{k}: {v:.3f}")

# Evaluate on Validation Set
p_val, t_val, _ = run_epoch(val_loader, training=False)
val_metrics = calculate_comprehensive_metrics(t_val, p_val, "Validation")
print("\nFinal Validation Metrics:")
for k, v in val_metrics.items(): print(f"{k}: {v:.3f}")

# Evaluate on Test Set
p_test, t_test, _ = run_epoch(test_loader, training=False) # Use test_loader
test_metrics = calculate_comprehensive_metrics(t_test, p_test, "Test")
print("\nTest Set Metrics:")
for k, v in test_metrics.items(): print(f"{k}: {v:.3f}")

# For plotting validation errors
errors_val = [p - t for p, t in zip(p_val, t_val)]
# For plotting test errors (will be used in next step)
errors_test = [p - t for p, t in zip(p_test, t_test)]

# 9. Visualization
script_dir_for_plots = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(script_dir_for_plots): # Ensure directory exists, though __file__ should make it valid
    os.makedirs(script_dir_for_plots)

# Validation Scatter Plot
plt.figure()
plt.scatter(t_val, p_val)
all_val_data = t_val + p_val
if all_val_data:
    min_val = min(all_val_data)
    max_val = max(all_val_data)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x')
    plt.legend()
plt.xlabel('True pKa')
plt.ylabel('Predicted pKa')
plt.title('Validation: Predicted vs True pKa')
plt.savefig(os.path.join(script_dir_for_plots, "parity_plot_validation.png"))
plt.close()

# Validation Error Histogram
plt.figure()
plt.hist(errors_val, bins=20)
plt.xlabel('Error (Predicted - True)')
plt.ylabel('Frequency') # Changed to Frequency as per typical histograms, Count is also fine.
plt.title('Validation: Error Distribution')
plt.savefig(os.path.join(script_dir_for_plots, "error_dist_validation.png"))
plt.close()

# Test Scatter Plot
plt.figure()
plt.scatter(t_test, p_test)
all_test_data = t_test + p_test
if all_test_data:
    min_val_test = min(all_test_data)
    max_val_test = max(all_test_data)
    plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'k--', lw=2, label='y=x')
    plt.legend()
plt.xlabel('True pKa')
plt.ylabel('Predicted pKa')
plt.title('Test: Predicted vs True pKa')
plt.savefig(os.path.join(script_dir_for_plots, "parity_plot_test.png"))
plt.close()

# Test Error Histogram
plt.figure()
plt.hist(errors_test, bins=20)
plt.xlabel('Error (Predicted - True)')
plt.ylabel('Count')
plt.title('Test: Error Distribution')
plt.savefig(os.path.join(script_dir_for_plots, "error_dist_test.png"))
plt.close()

# Close the output file and restore stdout
if sys.stdout.name == output_file_path: # Check if we still have our file handle
    sys.stdout.close()
sys.stdout = original_stdout # Restore original stdout
