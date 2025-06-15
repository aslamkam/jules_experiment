import os
import ast
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import sys

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

# Define relative paths
script_dir = os.path.dirname(__file__) # Assuming __file__ is defined; for robustness: os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(script_dir, "output.txt")

base_dir = os.path.join(script_dir, '..', '..') # Go up two levels

benson_path = os.path.join(base_dir, "Features", "Chembl-12C", "Benson-Groups", "ChEMBL_amines_12C.xlsx")
csv_path = os.path.join(base_dir, "Features", "Chembl-12C", "ChEMBL_amines_12C.csv")

# Redirect stdout
original_stdout = sys.stdout
output_file_handle = None

try:
    output_file_handle = open(output_file_path, 'w')
    sys.stdout = output_file_handle

    # Load data
    df_benson = pd.read_excel(benson_path)
df_csv = pd.read_csv(csv_path)

# Pre-process Benson data
df_benson['features'] = df_benson['benson_groups'].apply(parse_benson_groups)
df_benson = df_benson[['Smiles', 'features']]

# Pre-process CSV data
df_csv = df_csv[['Smiles', 'CX Basic pKa']]
df_csv = df_csv.rename(columns={'CX Basic pKa': 'pka_value'})

# Merge DataFrames
df = pd.merge(df_csv, df_benson, on='Smiles', how='inner')

# Handle Missing Values
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

# Train-validation-test split indices
all_indices = np.arange(len(graphs))

# Split into train_val and test
train_val_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

# Split train_val into train and validation
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42) # 0.2 of 0.8 is 0.16 for validation

train_graphs = [graphs[i] for i in train_idx]
train_groups = groups[train_idx]
train_y = targets[train_idx]

val_graphs = [graphs[i] for i in val_idx]
val_groups = groups[val_idx]
val_y = targets[val_idx]

test_graphs = [graphs[i] for i in test_idx]
test_groups = groups[test_idx]
test_y = targets[test_idx]

train_loader = DataLoader(list(zip(train_graphs, train_groups, train_y)), batch_size=32, shuffle=True)
val_loader = DataLoader(list(zip(val_graphs, val_groups, val_y)), batch_size=32)
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

def print_metrics(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {set_name} Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")

    errors = y_pred - y_true
    # print(f"{set_name} Errors (predicted - true): {errors.tolist()}") # Optional: can be very verbose
    abs_errors = np.abs(errors)

    if len(abs_errors) > 0:
        max_abs_error = np.max(abs_errors)
        pct_err_le_02 = np.sum(abs_errors <= 0.2) / len(abs_errors) * 100
        pct_err_gt0_le_02 = np.sum((errors > 0) & (errors <= 0.2)) / len(errors) * 100
        pct_err_gt_neg02_lt0 = np.sum((errors > -0.2) & (errors < 0)) / len(errors) * 100
        pct_err_le_04 = np.sum(abs_errors <= 0.4) / len(abs_errors) * 100
        pct_err_gt0_le_04 = np.sum((errors > 0) & (errors <= 0.4)) / len(errors) * 100
        pct_err_gt_neg04_lt0 = np.sum((errors > -0.4) & (errors < 0)) / len(errors) * 100

        print(f"Max Abs Error: {max_abs_error:.4f}")
        print(f"% |Err|<=0.2: {pct_err_le_02:.3f}%")
        print(f"% Err in (0,0.2]: {pct_err_gt0_le_02:.3f}%")
        print(f"% Err in (-0.2,0): {pct_err_gt_neg02_lt0:.3f}%")
        print(f"% |Err|<=0.4: {pct_err_le_04:.3f}%")
        print(f"% Err in (0,0.4]: {pct_err_gt0_le_04:.3f}%")
        print(f"% Err in (-0.4,0): {pct_err_gt_neg04_lt0:.3f}%")
    else:
        print("No errors to calculate additional metrics.")
    print("-------------------------")

print("Using device:", device)
if device.type == 'cuda':
    print(f"  --> GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  --> Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  --> Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Run training
epochs=1000
train_losses = []
val_losses = []

for ep in range(1, epochs+1):
    train_loss = train()
    train_losses.append(train_loss)

    if ep % 10 == 0 or ep == epochs:
        y_val_true, y_val_pred = evaluate(val_loader)
        # Ensure y_val_true is a tensor for criterion
        val_loss = criterion(torch.tensor(y_val_pred), torch.tensor(y_val_true)).item()
        val_losses.append(val_loss)
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        print(f"Epoch {ep}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

# Final evaluation
y_val_true, y_val_pred = evaluate(val_loader)
print_metrics(y_val_true, y_val_pred, "Validation Set")

y_test_true, y_test_pred = evaluate(test_loader)
print_metrics(y_test_true, y_test_pred, "Test Set")

# -------------------------------
# Plotting Results
# -------------------------------

# Plot Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train Loss")
val_loss_epochs = [ep for ep in range(1, epochs + 1) if ep % 10 == 0 or ep == epochs]
if len(val_loss_epochs) == len(val_losses):
    plt.plot(val_loss_epochs, val_losses, label="Validation Loss", marker='o')
else: # Fallback if lengths don't match
    # This case might indicate an issue in how val_losses were collected or epochs variable availability at this stage.
    # For robustness, plot with a simple sequence if epoch numbers are not perfectly aligned.
    plt.plot(np.linspace(1, len(train_losses), len(val_losses)), val_losses, label="Validation Loss (scaled x-axis)", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss Curve - Fusion Model")
plt.legend()
plt.savefig("loss_curve_fusion_model.png")
plt.close()

# Plot Parity and Error Distribution for Validation Set
plt.figure(figsize=(6, 5))
plt.scatter(y_val_true, y_val_pred, alpha=0.6, label='Validation Data')
plt.xlabel("True pKa (Validation Set)")
plt.ylabel("Predicted pKa (Validation Set)")
plt.title("Validation Set: True vs Predicted pKa")
if len(y_val_true) > 0 and len(y_val_pred) > 0:
    min_val = min(np.min(y_val_true), np.min(y_val_pred))
    max_val = max(np.max(y_val_true), np.max(y_val_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
plt.legend()
plt.savefig("parity_plot_validation_set.png")
plt.close()

validation_errors = y_val_pred - y_val_true
plt.figure(figsize=(6, 5))
plt.hist(validation_errors, bins=20)
plt.xlabel('Error (Predicted - True) on Validation Set')
plt.ylabel('Count')
plt.title('Validation Set: Error Distribution')
plt.savefig("error_dist_validation_set.png")
plt.close()

# Plot Parity and Error Distribution for Test Set
plt.figure(figsize=(6, 5))
plt.scatter(y_test_true, y_test_pred, alpha=0.6, label='Test Data')
plt.xlabel("True pKa (Test Set)")
plt.ylabel("Predicted pKa (Test Set)")
plt.title("Test Set: True vs Predicted pKa")
if len(y_test_true) > 0 and len(y_test_pred) > 0:
    min_val = min(np.min(y_test_true), np.min(y_test_pred))
    max_val = max(np.max(y_test_true), np.max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
plt.legend()
plt.savefig("parity_plot_test_set.png")
plt.close()

test_errors = y_test_pred - y_test_true
plt.figure(figsize=(6, 5))
plt.hist(test_errors, bins=20)
plt.xlabel('Error (Predicted - True) on Test Set')
plt.ylabel('Count')
plt.title('Test Set: Error Distribution')
plt.savefig("error_dist_test_set.png")
plt.close()

print("\nPlots generated: loss_curve_fusion_model.png, parity_plot_validation_set.png, error_dist_validation_set.png, parity_plot_test_set.png, error_dist_test_set.png")
print(f"\nAll output successfully written to {output_file_path}")

except Exception as e:
    # Print to original stdout if an error occurs
    print(f"An error occurred: {e}", file=original_stdout)
    # Optionally, log to file if it's open
    if output_file_handle and not output_file_handle.closed:
        print(f"An error occurred during execution: {e}", file=output_file_handle)
    import traceback
    traceback.print_exc(file=original_stdout)
    if output_file_handle and not output_file_handle.closed:
        traceback.print_exc(file=output_file_handle)
    raise # Re-raise the exception

finally:
    if output_file_handle and not output_file_handle.closed:
        output_file_handle.close()
    sys.stdout = original_stdout # Restore original stdout
    print("Execution finished. Standard output was redirected.") # This will print to console
