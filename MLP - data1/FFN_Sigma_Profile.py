import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_sigma_profile(file_path):
    """Load sigma profile from file."""
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_pka_value(pka_str):
    """Convert pKa string to float; if a range is provided, use its average."""
    try:
        return float(pka_str)
    except (ValueError, TypeError):
        if isinstance(pka_str, str) and 'to' in pka_str:
            try:
                values = [float(x.strip()) for x in pka_str.split('to')]
                return sum(values) / len(values)
            except Exception:
                return None
        return None

def preprocess_data(dataset_path, sigma_profiles_path):
    """Load and preprocess dataset and sigma profile files."""
    # Load amines dataset
    amines_df = pd.read_csv(dataset_path)
    columns_to_keep = ['InChI', 'SMILES', 'pka_value', 'InChI_UID']
    amines_df = amines_df[columns_to_keep]
    amines_df['pka_value'] = amines_df['pka_value'].apply(process_pka_value)
    amines_df = amines_df.dropna(subset=['pka_value'])
    
    sigma_profiles = []
    valid_inchis = []
    for inchi_uid in amines_df['InChI_UID']:
        file_path = os.path.join(sigma_profiles_path, f'{inchi_uid}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            valid_inchis.append(inchi_uid)
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), columns=column_names)
    sigma_profiles_df['InChI_UID'] = valid_inchis
    
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='InChI_UID')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df, column_names

# -------------------------------
# Feedforward Model Definition
# -------------------------------

class SigmaFeedForwardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SigmaFeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out.squeeze()

# -------------------------------
# Training and Evaluation Functions
# -------------------------------

def train_epoch(model, optimizer, criterion, X, y, device):
    model.train()
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()
    return loss.item(), preds

def evaluate(model, criterion, X, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(X)
        loss = criterion(preds, y)
    return loss.item(), preds

# -------------------------------
# Main Function for Feedforward Model
# -------------------------------

def main():
    # Update these paths as needed.
    dataset_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\amine_molecules_full_with_UID.csv'
    sigma_profiles_path = r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\sigma_profiles'
    
    print("Loading and preprocessing data for Feedforward model...")
    merged_df, sigma_columns = preprocess_data(dataset_path, sigma_profiles_path)
    
    X_sigma = merged_df[sigma_columns].values
    scaler = StandardScaler()
    X_sigma_scaled = scaler.fit_transform(X_sigma)
    
    y = merged_df['pka_value'].values
    
    # Split data
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_sigma_scaled[train_idx], dtype=torch.float)
    X_test = torch.tensor(X_sigma_scaled[test_idx], dtype=torch.float)
    y_train = torch.tensor(y[train_idx], dtype=torch.float)
    y_test = torch.tensor(y[test_idx], dtype=torch.float)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    input_dim = X_sigma_scaled.shape[1]
    model = SigmaFeedForwardModel(input_dim, hidden_dim=64).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    num_epochs = 2200
    train_losses = []
    test_losses = []
    
    print("Starting training for Feedforward model...")
    for epoch in range(num_epochs):
        train_loss, _ = train_epoch(model, optimizer, criterion, X_train, y_train, device)
        test_loss, preds = evaluate(model, criterion, X_test, y_test, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Calculate evaluation metrics
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    
    print("Evaluation Metrics for Feedforward Model:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    
    # Plot loss curves and scatter plot for predictions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Loss Curve - Feedforward Model")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, preds, alpha=0.6)
    plt.xlabel("True pKa")
    plt.ylabel("Predicted pKa")
    plt.title("True vs Predicted pKa - Feedforward Model")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
