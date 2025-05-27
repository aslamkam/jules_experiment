import pandas as pd
import numpy as np
from joblib import load
import os

# Load models and scalers
mla_model_path = "mla_random_forest_model.joblib"
mlb_model_path = "mlb_random_forest_model.joblib"
mla_scaler_path = "mla_feature_scaler.joblib"
mlb_scaler_path = "mlb_feature_scaler.joblib"

mla_model = load(mla_model_path)
mlb_model = load(mlb_model_path)
mla_scaler = load(mla_scaler_path)
mlb_scaler = load(mlb_scaler_path)

# File paths
sigma_profiles_dir = "../../data/SigmaProfileData/SigmaProfileData"
input_csv_path = "../../data/available-amine-pka-dataset-full.csv"
mla_output_path = "MLa_all_molecules.csv"
mla_filtered_output_path = "MLA_7+_molecules.csv"
mlb_output_path = "MLb_all_molecules.csv"

# Load input data
molecule_data = pd.read_csv(input_csv_path)

# Function to load sigma profile
def load_sigma_profile(molecule_id):
    file_path = os.path.join(sigma_profiles_dir, f"{molecule_id:06d}.txt")
    if os.path.exists(file_path):
        profile = pd.read_csv(file_path, sep='\t', header=None, usecols=[1]).values.flatten()
        return profile[:52]  # Ensure only 52 features
    return None

# Step 1: Predict with MLA model
mla_predictions = []
molecule_ids = []

for _, row in molecule_data.iterrows():
    molecule_id = row['ID']
    sigma_profile = load_sigma_profile(molecule_id)
    if sigma_profile is not None:
        scaled_features = mla_scaler.transform([sigma_profile])
        predicted_pka = mla_model.predict(scaled_features)[0]
        mla_predictions.append(predicted_pka)
        molecule_ids.append(molecule_id)

mla_results = pd.DataFrame({
    "ID": molecule_ids,
    "MLa_Prediction": mla_predictions
})
mla_results = molecule_data.merge(mla_results, on="ID")
mla_results.to_csv(mla_output_path, index=False)

# Step 2: Filter MLA predictions for pKa >= 7
mla_filtered = mla_results[mla_results["MLa_Prediction"] >= 7]
mla_filtered.to_csv(mla_filtered_output_path, index=False)

# Step 3: Predict with MLB model for filtered molecules
mlb_predictions = []
filtered_ids = []

for _, row in mla_filtered.iterrows():
    molecule_id = row['ID']
    sigma_profile = load_sigma_profile(molecule_id)
    if sigma_profile is not None:
        scaled_features = mlb_scaler.transform([sigma_profile])
        predicted_value = mlb_model.predict(scaled_features)[0]
        mlb_predictions.append(predicted_value)
        filtered_ids.append(molecule_id)

mlb_results = pd.DataFrame({
    "ID": filtered_ids,
    "MLb_Prediction": mlb_predictions
})
mlb_results = mla_filtered.merge(mlb_results, on="ID")
mlb_results.to_csv(mlb_output_path, index=False)

print("Pipeline completed successfully.")