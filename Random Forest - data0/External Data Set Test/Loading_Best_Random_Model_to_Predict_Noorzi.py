import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# File paths
model_path = "./trained_random_forest_model.joblib"
scaler_path = "./feature_scaler.joblib"
input_csv_path = "../../data/Thomas_Data/benchmark.csv"
sigma_profiles_dir = "../../data/SigmaProfileData/96Molecules"
output_csv_path = "./predicted_noorzi_pka_values.csv"
metrics_output_path = "./prediction_noorzi_metrics.txt"
plot_output_path = "./pka_noorzi_actual_vs_predicted.png"

# Load the model and scaler
model = load(model_path)
scaler = load(scaler_path)

# Load the input CSV file
molecule_data = pd.read_csv(input_csv_path)

# Prepare to collect data for predictions
predictions = []
names_not_found = []

# Iterate through molecules and predict pKa values
for _, row in molecule_data.iterrows():
    name = row["ID"]
    sigma_file_path = os.path.join(sigma_profiles_dir, f"{name}.txt")
    
    if os.path.exists(sigma_file_path):
        sigma_profile = pd.read_csv(sigma_file_path, sep='\s+', header=None, usecols=[1])
        sigma_features = sigma_profile.values.flatten()[:52]  # Ensure only 52 features are used
        
        # Scale features
        scaled_features = scaler.transform([sigma_features])
        
        # Predict pKa
        predicted_pka = model.predict(scaled_features)[0]
        predictions.append(predicted_pka)
    else:
        names_not_found.append(name)
        predictions.append(np.nan)

# Add predictions to the dataframe
molecule_data["predicted_pka_value"] = predictions

# Save predictions to a new CSV file
molecule_data.to_csv(output_csv_path, index=False)

# Filter rows with valid predictions for metric calculations
valid_data = molecule_data.dropna(subset=["predicted_pka_value", "pka_value"])

# Metrics calculation
true_pka = valid_data["pka_value"]
predicted_pka = valid_data["predicted_pka_value"]

mse = mean_squared_error(true_pka, predicted_pka)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_pka, predicted_pka)
r2 = r2_score(true_pka, predicted_pka)

# Save metrics to a text file
with open(metrics_output_path, "w") as metrics_file:
    metrics_file.write(f"MSE: {mse:.4f}\n")
    metrics_file.write(f"RMSE: {rmse:.4f}\n")
    metrics_file.write(f"MAE: {mae:.4f}\n")
    metrics_file.write(f"R^2: {r2:.4f}\n")

# Generate a scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(true_pka, predicted_pka, alpha=0.7, label="Data points")
plt.plot([true_pka.min(), true_pka.max()], [true_pka.min(), true_pka.max()], color="red", linestyle="--", label="Ideal fit")
plt.plot([true_pka.min(), true_pka.max()], [true_pka.min() + 1, true_pka.max() + 1], color='orange', linestyle='--', lw=2, label='+1 Unit', alpha=0.7)
plt.plot([true_pka.min(), true_pka.max()], [true_pka.min() - 1, true_pka.max() - 1], color='orange', linestyle='--', lw=2, label='-1 Unit', alpha=0.7)
plt.xlabel("Actual pKa values")
plt.ylabel("Predicted pKa values")
plt.title("Actual vs Predicted pKa values")
plt.legend()
plt.grid(True)
plt.savefig(plot_output_path)
plt.close()


import plotly.express as px

# Filter valid data for plotting
valid_data = molecule_data.dropna(subset=["predicted_pka_value", "pka_value"])

# Prepare data for plotting
plot_df = valid_data[["ID", "pka_value", "predicted_pka_value"]]

# Create dynamic scatter plot
fig = px.scatter(
    plot_df,
    x="pka_value",
    y="predicted_pka_value",
    text="ID",
    labels={"pka_value": "Actual pKa values", "predicted_pka_value": "Predicted pKa values"},
    title="Actual vs Predicted pKa values (Dynamic)",
    template="plotly"
)

# Add line of equality
fig.add_shape(
    type="line",
    x0=plot_df["pka_value"].min(),
    y0=plot_df["pka_value"].min(),
    x1=plot_df["pka_value"].max(),
    y1=plot_df["pka_value"].max(),
    line=dict(color="red", dash="dash"),
    name="Ideal fit"
)

# Add line of -1.0 pka
fig.add_shape(
    type="line",
    x0=plot_df["pka_value"].min(),
    y0=plot_df["pka_value"].min() - 1,
    x1=plot_df["pka_value"].max(),
    y1=plot_df["pka_value"].max() -1,
    line=dict(color="orange", dash="dash"),
    name="-1 pka"
)

# Add line of +1.0 pka
fig.add_shape(
    type="line",
    x0=plot_df["pka_value"].min(),
    y0=plot_df["pka_value"].min() + 1,
    x1=plot_df["pka_value"].max(),
    y1=plot_df["pka_value"].max() + 1,
    line=dict(color="orange", dash="dash"),
    name="+1 pka"
)

# Update layout for hover
fig.update_traces(textposition="top center")
fig.update_layout(
    hovermode="closest",
    xaxis_title="Actual pKa values",
    yaxis_title="Predicted pKa values",
    showlegend=False
)

# Show plot
fig.show()

# Optionally, save the plot as an HTML file
fig.write_html("dynamic_pka_plot.html")



print(f"Predictions saved to {output_csv_path}")
print(f"Metrics saved to {metrics_output_path}")
print(f"Scatter plot saved to {plot_output_path}")

# Optional: Notify about missing sigma files
if names_not_found:
    print(f"Warning: Sigma files not found for {len(names_not_found)} molecules.")
