import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load, dump

# File paths
model_path = "./trained_random_forest_model.joblib"
scaler_path = "./feature_scaler.joblib"

input_csv_path = "../../data/available-amine-pka-dataset.csv"
sigma_profiles_dir = "../../data/SigmaProfileData/SigmaProfileData"

external_dataset_input_csv_path = "../../data/Thomas_Data/benchmark.csv"
external_dataset_sigma_profiles_dir = "../../data/SigmaProfileData/96Molecules"

training_output_csv_path = "./training_predicted_pka_values.csv"
training_metrics_output_path = "./training_prediction_metrics.txt"
training_plot_output_path = "./training_pka_actual_vs_predicted.png"

validation_output_csv_path = "./validation_predicted_pka_values.csv"
validation_metrics_output_path = "./validation_prediction_metrics.txt"
validation_plot_output_path = "./validation_pka_actual_vs_predicted.png"

external_testset_output_csv_path = "./external_testset_predicted_pka_values.csv"
external_testset_metrics_output_path = "./external_testset_prediction_metrics.txt"
external_testset_plot_output_path = "./external_testset_pka_actual_vs_predicted.png"

linear_regression_model_path = "./linear_regression_model.joblib"

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
    sigma_file_path = os.path.join(sigma_profiles_dir, f"{name:06d}.txt")

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

# Split the dataset into training (80%) and validation (20%) sets
train_data, val_data = train_test_split(molecule_data.dropna(subset=["predicted_pka_value", "pka_value"]),
                                        test_size=0.2, random_state=42)

# Train Linear Regression model on training set
X_train = train_data[["predicted_pka_value"]]
y_train = train_data["pka_value"]
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Save the trained linear regression model
dump(linear_model, linear_regression_model_path)

# Predict on training set
train_predictions = linear_model.predict(X_train)
train_data["linear_predicted_pka"] = train_predictions
train_data.to_csv(training_output_csv_path, index=False)

# Compute and save training metrics
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

with open(training_metrics_output_path, "w") as metrics_file:
    metrics_file.write(f"MSE: {train_mse:.4f}\n")
    metrics_file.write(f"RMSE: {train_rmse:.4f}\n")
    metrics_file.write(f"MAE: {train_mae:.4f}\n")
    metrics_file.write(f"R^2: {train_r2:.4f}\n")

# Generate scatter plot for training set
plt.figure(figsize=(6, 6))
plt.scatter(y_train, train_predictions, alpha=0.7, label="Data points")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color="red", linestyle="--", label="Ideal fit")
plt.xlabel("Actual pKa values")
plt.ylabel("Linear Model Predicted pKa values")
plt.title("Training Set: Actual vs Predicted pKa values")
plt.legend()
plt.grid(True)
plt.savefig(training_plot_output_path)
plt.close()

# Predict on validation set
X_val = val_data[["predicted_pka_value"]]
y_val = val_data["pka_value"]
val_predictions = linear_model.predict(X_val)

print("coef: ")
print(linear_model.coef_)
print("Intercept: ")
print(linear_model.intercept_)

# Save validation predictions
val_data["linear_predicted_pka"] = val_predictions
val_data.to_csv(validation_output_csv_path, index=False)

# Compute and save validation metrics
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

with open(validation_metrics_output_path, "w") as metrics_file:
    metrics_file.write(f"MSE: {val_mse:.4f}\n")
    metrics_file.write(f"RMSE: {val_rmse:.4f}\n")
    metrics_file.write(f"MAE: {val_mae:.4f}\n")
    metrics_file.write(f"R^2: {val_r2:.4f}\n")

# Generate scatter plot for validation set
plt.figure(figsize=(6, 6))
plt.scatter(y_val, val_predictions, alpha=0.7, label="Data points")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color="red", linestyle="--", label="Ideal fit")
plt.xlabel("Actual pKa values")
plt.ylabel("Linear Model Predicted pKa values")
plt.title("Validation Set: Actual vs Predicted pKa values")
plt.legend()
plt.grid(True)
plt.savefig(validation_plot_output_path)
plt.close()

# Load the external dataset
external_data = pd.read_csv(external_dataset_input_csv_path)
external_predictions = []
external_names_not_found = []

# Process the external dataset for predictions
for _, row in external_data.iterrows():
    name = row["ID"]
    sigma_file_path = os.path.join(external_dataset_sigma_profiles_dir, f"{name}.txt")

    if os.path.exists(sigma_file_path):
        sigma_profile = pd.read_csv(sigma_file_path, sep='\s+', header=None, usecols=[1])
        sigma_features = sigma_profile.values.flatten()[:52]  # Ensure only 52 features are used

        # Scale features
        scaled_features = scaler.transform([sigma_features])

        # Predict pKa
        predicted_pka = model.predict(scaled_features)[0]
        external_predictions.append(predicted_pka)
    else:
        external_names_not_found.append(name)
        external_predictions.append(np.nan)

external_data["predicted_pka_value"] = external_predictions

# Predict with Linear Regression on external dataset
external_testset = external_data.dropna(subset=["predicted_pka_value"])
X_test = external_testset[["predicted_pka_value"]]
y_test = external_testset["pka_value"]
test_predictions = linear_model.predict(X_test)

# Save external test set predictions
external_testset["linear_predicted_pka"] = test_predictions
external_testset.to_csv(external_testset_output_csv_path, index=False)

# Compute and save external test set metrics
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

with open(external_testset_metrics_output_path, "w") as metrics_file:
    metrics_file.write(f"MSE: {test_mse:.4f}\n")
    metrics_file.write(f"RMSE: {test_rmse:.4f}\n")
    metrics_file.write(f"MAE: {test_mae:.4f}\n")
    metrics_file.write(f"R^2: {test_r2:.4f}\n")

# Generate scatter plot for external test set
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_predictions, alpha=0.7, label="Data points")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal fit")
plt.xlabel("Actual pKa values")
plt.ylabel("Linear Model Predicted pKa values")
plt.title("External Test Set: Actual vs Predicted pKa values")
plt.legend()
plt.grid(True)
plt.savefig(external_testset_plot_output_path)
plt.close()

print(f"Training predictions saved to {training_output_csv_path}")
print(f"Training metrics saved to {training_metrics_output_path}")
print(f"Validation predictions saved to {validation_output_csv_path}")
print(f"Validation metrics saved to {validation_metrics_output_path}")
print(f"External test set predictions saved to {external_testset_output_csv_path}")
print(f"External test set metrics saved to {external_testset_metrics_output_path}")
