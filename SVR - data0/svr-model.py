import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance # Commented out for Step 6
import matplotlib.pyplot as plt
import sys

# Define main function to encapsulate script logic
def main():
    # Paths to datasets
    dataset_path = '../../Features/Chembl-12C/ChEMBL_amines_12C.csv'  # Update with correct path if needed
    sigma_profiles_path = '../../Features/Chembl-12C/Orca-Sigma-Profile/ChEMBL_12C_SigmaProfiles_Orca-5899'  # Update with correct path if needed

# dataset_path = '/home/kaslam/scratch/data/available-amine-pka-dataset.csv'
# sigma_profiles_path = '/home/kaslam/scratch/data/SigmaProfileData/SigmaProfileData'

# Load the amines pKa dataset
amines_df = pd.read_csv(dataset_path)
amines_df.rename(columns={'Smiles': 'SMILES'}, inplace=True)
amines_df = amines_df[['ChEMBL ID', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']]

# Modified function to load only sigma profile values
def load_sigma_profile(file_path):
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        # Return only sigma profile values (column 1)
        return profile_data[1].values
    except Exception as e:
        return None

def print_detailed_error_stats(y_true, y_pred, set_name):
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    print(f"\nDetailed Error Statistics for {set_name} Set:")
    if len(errors) == 0:
        print("No data to calculate statistics.")
        return

    print(f"Max Abs Error: {np.max(abs_errors):.4f}")

    for threshold in [0.2, 0.4]:
        within_threshold = abs_errors <= threshold
        print(f"% |Err|<={threshold:.1f}: {np.mean(within_threshold) * 100:.2f}%")

        err_pos_within_threshold = (errors > 0) & (errors <= threshold)
        print(f"% Err in (0,{threshold:.1f}]: {np.sum(err_pos_within_threshold) / len(errors) * 100:.2f}%")

        err_neg_within_threshold = (errors < 0) & (errors >= -threshold)
        print(f"% Err in (-{threshold:.1f},0): {np.sum(err_neg_within_threshold) / len(errors) * 100:.2f}%")

# Aggregate Sigma profile data and merge with amines dataset
sigma_profiles = []
chembl_ids_with_profiles = []

for index, row in amines_df[['ChEMBL ID', 'Inchi Key']].iterrows():
    chembl_id = row['ChEMBL ID']
    inchi_key = row['Inchi Key']
    file_path = os.path.join(sigma_profiles_path, f'{inchi_key}.txt')
    sigma_profile = load_sigma_profile(file_path)
    if sigma_profile is not None:
        sigma_profiles.append(sigma_profile)
        chembl_ids_with_profiles.append(chembl_id)

# Create dataframe of Sigma profiles for molecules with available profiles
sigma_profiles_array = np.array(sigma_profiles)
# Only use sigma profile columns
column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
sigma_profiles_df = pd.DataFrame(sigma_profiles_array, columns=column_names)
sigma_profiles_df['ChEMBL ID'] = chembl_ids_with_profiles

# Merge with pKa data
merged_df = pd.merge(amines_df, sigma_profiles_df, on='ChEMBL ID')

# Handle np.inf and np.nan values
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

# Define features and target
X = merged_df.drop(columns=['ChEMBL ID', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']).values
y = merged_df['CX Basic pKa'].values
metadata = merged_df[['Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']]

# Initial Split (Train/Val + Test)
X_train_val, X_test, y_train_val, y_test, metadata_train_val, metadata_test = train_test_split(
    X, y, metadata, test_size=0.2, random_state=42
)

# Second Split (Train + Validation)
X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
    X_train_val, y_train_val, metadata_train_val, test_size=0.2, random_state=42
)

param_distributions = {
    'svr__C': uniform(10, 100),
    'svr__epsilon': uniform(0.01, 0.1),
    'svr__gamma': ['scale', 'auto'] + list(uniform(0.01, 0.5).rvs(10)),
    'svr__kernel': ['rbf']  # Focus on the RBF kernel based on initial results
}

# Create a pipeline with scaling and SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# Initialize RandomizedSearchCV
n_iter = 100  # Number of parameter settings sampled
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Perform randomized search
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("\nBest Parameters:")
best_params = {key.replace('svr__', ''): value for key, value in random_search.best_params_.items()}
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"\nBest CV Score: {-random_search.best_score_:.4f} MSE")

# Train final model with best parameters
best_model = random_search.best_estimator_

# Predictions for training and test sets
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# Performance metrics for training set
print("\nTraining Set Performance:")
print(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"R^2: {r2_score(y_train, y_train_pred):.4f}")
# Detailed error statistics for Training set can be added if needed

# Performance metrics for validation set
print("\nValidation Set Performance:")
print(f"MSE: {mean_squared_error(y_val, y_val_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_val, y_val_pred):.4f}")
print(f"R^2: {r2_score(y_val, y_val_pred):.4f}")
print_detailed_error_stats(y_val, y_val_pred, "Validation")

# Validation set plots
# Parity Plot for Validation Set
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Actual pKa (Validation Set)')
plt.ylabel('Predicted pKa (Validation Set)')
plt.title('Validation Set: True vs Predicted pKa')
plt.tight_layout()
plt.savefig('parity_plot_validation_set.png')
plt.close()

# Error Distribution Plot for Validation Set
validation_errors = y_val_pred - y_val
plt.figure(figsize=(8, 6))
plt.hist(validation_errors, bins=30, edgecolor='black')
plt.xlabel('Error (Predicted - True) on Validation Set')
plt.ylabel('Count')
plt.title('Validation Set: Error Distribution')
plt.tight_layout()
plt.savefig('error_dist_validation_set.png')
plt.close()

# Performance metrics for test set
print("\nTest Set Performance:")
print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"R^2: {r2_score(y_test, y_test_pred):.4f}")
print_detailed_error_stats(y_test, y_test_pred, "Test")

# Test set plots
# Parity Plot for Test Set
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual pKa (Test Set)')
plt.ylabel('Predicted pKa (Test Set)')
plt.title('Test Set: True vs Predicted pKa')
plt.tight_layout()
plt.savefig('parity_plot_test_set.png')
plt.close()

# Error Distribution Plot for Test Set
test_errors = y_test_pred - y_test
plt.figure(figsize=(8, 6))
plt.hist(test_errors, bins=30, edgecolor='black')
plt.xlabel('Error (Predicted - True) on Test Set')
plt.ylabel('Count')
plt.title('Test Set: Error Distribution')
plt.tight_layout()
plt.savefig('error_dist_test_set.png')
plt.close()

# Save test set predictions
# Ensure metadata_test indices are reset if y_test and y_test_pred are numpy arrays without original indices
metadata_test_for_saving = metadata_test.reset_index(drop=True)
predictions_df = pd.DataFrame({
    'Molecular Formula': metadata_test_for_saving['Molecular Formula'],
    'Amine Class': metadata_test_for_saving['Amine Class'],
    'SMILES': metadata_test_for_saving['SMILES'],
    'Inchi Key': metadata_test_for_saving['Inchi Key'],
    'Actual_pKa': y_test,
    'Predicted_pKa': y_test_pred
})
predictions_df.to_csv('pka_predictions_svr.csv', index=False)
print("\nPredictions saved to 'pka_predictions_svr.csv'")

# # Feature importance analysis using permutation importance # Commented out for Step 6
# print("\nCalculating permutation feature importances...") # Commented out for Step 6
# result = permutation_importance( # Commented out for Step 6
#     best_model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=30, random_state=42, n_jobs=-1 # Commented out for Step 6
# ) # Commented out for Step 6
# importances = result.importances_mean # Commented out for Step 6
# feature_importance = pd.DataFrame({ # Commented out for Step 6
#     'feature': column_names, # Commented out for Step 6
#     'importance': importances # Commented out for Step 6
# }) # Commented out for Step 6
# print("\nTop 10 Most Important Features (Permutation Importance):") # Commented out for Step 6
# print(feature_importance.sort_values('importance', ascending=False).head(10)) # Commented out for Step 6

# Save the best model parameters for future reference
import joblib

model_path = 'best_svr_model.joblib' # Updated filename for Step 6
joblib.dump(best_model, model_path)
print(f"\nBest SVR model saved to '{model_path}'")

    # Optionally, save the best parameters to a CSV file
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv('best_svr_parameters.csv', index=False) # Updated filename for Step 6
    print("Best parameters saved to 'best_svr_parameters.csv'") # Updated print message for Step 6

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, "output.txt")
    original_stdout = sys.stdout

    try:
        with open(output_file_path, 'w') as f:
            sys.stdout = f
            main()
    finally:
        # Ensure stdout is restored even if it was not a file object (e.g. during testing)
        if hasattr(sys.stdout, 'close') and callable(sys.stdout.close) and sys.stdout is not original_stdout:
            sys.stdout.close()
        sys.stdout = original_stdout
