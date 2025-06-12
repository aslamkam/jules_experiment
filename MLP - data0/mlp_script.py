import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from scipy.stats import randint, uniform # Not used with current param_distributions
from sklearn.preprocessing import StandardScaler

# Paths to datasets
dataset_path = '../../Features/Chembl-12C/ChEMBL_amines_12C.csv'
sigma_profiles_path = '../../Features/Chembl-12C/Orca-Sigma-Profile/ChEMBL_12C_SigmaProfiles_Orca-5899'

def load_sigma_profile(file_path):
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        # We only want the sigma profile values (column 1), not the charge density
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def preprocess_data():
    # Load the amines pKa dataset with correct column names
    amines_df = pd.read_csv(dataset_path)
    amines_df.rename(columns={'Smiles': 'SMILES'}, inplace=True)
    
    # Ensure we have the additional columns we want
    columns_to_keep = ['ChEMBL ID', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']
    amines_df = amines_df[columns_to_keep]
    
    # Aggregate Sigma profile data
    sigma_profiles = []
    ids_with_profiles = []
    
    for chembl_id, inchi_key in amines_df[['ChEMBL ID', 'Inchi Key']].values:
        file_path = os.path.join(sigma_profiles_path, f'{inchi_key}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            ids_with_profiles.append(chembl_id) # Store ChEMBL ID
    
    # Create dataframe of Sigma profiles (only sigma values, no charge density)
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    
    # Convert to float32 to prevent precision issues
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), 
                                   columns=column_names)
    sigma_profiles_df['ChEMBL ID'] = ids_with_profiles # Use ChEMBL ID for merging
    
    # Merge with pKa data
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='ChEMBL ID')
    
    # Remove any remaining invalid values
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df, column_names

def train_and_evaluate_model(X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test, X_val, y_val, metadata_val):
    # Define the parameters for MLP (currently set for a single run, not a search)
    param_distributions = {
        'hidden_layer_sizes': [(100,)], # Example: single layer of 100 neurons
        'activation': ['logistic'],    # Sigmoid activation function
        'solver': ['adam'],            # Adam optimizer
        'alpha': [0.001],              # L2 regularization term
        'learning_rate': ['adaptive'], # Adaptive learning rate
        'max_iter': [1000],            # Maximum number of iterations
        'early_stopping': [True],      # Enable early stopping
        'n_iter_no_change': [20]       # Number of iterations with no improvement to trigger early stopping
    }
    
    # Initialize model
    base_model = MLPRegressor(random_state=42)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=1,
        scoring='neg_mean_squared_error',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Perform search
    print("Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest Parameters Found (based on CV within RandomizedSearchCV):")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    # Note: RandomizedSearchCV performs cross-validation. The 'Best CV Score' is based on that.
    # If n_iter=1, it's just one specific parameter combination being validated.
    print(f"\nBest CV Score (Negative MSE): {random_search.best_score_:.4f}") # random_search.best_score_ is neg_mean_squared_error
    print(f"Equivalent Best CV MSE: {-random_search.best_score_:.4f}")
    
    # Retrieve the best model trained by RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predictions for training, validation, and test sets
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Performance metrics for training set
    print("\nTraining Set Performance:")
    print("MSE:", mean_squared_error(y_train, y_train_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("MAE:", mean_absolute_error(y_train, y_train_pred))
    print("R^2:", r2_score(y_train, y_train_pred))

    # Performance metrics for validation set
    print("\nValidation Set Performance:")
    print("MSE:", mean_squared_error(y_val, y_val_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))
    print("MAE:", mean_absolute_error(y_val, y_val_pred))
    print("R^2:", r2_score(y_val, y_val_pred))

    # Detailed error statistics for validation set
    validation_errors = y_val_pred - y_val
    abs_validation_errors = np.abs(validation_errors)

    # Validation Set Parity Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.7)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('Actual pKa (Validation Set)')
    plt.ylabel('Predicted pKa (Validation Set)')
    plt.title('Validation Set: True vs Predicted pKa')
    plt.tight_layout()
    plt.savefig('parity_plot_validation_set.png')
    plt.close()

    # Validation Set Error Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(validation_errors, bins=30, alpha=0.7)
    plt.xlabel('Error (Predicted - True) on Validation Set')
    plt.ylabel('Count')
    plt.title('Validation Set: Error Distribution')
    plt.tight_layout()
    plt.savefig('error_dist_validation_set.png')
    plt.close()

    if len(abs_validation_errors) > 0:
        print("Max Abs Error (Validation):", np.max(abs_validation_errors))
        print("% |Err|<=0.2 (Validation):", np.sum(abs_validation_errors <= 0.2) / len(abs_validation_errors) * 100)
        print("% Err in (0,0.2] (Validation):", np.sum((validation_errors > 0) & (validation_errors <= 0.2)) / len(validation_errors) * 100)
        print("% Err in (-0.2,0) (Validation):", np.sum((validation_errors < 0) & (validation_errors >= -0.2)) / len(validation_errors) * 100)
        print("% |Err|<=0.4 (Validation):", np.sum(abs_validation_errors <= 0.4) / len(abs_validation_errors) * 100)
        print("% Err in (0,0.4] (Validation):", np.sum((validation_errors > 0) & (validation_errors <= 0.4)) / len(validation_errors) * 100)
        print("% Err in (-0.4,0) (Validation):", np.sum((validation_errors < 0) & (validation_errors >= -0.4)) / len(validation_errors) * 100)
    else:
        print("Max Abs Error (Validation): 0.0")
        print("% |Err|<=0.2 (Validation): 0.0")
        print("% Err in (0,0.2] (Validation): 0.0")
        print("% Err in (-0.2,0) (Validation): 0.0")
        print("% |Err|<=0.4 (Validation): 0.0")
        print("% Err in (0,0.4] (Validation): 0.0")
        print("% Err in (-0.4,0) (Validation): 0.0")

    # Performance metrics for test set
    print("\nTest Set Performance:")
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("MAE:", mean_absolute_error(y_test, y_test_pred))
    print("R^2:", r2_score(y_test, y_test_pred))

    # Detailed error statistics for test set
    test_errors = y_test_pred - y_test
    abs_test_errors = np.abs(test_errors)
    if len(abs_test_errors) > 0:
        print("Max Abs Error (Test):", np.max(abs_test_errors))
        print("% |Err|<=0.2 (Test):", np.sum(abs_test_errors <= 0.2) / len(abs_test_errors) * 100)
        print("% Err in (0,0.2] (Test):", np.sum((test_errors > 0) & (test_errors <= 0.2)) / len(test_errors) * 100)
        print("% Err in (-0.2,0) (Test):", np.sum((test_errors < 0) & (test_errors >= -0.2)) / len(test_errors) * 100)
        print("% |Err|<=0.4 (Test):", np.sum(abs_test_errors <= 0.4) / len(abs_test_errors) * 100)
        print("% Err in (0,0.4] (Test):", np.sum((test_errors > 0) & (test_errors <= 0.4)) / len(test_errors) * 100)
        print("% Err in (-0.4,0) (Test):", np.sum((test_errors < 0) & (test_errors >= -0.4)) / len(test_errors) * 100)
    else:
        print("Max Abs Error (Test): 0.0")
        print("% |Err|<=0.2 (Test): 0.0")
        print("% Err in (0,0.2] (Test): 0.0")
        print("% Err in (-0.2,0) (Test): 0.0")
        print("% |Err|<=0.4 (Test): 0.0")
        print("% Err in (0,0.4] (Test): 0.0")
        print("% Err in (-0.4,0) (Test): 0.0")
    
    # Create predictions dataframe with metadata
    predictions_df = pd.DataFrame({
        'Molecular Formula': metadata_test['Molecular Formula'],
        'Amine Class': metadata_test['Amine Class'],
        'SMILES': metadata_test['SMILES'],
        'Inchi Key': metadata_test['Inchi Key'],
        'Actual_pKa': y_test,
        'Predicted_pKa': y_test_pred
    })
    predictions_df.to_csv('pka_predictions_mlp.csv', index=False)
    print("\nPredictions saved to 'pka_predictions_mlp.csv'")
    
    # Visualization: Actual vs Predicted pKa (Test Set)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual pKa (Test Set)')
    plt.ylabel('Predicted pKa (Test Set)')
    plt.title('Test Set: True vs Predicted pKa')
    plt.tight_layout()
    plt.savefig('parity_plot_test_set.png') # Updated filename
    plt.close()

    # Test Set Error Distribution Plot
    plt.figure(figsize=(10, 6))
    plt.hist(test_errors, bins=30, alpha=0.7)
    plt.xlabel('Error (Predicted - True) on Test Set')
    plt.ylabel('Count')
    plt.title('Test Set: Error Distribution')
    plt.tight_layout()
    plt.savefig('error_dist_test_set.png')
    plt.close()
    
    # Save parameters
    best_params_df = pd.DataFrame([random_search.best_params_])
    best_params_df.to_csv('best_mlp_parameters.csv', index=False)
    print("\nBest parameters saved to 'best_mlp_parameters.csv'")
    
    return best_model, None  # Returning None instead of feature importance as MLPs don't have feature importance like Random Forests

def main():
    # Preprocess data
    print("Loading and preprocessing data...")
    merged_df, column_names = preprocess_data()
    
    # Prepare features, target, and metadata
    X = merged_df.drop(columns=['ChEMBL ID', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']).values
    y = merged_df['CX Basic pKa'].values
    metadata = merged_df[['Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training/validation and test sets
    X_train_val, X_test, y_train_val, y_test, metadata_train_val, metadata_test = train_test_split(
        X_scaled, y, metadata, test_size=0.2, random_state=42
    )
    
    # Split training/validation set into training and validation sets
    X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
        X_train_val, y_train_val, metadata_train_val, test_size=0.2, random_state=42
    )

    # Train and evaluate model
    best_model, _ = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test,
        X_val, y_val, metadata_val
    )

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, "output.txt")
    original_stdout = sys.stdout

    try:
        with open(output_file_path, 'w') as f:
            sys.stdout = f
            main()
    finally:
        sys.stdout.close() # Close the file stream
        sys.stdout = original_stdout # Restore original stdout
