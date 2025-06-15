import pandas as pd
import numpy as np
import os
import sys # Added for output redirection
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib  # Added import for model serialization

# Paths to datasets

dataset_path = '../../Features/Chembl-12C/ChEMBL_amines_12C.csv'
sigma_profiles_path = '../../Features/Chembl-12C/Orca-Sigma-Profile/ChEMBL_12C_SigmaProfiles_Orca-5899'


#dataset_path = '/home/kaslam/scratch/data/available-amine-pka-dataset.csv'
#sigma_profiles_path = '/home/kaslam/scratch/data/SigmaProfileData/SigmaProfileData'

best_params_path = 'best_rf_parameters.csv'

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
            ids_with_profiles.append(chembl_id)
    
    # Create dataframe of Sigma profiles (only sigma values, no charge density)
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    
    # Convert to float32 to prevent precision issues
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), 
                                   columns=column_names)
    sigma_profiles_df['ChEMBL ID'] = ids_with_profiles
    
    # Merge with pKa data
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='ChEMBL ID')
    
    # Remove any remaining invalid values
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df, column_names

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, column_names, metadata_train, metadata_val, metadata_test, best_params, scaler):
    # Create model with best parameters
    best_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=0.6,
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    
    # Train the model
    best_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(best_model, 'trained_random_forest_model.joblib')
    print("\nModel saved to 'trained_random_forest_model.joblib'")
    
    # Save the scaler to allow future preprocessing
    joblib.dump(scaler, 'feature_scaler.joblib')
    print("Scaler saved to 'feature_scaler.joblib'")

    # Predictions for training and test sets
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
    if len(abs_validation_errors) > 0:
        print("Max Abs Error (Validation):", np.max(abs_validation_errors))
        print("% |Err|<=0.2 (Validation):", np.sum(abs_validation_errors <= 0.2) / len(abs_validation_errors) * 100)
        print("% Err in (0,0.2] (Validation):", np.sum((validation_errors > 0) & (validation_errors <= 0.2)) / len(validation_errors) * 100)
        print("% Err in (-0.2,0) (Validation):", np.sum((validation_errors < 0) & (validation_errors >= -0.2)) / len(validation_errors) * 100)
        print("% |Err|<=0.4 (Validation):", np.sum(abs_validation_errors <= 0.4) / len(abs_validation_errors) * 100)
        print("% Err in (0,0.4] (Validation):", np.sum((validation_errors > 0) & (validation_errors <= 0.4)) / len(validation_errors) * 100)
        print("% Err in (-0.4,0) (Validation):", np.sum((validation_errors < 0) & (validation_errors >= -0.4)) / len(validation_errors) * 100)
    else:
        print("Validation set is empty, skipping detailed error statistics.")

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
        print("Test set is empty, skipping detailed error statistics.")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': column_names,
        'importance': best_model.feature_importances_
    })
    print("\nTop 10 Most Important Features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))
    
    # Create predictions dataframe with metadata
    predictions_df = pd.DataFrame({
        'Molecular Formula': metadata_test['Molecular Formula'],
        'Amine Class': metadata_test['Amine Class'],
        'SMILES': metadata_test['SMILES'],
        'Inchi Key': metadata_test['Inchi Key'],
        'Actual_pKa': y_test,
        'Predicted_pKa': y_test_pred
    })
    predictions_df.to_csv('pka_predictions_rf.csv', index=False)
    print("\nPredictions saved to 'pka_predictions_rf.csv'")

    # Validation Set Plots
    if y_val.size > 0:
        # Parity Plot (Validation Set)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_val_pred, alpha=0.7)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('Actual pKa (Validation Set)')
        plt.ylabel('Predicted pKa (Validation Set)')
        plt.title('Validation Set: True vs Predicted pKa')
        plt.tight_layout()
        plt.savefig('parity_plot_validation_set.png')
        plt.close()

        # Error Distribution Plot (Validation Set)
        if validation_errors.size > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(validation_errors, bins=30, alpha=0.7)
            plt.xlabel('Error (Predicted - True) on Validation Set')
            plt.ylabel('Count')
            plt.title('Validation Set: Error Distribution')
            plt.tight_layout()
            plt.savefig('error_dist_validation_set.png')
            plt.close()
    else:
        print("Validation set is empty, skipping validation plots.")

    # Test Set Plots
    if y_test.size > 0:
        # Parity Plot (Test Set)
        plt.figure(figsize=(10, 6)) # Changed figsize to (10,6) to match MLP
        plt.scatter(y_test, y_test_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual pKa (Test Set)')
        plt.ylabel('Predicted pKa (Test Set)')
        plt.title('Test Set: True vs Predicted pKa')
        plt.tight_layout()
        plt.savefig('parity_plot_test_set.png') # Changed filename
        plt.close()

        # Error Distribution Plot (Test Set)
        if test_errors.size > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(test_errors, bins=30, alpha=0.7)
            plt.xlabel('Error (Predicted - True) on Test Set')
            plt.ylabel('Count')
            plt.title('Test Set: Error Distribution')
            plt.tight_layout()
            plt.savefig('error_dist_test_set.png')
            plt.close()
    else:
        print("Test set is empty, skipping test plots.")
    
    return best_model, feature_importance

def main():
    # Load best parameters
    best_params_df = pd.read_csv(best_params_path)
    best_params = best_params_df.to_dict(orient='records')[0]

    # Save the parameters that were actually used for this run
    # (which are the ones loaded from best_rf_parameters.csv)
    used_params_df = pd.DataFrame([best_params])
    used_params_df.to_csv('parameters_used_rf.csv', index=False)
    print("\nParameters used for this run saved to 'parameters_used_rf.csv'")
    
    # Convert string parameters to appropriate types
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['bootstrap'] = bool(best_params['bootstrap'])
    
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
    
    # Split data
    # First split: Train-Validation vs Test
    X_train_val, X_test, y_train_val, y_test, metadata_train_val, metadata_test = train_test_split(
        X_scaled, y, metadata, test_size=0.2, random_state=42
    )
    
    # Second split: Train vs Validation
    X_train, X_val, y_train, y_val, metadata_train, metadata_val = train_test_split(
        X_train_val, y_train_val, metadata_train_val, test_size=0.2, random_state=42 # 0.2 of 0.8 is 0.16 of total
    )

    # Train and evaluate model with loaded parameters
    best_model, feature_importance = train_and_evaluate_model(
        X_train, X_val, X_test, y_train, y_val, y_test, column_names, metadata_train, metadata_val, metadata_test, best_params, scaler
    )

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(script_dir, "output.txt")
    original_stdout = sys.stdout

    try:
        with open(output_file_path, 'w') as f:
            sys.stdout = f
            main()  # Call the main function
    finally:
        sys.stdout.close() # Close the file stream
        sys.stdout = original_stdout # Restore original stdout
    print(f"All console output saved to {output_file_path}")
