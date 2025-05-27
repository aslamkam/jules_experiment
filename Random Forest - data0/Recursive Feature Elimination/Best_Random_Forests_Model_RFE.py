import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# Paths to datasets
dataset_path = 'available-amine-pka-dataset.csv'
sigma_profiles_path = '/home/kaslam/scratch/data/SigmaProfileData/SigmaProfileData'
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
    
    # Ensure we have the additional columns we want
    columns_to_keep = ['ID', 'pka_value', 'formula', 'amine_class', 'smiles']
    amines_df = amines_df[columns_to_keep]
    
    # Aggregate Sigma profile data
    sigma_profiles = []
    ids_with_profiles = []
    
    for molecule_id in amines_df['ID']:
        file_path = os.path.join(sigma_profiles_path, f'{molecule_id:06d}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            ids_with_profiles.append(molecule_id)
    
    # Create dataframe of Sigma profiles (only sigma values, no charge density)
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    
    # Convert to float32 to prevent precision issues
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), 
                                   columns=column_names)
    sigma_profiles_df['ID'] = ids_with_profiles
    
    # Merge with pKa data
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='ID')
    
    # Remove any remaining invalid values
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df, column_names

def perform_recursive_feature_elimination(X, y, column_names, n_features_to_select=44):
    """
    Perform Recursive Feature Elimination (RFE) on the Random Forest model
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - column_names: List of feature names
    - n_features_to_select: Number of top features to select
    
    Returns:
    - Selected feature names
    - RFE object for further investigation
    """
    # Create the base Random Forest model
    best_params_df = pd.read_csv(best_params_path)
    best_params = best_params_df.to_dict(orient='records')[0]
    
    # Convert string parameters to appropriate types
    rf_model = RandomForestRegressor(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_features=0.6,
        bootstrap=bool(best_params['bootstrap']),
        random_state=42
    )
    
    # Perform Recursive Feature Elimination
    rfe = RFE(estimator=rf_model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    
    # Get the selected feature names
    selected_features = [column_names[i] for i in range(len(column_names)) if rfe.support_[i]]
    
    # Print out details about feature selection
    print("\nRecursive Feature Elimination Results:")
    print("Selected Features:")
    for feature in selected_features:
        print(feature)
    
    # Create a feature ranking DataFrame
    feature_ranking = pd.DataFrame({
        'feature': column_names,
        'ranking': rfe.ranking_
    }).sort_values('ranking')
    
    print("\nFeature Ranking:")
    print(feature_ranking)
    
    return selected_features, rfe

def train_and_evaluate_model(X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test, best_params, selected_features=None):
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
    
    # If selected features are provided, subset the training and test data
    if selected_features is not None:
        feature_indices = [column_names.index(feat) for feat in selected_features]
        X_train = X_train[:, feature_indices]
        X_test = X_test[:, feature_indices]
        column_names = selected_features
    
    # Train the model
    best_model.fit(X_train, y_train)

    # Predictions for training and test sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Performance metrics for training set
    print("\nTraining Set Performance:")
    print("MSE:", mean_squared_error(y_train, y_train_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("MAE:", mean_absolute_error(y_train, y_train_pred))
    print("R^2:", r2_score(y_train, y_train_pred))
    
    # Performance metrics for test set
    print("\nTest Set Performance:")
    print("MSE:", mean_squared_error(y_test, y_test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("MAE:", mean_absolute_error(y_test, y_test_pred))
    print("R^2:", r2_score(y_test, y_test_pred))
    
    # Feature importance if features were selected
    if selected_features is not None:
        feature_importance = pd.DataFrame({
            'feature': column_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop Feature Importances:")
        print(feature_importance)
    
    # Create predictions dataframe with metadata
    predictions_df = pd.DataFrame({
        'formula': metadata_test['formula'],
        'amine_class': metadata_test['amine_class'],
        'smiles': metadata_test['smiles'],
        'Actual_pKa': y_test,
        'Predicted_pKa': y_test_pred
    })
    predictions_df.to_csv('pka_predictions_rfe.csv', index=False)
    print("\nPredictions saved to 'pka_predictions_rfe.csv'")
    
    # Visualization: Actual vs Predicted pKa
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    
    # Add dotted lines intersecting at (6,6)
    plt.axhline(y=6, color='g', linestyle=':', lw=2)
    plt.axvline(x=6, color='g', linestyle=':', lw=2)
    
    plt.xlabel('Actual pKa')
    plt.ylabel('Predicted pKa')
    plt.title('Actual vs Predicted pKa')
    plt.legend()
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_pka.png')
    plt.close()

    # Visualization: Residual Plot
    plt.figure(figsize=(10, 8))
    residuals = y_test_pred - y_test
    plt.scatter(y_test_pred, residuals, alpha=0.7, edgecolor='k')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted pKa')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted pKa')
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    plt.close()
    
    return best_model

def main():
    # Load best parameters
    best_params_df = pd.read_csv(best_params_path)
    best_params = best_params_df.to_dict(orient='records')[0]
    
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
    X = merged_df.drop(columns=['ID', 'pka_value', 'formula', 'amine_class', 'smiles']).values
    y = merged_df['pka_value'].values
    metadata = merged_df[['formula', 'amine_class', 'smiles']]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        X_scaled, y, metadata, test_size=0.2, random_state=42
    )
    
    # Perform Recursive Feature Elimination
    selected_features, rfe = perform_recursive_feature_elimination(X_train, y_train, column_names)
    
    # Train and evaluate model with selected features
    train_and_evaluate_model(
        X_train, X_test, y_train, y_test, column_names, 
        metadata_train, metadata_test, best_params, 
        selected_features=selected_features
    )

if __name__ == "__main__":
    main()
