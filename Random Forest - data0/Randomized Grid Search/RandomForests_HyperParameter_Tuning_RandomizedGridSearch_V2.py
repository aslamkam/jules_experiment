import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler

# Paths to datasets
dataset_path = '../../data/available-amine-pka-dataset-full.csv'
sigma_profiles_path = '../../data/SigmaProfileData/SigmaProfileData'

# dataset_path = '/home/kaslam/scratch/data/available-amine-pka-dataset.csv'
# sigma_profiles_path = '/home/kaslam/scratch/data/SigmaProfileData/SigmaProfileData'

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

def train_and_evaluate_model(X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test):
    # Define the parameter space
    param_distributions = {
        'n_estimators': randint(1, 1200),
        'max_depth': randint(1, 100),
        'min_samples_split': randint(1, 60),
        'min_samples_leaf': randint(1, 60),
        'max_features': ['sqrt', 'log'],
        'bootstrap': [True]
    }
    
    # Initialize model
    base_model = RandomForestRegressor(random_state=42)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=500,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Perform search
    print("Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"\nBest CV Score: {-random_search.best_score_:.4f} MSE")
    
    # Evaluate on test set
    best_model = random_search.best_estimator_

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
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': column_names,
        'importance': best_model.feature_importances_
    })
    print("\nTop 10 Most Important Features:")
    print(feature_importance.sort_values('importance', ascending=False).head(10))
    
    # Create predictions dataframe with metadata
    predictions_df = pd.DataFrame({
        'formula': metadata_test['formula'],
        'amine_class': metadata_test['amine_class'],
        'smiles': metadata_test['smiles'],
        'Actual_pKa': y_test,
        'Predicted_pKa': y_test_pred
    })
    predictions_df.to_csv('pka_predictions.csv', index=False)
    print("\nPredictions saved to 'pka_predictions.csv'")
    
    # Visualization: Actual vs Predicted pKa
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min() + 1, y_test.max() + 1], color='orange', linestyle='--', lw=2, label='+1 Unit', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min() - 1, y_test.max() - 1], color='orange', linestyle='--', lw=2, label='-1 Unit', alpha=0.7)
    plt.xlabel('Actual pKa')
    plt.ylabel('Predicted pKa')
    plt.title('Actual vs Predicted pKa Values')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_pka.png')
    plt.close()
    
    # Visualization: Feature Importance (replacing seaborn barplot with matplotlib)
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.sort_values('importance', ascending=False).head(10)
    
    # Create horizontal bar plot
    plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Save parameters
    best_params_df = pd.DataFrame([random_search.best_params_])
    best_params_df.to_csv('best_rf_parameters.csv', index=False)
    print("\nBest parameters saved to 'best_rf_parameters.csv'")
    
    return best_model, feature_importance

def main():
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
    
    # Train and evaluate model
    best_model, feature_importance = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test
    )

if __name__ == "__main__":
    main()
