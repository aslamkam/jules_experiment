import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler

def sanitize_inchi(inchi):
    """Convert InChI to filename-safe format."""
    return inchi.replace('/', '_')

def load_sigma_profile(file_path):
    """Load sigma profile from file."""
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        # We only want the sigma profile values (column 1), not the charge density
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def process_pka_value(pka_str):
    """Convert pKa string to float, handling ranges by taking their average."""
    try:
        # If it's already a float, return it
        return float(pka_str)
    except (ValueError, TypeError):
        if isinstance(pka_str, str) and 'to' in pka_str:
            try:
                # Split the range and get average
                values = [float(x.strip()) for x in pka_str.split('to')]
                return sum(values) / len(values)
            except (ValueError, TypeError):
                return None
        return None

def preprocess_data(dataset_path, sigma_profiles_path):
    """Load and preprocess the data."""
    # Load the amines pKa dataset
    amines_df = pd.read_csv(dataset_path)
    
    # Keep only necessary columns
    columns_to_keep = ['InChI', 'SMILES', 'pka_value']
    amines_df = amines_df[columns_to_keep]
    
    # Convert pKa values to floats, handling ranges
    amines_df['pka_value'] = amines_df['pka_value'].apply(process_pka_value)
    
    # Remove rows with invalid pKa values
    amines_df = amines_df.dropna(subset=['pka_value'])
    
    # Aggregate Sigma profile data
    sigma_profiles = []
    inchis_with_profiles = []
    
    for inchi in amines_df['InChI']:
        # Convert InChI to filename-safe format
        safe_inchi = sanitize_inchi(inchi)
        file_path = os.path.join(sigma_profiles_path, f'{safe_inchi}.txt')
        
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            inchis_with_profiles.append(inchi)
    
    # Create dataframe of Sigma profiles
    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    
    # Convert to float32 to prevent precision issues
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), 
                                   columns=column_names)
    sigma_profiles_df['InChI'] = inchis_with_profiles
    
    # Merge with pKa data
    merged_df = pd.merge(amines_df, sigma_profiles_df, on='InChI')
    
    # Remove any remaining invalid values
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return merged_df, column_names

def train_and_evaluate_model(X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test):
    """Train and evaluate the Random Forest model."""
    # Define the parameter space
    param_distributions = {
        'n_estimators': randint(100, 1200),
        'max_depth': randint(10, 100),
        'min_samples_split': randint(2, 60),
        'min_samples_leaf': randint(1, 60),
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }
    
    # Initialize model
    base_model = RandomForestRegressor(random_state=42)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=100,  # Reduced from 500 for faster iteration
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
    
    # Evaluate model
    best_model = random_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Performance metrics
    metrics = {
        'train': {
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred)
        }
    }
    
    # Print metrics
    for dataset in ['train', 'test']:
        print(f"\n{dataset.capitalize()} Set Performance:")
        for metric, value in metrics[dataset].items():
            print(f"{metric}: {value:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': column_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'InChI': metadata_test['InChI'],
        'SMILES': metadata_test['SMILES'],
        'Actual_pKa': y_test,
        'Predicted_pKa': y_test_pred
    })
    predictions_df.to_csv('pka_predictions.csv', index=False)
    
    # Visualizations
    create_visualizations(y_test, y_test_pred, feature_importance)
    
    return best_model, feature_importance

def create_visualizations(y_test, y_test_pred, feature_importance):
    """Create and save visualizations."""
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min() + 1, y_test.max() + 1], 'orange', '--', lw=2, label='+1 Unit')
    plt.plot([y_test.min(), y_test.max()], [y_test.min() - 1, y_test.max() - 1], 'orange', '--', lw=2, label='-1 Unit')
    plt.xlabel('Actual pKa')
    plt.ylabel('Predicted pKa')
    plt.title('Actual vs Predicted pKa Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_pka.png')
    plt.close()
    
    # Feature Importance plot
    plt.figure(figsize=(12, 6))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Define paths
    dataset_path = 'amine_molecules.csv'
    sigma_profiles_path = 'sigma_profiles'  # Directory containing sigma profile files
    
    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Preprocess data
    print("Loading and preprocessing data...")
    merged_df, column_names = preprocess_data(dataset_path, sigma_profiles_path)
    
    # Log data statistics
    logger.info(f"Total number of samples after preprocessing: {len(merged_df)}")
    logger.info(f"pKa value range: {merged_df['pka_value'].min():.2f} to {merged_df['pka_value'].max():.2f}")
    logger.info(f"Number of features: {len(column_names)}")
    
    # Prepare features and target
    X = merged_df[column_names].values
    y = merged_df['pka_value'].values
    metadata = merged_df[['InChI', 'SMILES']]
    
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