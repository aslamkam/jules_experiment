import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint
from joblib import dump

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib style
plt.style.use('ggplot')


def load_sigma_profile(file_path: Path) -> Optional[np.ndarray]:
    """Load sigma profile from a text file.

    Args:
        file_path: Path to the sigma profile file

    Returns:
        Numpy array of sigma profile values or None if loading fails
    """
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        return profile_data[1].values.astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None


def process_pka_value(pka_str: str) -> Optional[float]:
    """Convert pKa string to float, handling ranges by taking their average.

    Args:
        pka_str: Input pKa value as string

    Returns:
        Processed pKa value as float or None if conversion fails
    """
    try:
        return float(pka_str)
    except (ValueError, TypeError):
        if isinstance(pka_str, str) and 'to' in pka_str:
            try:
                values = [float(x.strip()) for x in pka_str.split('to')]
                return sum(values) / len(values)
            except (ValueError, TypeError):
                logger.warning(f"Failed to process pKa value: {pka_str}")
                return None
        return None


def preprocess_data(dataset_path: Path, sigma_profiles_dir: Path) -> Tuple[pd.DataFrame, list]:
    """Load and preprocess the dataset with sigma profiles.

    Args:
        dataset_path: Path to the main dataset CSV
        sigma_profiles_dir: Directory containing sigma profile text files

    Returns:
        Tuple of merged DataFrame and feature column names
    """
    # Load and filter main dataset
    columns_to_keep = ['InChI', 'SMILES', 'pka_value', 'InChI_UID']
    amines_df = pd.read_csv(dataset_path, usecols=columns_to_keep)

    # Process pKa values
    amines_df['pka_value'] = amines_df['pka_value'].apply(process_pka_value)
    amines_df = amines_df.dropna(subset=['pka_value'])

    # Load sigma profiles
    sigma_profiles = []
    valid_inchis = []

    for inchi_uid in amines_df['InChI_UID']:
        profile_path = sigma_profiles_dir / f'{inchi_uid}.txt'
        profile = load_sigma_profile(profile_path)

        if profile is not None and profile.size > 0 and np.all(np.isfinite(profile)):
            sigma_profiles.append(profile)
            valid_inchis.append(inchi_uid)

    if not sigma_profiles:
        raise ValueError("No valid sigma profiles found")

    # Create features DataFrame
    sigma_array = np.array(sigma_profiles)
    feature_cols = [f'sigma_value_{i}' for i in range(sigma_array.shape[1])]
    sigma_df = pd.DataFrame(sigma_array, columns=feature_cols)
    sigma_df['InChI_UID'] = valid_inchis

    # Merge and clean data
    merged_df = pd.merge(amines_df, sigma_df, on='InChI_UID', how='inner')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"Final dataset contains {len(merged_df)} samples")
    return merged_df, feature_cols


def train_and_evaluate_model(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        metadata: pd.DataFrame
) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    """Train and evaluate Random Forest model with hyperparameter tuning.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        feature_names: List of feature names
        metadata: DataFrame containing molecular identifiers

    Returns:
        Tuple of best model and feature importance DataFrame
    """
    param_dist = {
        'n_estimators': randint(200, 1000),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.8],
        'bootstrap': [True, False],
        'max_samples': [0.6, 0.8, None]
    }

    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=150,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    logger.info("Starting hyperparameter search...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Save best parameters
    pd.DataFrame([search.best_params_]).to_csv('best_params.csv', index=False)
    dump(best_model, 'best_rf_model.joblib')

    # Generate predictions
    preds = {
        'train': best_model.predict(X_train),
        'test': best_model.predict(X_test)
    }

    # Calculate and log metrics
    metrics = {}
    for set_name in ['train', 'test']:
        y_true = y_train if set_name == 'train' else y_test
        metrics[set_name] = {
            'MSE': mean_squared_error(y_true, preds[set_name]),
            'RMSE': np.sqrt(mean_squared_error(y_true, preds[set_name])),
            'MAE': mean_absolute_error(y_true, preds[set_name]),
            'R2': r2_score(y_true, preds[set_name])
        }

    logger.info("\nTraining Metrics:")
    for metric, value in metrics['train'].items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("\nTest Metrics:")
    for metric, value in metrics['test'].items():
        logger.info(f"{metric}: {value:.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv('feature_importances.csv', index=False)

    # Save predictions with metadata
    test_meta = metadata.iloc[X_test.index.get_level_values(0)]  # Assuming index preservation
    pred_df = test_meta.assign(
        Actual_pKa=y_test,
        Predicted_pKa=preds['test']
    )
    pred_df.to_csv('predictions.csv', index=False)

    # Generate visualizations
    generate_visualizations(y_test, preds['test'], importance_df)

    return best_model, importance_df


def generate_visualizations(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_importance: pd.DataFrame
) -> None:
    """Generate and save evaluation visualizations using Matplotlib.

    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        feature_importance: DataFrame with feature importance data
    """
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted pKa Values', fontsize=14)
    plt.xlabel('Actual pKa', fontsize=12)
    plt.ylabel('Predicted pKa', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Prediction Residuals', fontsize=14)
    plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['importance'], align='center', alpha=0.7)
    plt.yticks(y_pos, top_features['feature'])
    plt.title('Top 15 Feature Importances', fontsize=14)
    plt.xlabel('Importance Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    # Configure paths
    data_path = Path(r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\amine_molecules_full_with_UID.csv')
    sigma_dir = Path(r'C:\Users\kamal\OneDrive - University of Guelph\My Research\data1_ML_GSP_from_IuPac\GSP-main\GSP-main\Main\Python\sigma_profiles')

    try:
        # Preprocess data
        logger.info("Preprocessing data...")
        data_df, features = preprocess_data(data_path, sigma_dir)

        # Prepare data splits
        X = data_df[features].values
        y = data_df['pka_value'].values
        metadata = data_df[['InChI', 'SMILES', 'InChI_UID']]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train and evaluate model
        logger.info("Starting model training...")
        model, importance = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, features, metadata
        )

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()