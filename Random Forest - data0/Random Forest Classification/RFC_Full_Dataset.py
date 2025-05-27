import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # For model serialization

from imblearn.over_sampling import SMOTE

# Paths to datasets
dataset_path = '../../data/available-amine-pka-dataset-full.csv'
sigma_profiles_path = '../../data/SigmaProfileData/SigmaProfileData'
best_params_path = 'best_rf_parameters.csv'

def load_sigma_profile(file_path):
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        return profile_data[1].values
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def preprocess_data():
    amines_df = pd.read_csv(dataset_path)
    columns_to_keep = ['ID', 'pka_value', 'formula', 'amine_class', 'smiles']
    amines_df = amines_df[columns_to_keep]

    sigma_profiles = []
    ids_with_profiles = []

    for molecule_id in amines_df['ID']:
        file_path = os.path.join(sigma_profiles_path, f'{molecule_id:06d}.txt')
        sigma_profile = load_sigma_profile(file_path)
        if sigma_profile is not None and np.all(np.isfinite(sigma_profile)):
            sigma_profiles.append(sigma_profile)
            ids_with_profiles.append(molecule_id)

    sigma_profiles_array = np.array(sigma_profiles)
    column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
    sigma_profiles_df = pd.DataFrame(sigma_profiles_array.astype(np.float32), columns=column_names)
    sigma_profiles_df['ID'] = ids_with_profiles

    merged_df = pd.merge(amines_df, sigma_profiles_df, on='ID')
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()

    return merged_df, column_names

def train_and_evaluate_model(X_train, X_test, y_train, y_test, column_names, metadata_train, metadata_test, best_params, scaler):
    best_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_split=10,
        min_samples_leaf=8,
        max_features=0.35,
        bootstrap=True,
        class_weight= {0: 7, 1: 1},
        random_state=42
    )

    updated_params = {
        'n_estimators': 500,
        'max_depth': 7,
        'min_samples_split': 10,
        'min_samples_leaf': 8,
        'max_features': 0.35,
        'bootstrap': True,
        'class_weight': {0: 7, 1: 1}
    }

    best_model.fit(X_train, y_train)

    joblib.dump(best_model, 'trained_random_forest_classifier.joblib')
    print("\nModel saved to 'trained_random_forest_classifier.joblib'")

    joblib.dump(scaler, 'feature_scaler.joblib')
    print("Scaler saved to 'feature_scaler.joblib'")

    y_train_pred = (best_model.predict_proba(X_train)[:, 1] > 0.4).astype(int)
    y_test_pred = (best_model.predict_proba(X_test)[:, 1] > 0.4).astype(int)

    print("\nTraining Set Performance:")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print("Classification Report:")
    print(classification_report(y_train, y_train_pred))

    print("\nTest Set Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    predictions_df = pd.DataFrame({
        'formula': metadata_test['formula'],
        'amine_class': metadata_test['amine_class'],
        'smiles': metadata_test['smiles'],
        'Actual_Category': y_test,
        'Predicted_Category': y_test_pred
    })
    predictions_df.to_csv('classification_predictions.csv', index=False)
    print("\nPredictions saved to 'classification_predictions.csv'")

def main():
    # Load best parameters
    best_params_df = pd.read_csv(best_params_path)
    best_params = best_params_df.to_dict(orient='records')[0]

    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['bootstrap'] = bool(best_params['bootstrap'])

    print("Loading and preprocessing data...")
    merged_df, column_names = preprocess_data()

    X = merged_df.drop(columns=['ID', 'pka_value', 'formula', 'amine_class', 'smiles']).values
    y = (merged_df['pka_value'] >= 7.0).astype(int).values  # Binary classification labels
    metadata = merged_df[['formula', 'amine_class', 'smiles']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
        X_scaled, y, metadata, test_size=0.2, random_state=42
    )

    # Apply SMOTE to the training set
    print("\nApplying SMOTE for oversampling...")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Original training set size: {X_train.shape[0]} samples")
    print(f"Resampled training set size: {X_train_resampled.shape[0]} samples")

    # Train and evaluate the model
    train_and_evaluate_model(
        X_train_resampled, X_test, y_train_resampled, y_test, column_names, metadata_train, metadata_test, best_params, scaler
    )


if __name__ == "__main__":
    main()
