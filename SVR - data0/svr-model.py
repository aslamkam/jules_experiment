import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Paths to datasets

dataset_path = '../data/available-amine-pka-dataset.csv'  # Update with correct path if needed
sigma_profiles_path = '../data/SigmaProfileData/SigmaProfileData'  # Update with correct path if needed

# dataset_path = '/home/kaslam/scratch/data/available-amine-pka-dataset.csv'
# sigma_profiles_path = '/home/kaslam/scratch/data/SigmaProfileData/SigmaProfileData'

# Load the amines pKa dataset
amines_df = pd.read_csv(dataset_path)
amines_df = amines_df[['ID', 'pka_value']]

# Modified function to load only sigma profile values
def load_sigma_profile(file_path):
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        # Return only sigma profile values (column 1)
        return profile_data[1].values
    except Exception as e:
        return None

# Aggregate Sigma profile data and merge with amines dataset
sigma_profiles = []
ids_with_profiles = []

for molecule_id in amines_df['ID']:
    file_path = os.path.join(sigma_profiles_path, f'{molecule_id:06d}.txt')
    sigma_profile = load_sigma_profile(file_path)
    if sigma_profile is not None:
        sigma_profiles.append(sigma_profile)
        ids_with_profiles.append(molecule_id)

# Create dataframe of Sigma profiles for molecules with available profiles
sigma_profiles_array = np.array(sigma_profiles)
# Only use sigma profile columns
column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
sigma_profiles_df = pd.DataFrame(sigma_profiles_array, columns=column_names)
sigma_profiles_df['ID'] = ids_with_profiles

# Merge with pKa data
merged_df = pd.merge(amines_df, sigma_profiles_df, on='ID')

# Handle any missing data
merged_df = merged_df.dropna()

# Define features and target
X = merged_df.drop(columns=['ID', 'pka_value']).values
y = merged_df['pka_value'].values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

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
    random_state=20,
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
y_test_pred = best_model.predict(X_test)

# Performance metrics for training set
print("\nTraining Set Performance:")
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("R^2:", r2_score(y_train, y_train_pred))

print("\nTest Set Performance:")
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("R^2:", r2_score(y_test, y_test_pred))

# Feature importance analysis using permutation importance
print("\nCalculating permutation feature importances...")
result = permutation_importance(
    best_model, X_test, y_test, scoring='neg_mean_squared_error', n_repeats=30, random_state=20, n_jobs=-1
)
importances = result.importances_mean
feature_importance = pd.DataFrame({
    'feature': column_names,
    'importance': importances
})
print("\nTop 10 Most Important Features (Permutation Importance):")
print(feature_importance.sort_values('importance', ascending=False).head(10))

# Save the best model parameters for future reference
import joblib

model_path = 'best_svr_model_sigma_only.joblib'
joblib.dump(best_model, model_path)
print(f"\nBest SVR model saved to '{model_path}'")

# Optionally, save the best parameters to a CSV file
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('best_svr_parameters_sigma_only.csv', index=False)
print("Best parameters saved to 'best_svr_parameters_sigma_only.csv'")
