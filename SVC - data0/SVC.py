import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve, auc, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths to datasets
dataset_path = './data/available-amine-pka-dataset.csv'
sigma_profiles_path = './data/SigmaProfileData/SigmaProfileData'

# Load the amines pKa dataset
amines_df = pd.read_csv(dataset_path)
amines_df = amines_df[['ID', 'pka_value']]

# Create binary classification target
amines_df['pka_class'] = (amines_df['pka_value'] > 7.0).astype(int)

# Modified function to load only sigma profile values
def load_sigma_profile(file_path):
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
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
column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
sigma_profiles_df = pd.DataFrame(sigma_profiles_array, columns=column_names)
sigma_profiles_df['ID'] = ids_with_profiles

# Merge with pKa classification data
merged_df = pd.merge(amines_df, sigma_profiles_df, on='ID')
merged_df = merged_df.dropna()

# Define features and target
X = merged_df.drop(columns=['ID', 'pka_value', 'pka_class']).values
y = merged_df['pka_class'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=20)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

param_grid = {
    'C': [0.01, 0.1, 1, 10],                # Range of regularization strengths
    'gamma': [0.001, 0.01, 0.1, 1],        # Explore a wider range of gamma values
    'class_weight': [{0: 1, 1: 1}, {0: 1.5, 1: 1}, {0: 2, 1: 1}],  # Test class weights
    'kernel': ['rbf'],                     # Keep RBF kernel for now
    'decision_function_shape': ['ovr']     # Keep the decision function shape
}

# Create a pipeline with scaling and SVC
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True))
])

# Initialize the model
svc = SVC()

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    svc,
    param_grid,
    scoring='f1_weighted',  # Focus on weighted F1 score
    cv=5,                   # 5-fold cross-validation
    verbose=3,
    n_jobs=-1
)

random_search.fit(X_train_smote, y_train_smote)
best_model = random_search.best_estimator_

# Save the best model
model_path = 'best_svc_model_sigma_pka_classification_smote.joblib'
joblib.dump(best_model, model_path)
print(f"\nBest SVC model saved to '{model_path}'")

# Save best parameters
best_params = {key.replace('svc__', ''): value for key, value in random_search.best_params_.items()}
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('best_svc_parameters_sigma_pka_classification_smote.csv', index=False)
print("Best parameters saved to 'best_svc_parameters_sigma_pka_classification_smote.csv'")

# Predictions
y_train_pred = best_model.predict(X_train_smote)
y_test_pred = best_model.predict(X_test)

# Classification Report
train_report = classification_report(y_train_smote, y_train_pred, output_dict=False)
test_report = classification_report(y_test, y_test_pred, output_dict=False)

# Print classification reports to console
print("\nTraining Set Classification Report:")
print(train_report)

print("\nTest Set Classification Report:")
print(test_report)

# Save classification reports as CSV files
train_report_dict = classification_report(y_train_smote, y_train_pred, output_dict=True)
test_report_dict = classification_report(y_test, y_test_pred, output_dict=True)
pd.DataFrame(train_report_dict).transpose().to_csv('training_classification_report_smote.csv', index=True)
pd.DataFrame(test_report_dict).transpose().to_csv('test_classification_report_smote.csv', index=True)
print("Classification reports saved to 'training_classification_report_smote.csv' and 'test_classification_report_smote.csv'")

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['pKa <= 7', 'pKa > 7'],
            yticklabels=['pKa <= 7', 'pKa > 7'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_smote.png')
plt.show()

# ROC Curve
y_test_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve_smote.png')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('precision_recall_curve_smote.png')
plt.show()
