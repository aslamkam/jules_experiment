import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
# The model file needs to be in the same directory as this script
# Use absolute path by joining with current directory
model_path = os.path.join(os.path.dirname(__file__), 'best_svc_model_sigma_pka_classification_smote.joblib')
model = joblib.load(model_path)

# Paths to datasets
dataset_path = './data/available-amine-pka-remaining-dataset.csv'
sigma_profiles_path = './data/SigmaProfileData/SigmaProfileData'

# Load the remaining amines dataset
amines_df = pd.read_csv(dataset_path)
amines_df = amines_df[['ID', 'pka_value']]

# Create binary classification target
amines_df['pka_class'] = (amines_df['pka_value'] > 7.0).astype(int)

# Function to load sigma profile values
def load_sigma_profile(file_path):
    try:
        profile_data = pd.read_csv(file_path, sep='\t', header=None)
        return profile_data[1].values
    except Exception as e:
        return None

# Aggregate Sigma profile data
sigma_profiles = []
ids_with_profiles = []

for molecule_id in amines_df['ID']:
    file_path = os.path.join(sigma_profiles_path, f'{molecule_id:06d}.txt')
    sigma_profile = load_sigma_profile(file_path)
    if sigma_profile is not None:
        sigma_profiles.append(sigma_profile)
        ids_with_profiles.append(molecule_id)

# Create dataframe of Sigma profiles
sigma_profiles_array = np.array(sigma_profiles)
column_names = [f'sigma_value_{i}' for i in range(sigma_profiles_array.shape[1])]
sigma_profiles_df = pd.DataFrame(sigma_profiles_array, columns=column_names)
sigma_profiles_df['ID'] = ids_with_profiles

# Merge with pKa classification data
merged_df = pd.merge(amines_df, sigma_profiles_df, on='ID')
merged_df = merged_df.dropna()

# Prepare features and target
X = merged_df.drop(columns=['ID', 'pka_value', 'pka_class']).values
y = merged_df['pka_class'].values

# Make predictions
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# Print classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Save classification report
report_dict = classification_report(y, y_pred, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv('evaluation_classification_report.csv', index=True)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['pKa <= 7', 'pKa > 7'],
            yticklabels=['pKa <= 7', 'pKa > 7'])
plt.title('Confusion Matrix - Evaluation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('evaluation_confusion_matrix.png')
plt.close()

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve - Evaluation Set')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('evaluation_roc_curve.png')
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y, y_pred_proba)
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.title('Precision-Recall Curve - Evaluation Set')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('evaluation_precision_recall_curve.png')
plt.close()

# Save predictions to CSV
results_df = merged_df[['ID', 'pka_value', 'pka_class']].copy()
results_df['predicted_class'] = y_pred
results_df['predicted_probability'] = y_pred_proba
results_df.to_csv('evaluation_predictions.csv', index=False)
print("\nResults have been saved to 'evaluation_predictions.csv'") 