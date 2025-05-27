#!/usr/bin/env python3
"""
Train an XGBoost regressor on pKa values using Benson group counts as features,
predict on an external test set, compute metrics vs. Noroozi-pKa, and save predictions.
"""
import pandas as pd
import ast
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

def parse_benson_groups(text):
    """
    Extract the dictionary literal from a string representation of a defaultdict.
    """
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

# Paths to datasets
TRAIN_PATH = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa_and_removed_ccus.xlsx"
TEST_PATH  = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\External Data Set Test\External Data Set\ccus_96_molecules_benson.xlsx"

# Load and prepare training data
df_train = pd.read_excel(TRAIN_PATH)
df_train['features'] = df_train['benson_groups'].apply(parse_benson_groups)
vec = DictVectorizer(sparse=False)
X_train_full = vec.fit_transform(df_train['features'])
y_train_full = df_train['pka_value'].values

# Split into train/test for evaluation of model performance
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Initialize and train XGBoost regressor with tuned hyperparameters
model = XGBRegressor(
    colsample_bytree=0.8,
    gamma=0,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=200,
    reg_alpha=0.1,
    reg_lambda=1,
    subsample=0.7,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate on held-out validation set
y_val_pred = model.predict(X_val)

rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
mse_val  = mean_squared_error(y_val, y_val_pred)
mae_val  = mean_absolute_error(y_val, y_val_pred)
r2_val   = r2_score(y_val, y_val_pred)
print(f"Validation RMSE: {rmse_val:.3f}")
print(f"Validation MSE:  {mse_val:.3f}")
print(f"Validation MAE:  {mae_val:.3f}")
print(f"Validation R^2:   {r2_val:.3f}\n")

# Load and prepare external test data
df_test = pd.read_excel(TEST_PATH)
df_test['features'] = df_test['benson_groups'].apply(parse_benson_groups)
X_test_external = vec.transform(df_test['features'])
# Use Noroozi-pKa as true labels
y_test_external = df_test['Noroozi-pKa'].values

# Predict on external test set
y_ext_pred = model.predict(X_test_external)

test_rmse = mean_squared_error(y_test_external, y_ext_pred, squared=False)
test_mse  = mean_squared_error(y_test_external, y_ext_pred)
test_mae  = mean_absolute_error(y_test_external, y_ext_pred)
test_r2   = r2_score(y_test_external, y_ext_pred)

print(f"External Test RMSE: {test_rmse:.3f}")
print(f"External Test MSE:  {test_mse:.3f}")
print(f"External Test MAE:  {test_mae:.3f}")
print(f"External Test R^2:   {test_r2:.3f}\n")

# Save predictions alongside original data
output_df = df_test.copy()
output_df['predicted_pKa'] = y_ext_pred
cols_to_save = ['Amine', 'Abbreviation', 'SMILES', 'Noroozi-pKa', 'predicted_pKa', 'Inchi Key']
output_df.to_excel('ccus_96_molecules_benson_predicted.xlsx', index=False, columns=cols_to_save)

print("Predictions saved to ccus_96_molecules_benson_predicted.xlsx")
