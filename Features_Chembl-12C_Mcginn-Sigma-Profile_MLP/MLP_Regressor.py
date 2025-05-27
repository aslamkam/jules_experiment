import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
file_path = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Mcginn-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"
file_path = r"C:\Users\kaslam\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Mcginn-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"
df = pd.read_csv(file_path)

# Parse Sigma Profile into numeric features
def parse_sigma(x):
    try:
        arr = np.array(ast.literal_eval(x), dtype=float)
        return arr
    except Exception:
        return np.array([])

sigma_series = df['Sigma Profile'].apply(parse_sigma)
lengths = sigma_series.apply(len)
expected_len = lengths.mode().iloc[0]
valid = (lengths == expected_len) & df['CX Basic pKa'].notna()
X = np.vstack(sigma_series[valid].values)
y = df.loc[valid, 'CX Basic pKa'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Unified MLP definition and param distribution
mlp = MLPRegressor(max_iter=2000, random_state=42)
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (100,50), (50,25,10), (50,50), (100,100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [1e-3, 1e-2, 1e-1]
}

search = RandomizedSearchCV(
    mlp,
    param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# Fit search and select best model
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

train_mae, train_mse, train_rmse, train_r2 = compute_metrics(y_train, y_train_pred)
test_mae, test_mse, test_rmse, test_r2 = compute_metrics(y_test, y_test_pred)

# Save metrics
with open('output.txt', 'w') as f:
    f.write('Training Metrics:\n')
    f.write(f'MAE: {train_mae:.4f}\n')
    f.write(f'MSE: {train_mse:.4f}\n')
    f.write(f'RMSE: {train_rmse:.4f}\n')
    f.write(f'R^2: {train_r2:.4f}\n\n')
    f.write('Test Metrics:\n')
    f.write(f'MAE: {test_mae:.4f}\n')
    f.write(f'MSE: {test_mse:.4f}\n')
    f.write(f'RMSE: {test_rmse:.4f}\n')
    f.write(f'R^2: {test_r2:.4f}\n')

print("Best Parameters:", search.best_params_)
print("Training and test metrics saved to output.txt")

# Parity plots
for split, (y_true, y_pred, fname, title) in zip(
    ['train', 'test'],
    [(y_train, y_train_pred, 'parity_train.png', 'Training Set Parity Plot'),
     (y_test, y_test_pred, 'parity_test.png', 'Test Set Parity Plot')]
):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('True CX Basic pKa')
    plt.ylabel('Predicted CX Basic pKa')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

joblib.dump(best_model, 'best_mlp_model.joblib')
