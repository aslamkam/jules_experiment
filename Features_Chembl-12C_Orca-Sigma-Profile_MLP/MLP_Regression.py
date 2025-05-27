import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- load data ---
data_path = r"C:\Users\kaslam\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Orca-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"
df = pd.read_csv(data_path)

def parse_sigma_profile(sp_str):
    pairs = sp_str.split(';')
    return [float(p.split()[1]) for p in pairs if len(p.split()) == 2]

sigma_matrix = df['Sigma Profile'].dropna().apply(parse_sigma_profile).tolist()
X = np.array(sigma_matrix)
y = df.loc[df['Sigma Profile'].notna(), 'CX Basic pKa'].values

# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- define grid ---
param_grid = {
    # all combinations of 3 to 8 hidden layers with decreasing widths
    'hidden_layer_sizes': [
        (100,) * n_layers for n_layers in range(3, 9)
    ],
    'activation': ['relu'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [1e-3, 1e-2, 1e-1]
}

base_mlp = MLPRegressor(max_iter=2000, random_state=42)

# --- grid search (defaults to cv=5 internally) ---
grid = GridSearchCV(
    estimator=base_mlp,
    param_grid=param_grid,
    verbose=2,
    n_jobs=-1,
    scoring='neg_mean_squared_error'  # used only for selecting best_estimator_
)
grid.fit(X_train, y_train)

print("Best parameters (by CV MSE):", grid.best_params_)

# --- evaluate every combination on train & test ---
results = []
for params in ParameterGrid(param_grid):
    # instantiate & fit
    model = MLPRegressor(
        max_iter=2000,
        random_state=42,
        **params
    )
    model.fit(X_train, y_train)
    
    # predict
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)
    
    # compute metrics
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        return mae, mse, r2
    
    tr_mae, tr_mse, tr_r2 = metrics(y_train, y_tr_pred)
    te_mae, te_mse, te_r2 = metrics(y_test,  y_te_pred)
    
    row = {
        **params,
        'Train MAE': tr_mae,
        'Train MSE': tr_mse,
        'Train R2' : tr_r2,
        'Test MAE' : te_mae,
        'Test MSE' : te_mse,
        'Test R2'  : te_r2
    }
    results.append(row)

# --- save to Excel sorted by descending Test MSE ---
df_results = pd.DataFrame(results)
df_results.sort_values('Test MSE', ascending=False, inplace=True)

output_filename = 'all_hyperparam_test_train_metrics.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_results.to_excel(writer, index=False, sheet_name='GridSearchMetrics')

print(f"All hyperparameter metrics saved to {output_filename}")

# --- final parity plots for the best model ---
best_model = grid.best_estimator_
y_train_best = best_model.predict(X_train)
y_test_best  = best_model.predict(X_test)

for name, y_true, y_pred in [
    ('Train', y_train, y_train_best),
    ('Test',  y_test,  y_test_best)
]:
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.xlabel('True CX Basic pKa')
    plt.ylabel('Predicted CX Basic pKa')
    plt.title(f'Parity Plot: {name}')
    plt.tight_layout()
    plt.savefig(f'parity_{name.lower()}.png')
    plt.close()

print("Parity plots saved as parity_train.png and parity_test.png")
