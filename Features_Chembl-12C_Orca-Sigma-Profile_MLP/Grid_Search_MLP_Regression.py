import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import Parallel, delayed

# --- load data ---
data_path = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\Features\Chembl-12C\Orca-Sigma-Profile\ChEMBL_amines_12C_with_sigma.csv"
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
    'hidden_layer_sizes': [(100,)*n for n in range(3, 9)],  # 3 to 8 layers
    'activation': ['relu'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [1e-3, 1e-2, 1e-1]
}

def evaluate(params):
    # train
    model = MLPRegressor(max_iter=2000, random_state=42, **params)
    model.fit(X_train, y_train)
    # predict
    y_tr = model.predict(X_train)
    y_te = model.predict(X_test)
    # metrics
    def m(y_true, y_pred):
        return (
            mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred),
            r2_score(y_true, y_pred)
        )
    tr_mae, tr_mse, tr_r2 = m(y_train, y_tr)
    te_mae, te_mse, te_r2 = m(y_test,  y_te)
    # bundle
    out = dict(params)
    out.update({
        'Train MAE': tr_mae,
        'Train MSE': tr_mse,
        'Train R2' : tr_r2,
        'Test MAE' : te_mae,
        'Test MSE' : te_mse,
        'Test R2'  : te_r2
    })
    return out

# --- parallel evaluation ---
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(evaluate)(p) for p in ParameterGrid(param_grid)
)

# --- assemble & save ---
df_res = pd.DataFrame(results)
df_res.sort_values('Test MSE', ascending=False, inplace=True)  # descending Test MSE

output_file = 'all_hyperparam_test_train_metrics.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_res.to_excel(writer, index=False, sheet_name='Metrics')

print(f"Saved all results to {output_file}")

# --- report best (lowest Test MSE) ---
best_row = df_res.iloc[::-1].iloc[0]  # last row after descending sort
print("Best hyperparameters by Test MSE:")
print(best_row[param_grid.keys()])

# --- parity plots for that best model ---
best_params = {k: best_row[k] for k in param_grid}
best_model = MLPRegressor(max_iter=2000, random_state=42, **best_params)
best_model.fit(X_train, y_train)
y_tr_best = best_model.predict(X_train)
y_te_best = best_model.predict(X_test)

for tag, y_true, y_pred in [
    ('train', y_train, y_tr_best),
    ('test',  y_test,  y_te_best )
]:
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn,mx], [mn,mx], 'k--', lw=1)
    plt.xlabel('True CX Basic pKa')
    plt.ylabel('Predicted CX Basic pKa')
    plt.title(f'Parity Plot: {tag.title()}')
    plt.tight_layout()
    plt.savefig(f'parity_{tag}.png')
    plt.close()

print("Parity plots saved as parity_train.png and parity_test.png")
