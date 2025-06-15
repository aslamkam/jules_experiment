import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import sys

def load_sigma_profile(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t', header=None)
        return df[1].values
    except Exception:
        return None


def print_detailed_error_stats(y_true, y_pred, set_name):
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    print(f"\nDetailed Error Statistics for {set_name} Set:")
    if errors.size == 0:
        print("No data to calculate statistics.")
        return

    print(f"Max Abs Error: {np.max(abs_errors):.4f}")
    for threshold in [0.2, 0.4]:
        within = abs_errors <= threshold
        print(f"% |Err|<= {threshold}: {within.mean() * 100:.2f}%")
        pos = (errors > 0) & (errors <= threshold)
        neg = (errors < 0) & (errors >= -threshold)
        print(f"% Err in (0, {threshold}]: {pos.mean() * 100:.2f}%")
        print(f"% Err in (-{threshold}, 0): {neg.mean() * 100:.2f}%")


def main():
    # Paths (update if necessary)
    dataset_path = '../Features/Chembl-12C/ChEMBL_amines_12C.csv'
    sigma_profiles_path = '../Features/Chembl-12C/Orca-Sigma-Profile/ChEMBL_12C_SigmaProfiles_Orca-5899'

    # Load data
    amines_df = pd.read_csv(dataset_path)
    amines_df.rename(columns={'Smiles': 'SMILES'}, inplace=True)
    amines_df = amines_df[['ChEMBL ID', 'CX Basic pKa', 'Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']]

    # Load sigma profiles
    profiles, ids = [], []
    for _, row in amines_df.iterrows():
        path = os.path.join(sigma_profiles_path, f"{row['Inchi Key']}.txt")
        prof = load_sigma_profile(path)
        if prof is not None:
            profiles.append(prof)
            ids.append(row['ChEMBL ID'])

    sigma_array = np.vstack(profiles)
    col_names = [f'sigma_value_{i}' for i in range(sigma_array.shape[1])]
    sigma_df = pd.DataFrame(sigma_array, columns=col_names)
    sigma_df['ChEMBL ID'] = ids

    # Merge and clean
    merged = pd.merge(amines_df, sigma_df, on='ChEMBL ID')
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.dropna(inplace=True)

    X = merged[col_names].values
    y = merged['CX Basic pKa'].values
    metadata = merged[['Molecular Formula', 'Amine Class', 'SMILES', 'Inchi Key']]

    # Splits
    X_tr_val, X_test, y_tr_val, y_test, meta_tr_val, meta_test = train_test_split(
        X, y, metadata, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=0.2, random_state=42
    )

    # Hyperparameter search
    param_dist = {
        'svr__C': uniform(1, 100),
        'svr__epsilon': uniform(0.01, 0.1),
        'svr__gamma': ['scale', 'auto'] + list(uniform(0.01, 0.5).rvs(10)),
        'svr__kernel': ['rbf']
    }
    pipeline = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist,
        n_iter=50, scoring='neg_mean_squared_error', cv=5,
        verbose=2, random_state=42, n_jobs=-1
    )

    print("Starting hyperparameter tuning...")
    search.fit(X_train, y_train)

    # Report
    best = search.best_estimator_
    print("\nBest Parameters:")
    for k, v in search.best_params_.items():
        print(f"{k.replace('svr__', '')}: {v}")
    print(f"Best CV MSE: {-search.best_score_:.4f}")

    # Predictions
    for name, (X_, y_) in [('Train', (X_train, y_train)), ('Validation', (X_val, y_val)), ('Test', (X_test, y_test))]:
        y_pred = best.predict(X_)
        print(f"\n{name} Metrics:")
        print(f"MSE: {mean_squared_error(y_, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_, y_pred)):.4f}")
        print(f"MAE: {mean_absolute_error(y_, y_pred):.4f}")
        print(f"R^2: {r2_score(y_, y_pred):.4f}")
        if name != 'Train':
            print_detailed_error_stats(y_, y_pred, name)

        # Parity plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_, y_pred, alpha=0.5)
        plt.plot([y_.min(), y_.max()],[y_.min(), y_.max()], 'k--')
        plt.xlabel(f'Actual pKa ({name})')
        plt.ylabel(f'Predicted pKa ({name})')
        plt.title(f'{name} Set Parity')
        plt.savefig(f'parity_{name.lower()}.png')
        plt.close()

    # Save predictions
    meta_test = meta_test.reset_index(drop=True)
    preds = pd.DataFrame({
        'Formula': meta_test['Molecular Formula'],
        'Class': meta_test['Amine Class'],
        'SMILES': meta_test['SMILES'],
        'InchiKey': meta_test['Inchi Key'],
        'Actual_pKa': y_test,
        'Predicted_pKa': best.predict(X_test)
    })
    preds.to_csv('pka_predictions.csv', index=False)
    print("Predictions saved.")

    # Save model and params
    joblib.dump(best, 'best_svr_model.joblib')
    pd.DataFrame([{
        k.replace('svr__', ''): v for k, v in search.best_params_.items()
    }]).to_csv('best_svr_params.csv', index=False)
    print("Model and parameters saved.")

if __name__ == '__main__':
    # Optional redirection of stdout
    if len(sys.argv) > 1 and sys.argv[1] == '--to-file':
        out_path = 'output.txt'
        with open(out_path, 'w') as f:
            sys.stdout = f
            main()
            sys.stdout = sys.__stdout__
        print(f"Output written to {out_path}")
    else:
        main()
