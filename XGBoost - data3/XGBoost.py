#!/usr/bin/env python3
"""
Train an XGBoost regressor on Benson group count features and sigma profiles
 to predict pKa values.
"""
import os
import re
import ast
import sys # Import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# -------------------------------
# Benson group parsing & vectorization
# -------------------------------

def parse_benson_groups(text):
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

# -------------------------------
# Pipeline: load, preprocess, train & evaluate
# -------------------------------

def main():
    # Path to data file (update as needed)
    data_file = "../data3_12C_Chembl_Benson_Groups_Sigma_Profile/Amines_12C_CHEMBL_with_sigma_cleaned.xlsx"
    df = pd.read_excel(data_file)

    # Parse Benson group dictionaries
    df['benson_dict'] = df['benson_groups'].apply(parse_benson_groups)
    # Parse sigma profiles (assumes a single nested list per entry)
    df['sigma_list'] = df['sigma_profile'].apply(
        lambda x: ast.literal_eval(x)[0]
        if isinstance(x, str)
        else (x[0] if isinstance(x, list) and len(x) == 1 else x)
    )

    # Vectorize Benson features
    vec = DictVectorizer(sparse=False)
    X_benson = vec.fit_transform(df['benson_dict'])
    # Build sigma feature matrix
    sigma_matrix = np.vstack(df['sigma_list'].apply(lambda arr: np.array(arr)).values)

    # Combine tabular features
    X = np.hstack([X_benson, sigma_matrix])
    y = df['pka_value'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize XGBoost regressor
    xgb = XGBRegressor(
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

    # Train model
    xgb.fit(X_train, y_train)

    # Evaluate performance
    for name, X_, y_ in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        preds = xgb.predict(X_)
        # Calculate all metrics
        mse = mean_squared_error(y_, preds)
        rmse = mean_squared_error(y_, preds, squared=False) # Equivalent to np.sqrt(mse)
        mae = mean_absolute_error(y_, preds)
        r2 = r2_score(y_, preds)

        print(f"\n{name} Set Metrics:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  MSE:  {mse:.3f}")
        print(f"  RMSE: {rmse:.3f}") # RMSE after MSE for consistency with example
        print(f"  R2:   {r2:.3f}")

        # --- Generate and Save Plots ---
        # Parity Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_, preds, alpha=0.5, label=f'{name} Data')
        min_val = min(min(y_), min(preds)) if len(y_) > 0 and len(preds) > 0 else 0
        max_val = max(max(y_), max(preds)) if len(y_) > 0 and len(preds) > 0 else 1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y=x')
        plt.xlabel('True pKa')
        plt.ylabel('Predicted pKa')
        plt.title(f'{name} Set: Predicted vs True pKa')
        plt.legend()
        plt.grid(True)
        parity_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'parity_plot_{name.lower()}.png')
        plt.savefig(parity_plot_path)
        plt.close()
        print(f"  Saved parity plot to {parity_plot_path}")

        # Error Distribution Plot
        current_errors = preds - y_
        plt.figure(figsize=(8, 6))
        plt.hist(current_errors, bins=20, alpha=0.7, label=f'{name} Errors')
        plt.xlabel('Error (Predicted - True pKa)')
        plt.ylabel('Count')
        plt.title(f'{name} Set: Error Distribution')
        plt.legend()
        plt.grid(True)
        error_dist_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'error_dist_{name.lower()}.png')
        plt.savefig(error_dist_plot_path)
        plt.close()
        print(f"  Saved error distribution plot to {error_dist_plot_path}")

        if name == 'Test':
            # Define errors as (predicted - true) for consistency with example for pct metrics
            errors = preds - y_
            abs_errors = np.abs(errors)

            max_abs_err = np.max(abs_errors) if len(abs_errors) > 0 else 0.0

            # Calculate percentage metrics using the 'errors' defined as (predicted - true)
            pct_abs_err_le_02 = (np.sum(abs_errors <= 0.2) / len(abs_errors) * 100) if len(abs_errors) > 0 else 0.0
            pct_err_in_0_02 = (np.sum((errors > 0) & (errors <= 0.2)) / len(errors) * 100) if len(errors) > 0 else 0.0
            pct_err_in_neg02_0 = (np.sum((errors < 0) & (errors > -0.2)) / len(errors) * 100) if len(errors) > 0 else 0.0 # Match (-0.2, 0)

            pct_abs_err_le_04 = (np.sum(abs_errors <= 0.4) / len(abs_errors) * 100) if len(abs_errors) > 0 else 0.0
            pct_err_in_0_04 = (np.sum((errors > 0) & (errors <= 0.4)) / len(errors) * 100) if len(errors) > 0 else 0.0
            pct_err_in_neg04_0 = (np.sum((errors < 0) & (errors > -0.4)) / len(errors) * 100) if len(errors) > 0 else 0.0 # Match (-0.4, 0)

            print(f"  Max Abs Error:      {max_abs_err:.3f}")
            print(f"  % |Err| <= 0.2:     {pct_abs_err_le_02:.3f}%")
            print(f"  % Err in (0,0.2]:   {pct_err_in_0_02:.3f}%")
            print(f"  % Err in (-0.2,0):  {pct_err_in_neg02_0:.3f}%")
            print(f"  % |Err| <= 0.4:     {pct_abs_err_le_04:.3f}%")
            print(f"  % Err in (0,0.4]:   {pct_err_in_0_04:.3f}%")
            print(f"  % Err in (-0.4,0):  {pct_err_in_neg04_0:.3f}%")

if __name__ == '__main__':
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the output file path
    output_file_path = os.path.join(script_dir, "output.txt")

    # Store the original stdout
    original_stdout = sys.stdout

    try:
        # Redirect stdout to the output file
        sys.stdout = open(output_file_path, 'w')
        print(f"Outputting to: {output_file_path}") # Optional: confirm redirection

        main() # Call the main function where all operations and prints happen

    except Exception as e:
        # If an error occurs, restore stdout to print the error to console
        if sys.stdout.name == output_file_path:
            sys.stdout.close()
        sys.stdout = original_stdout
        print(f"An error occurred: {e}", file=sys.stderr)
        # Optionally, log the exception to the output file as well before re-raising
        with open(output_file_path, 'a') as f_err:
            import traceback
            f_err.write("\n--- ERROR DURING EXECUTION ---\n")
            traceback.print_exc(file=f_err)
        raise # Re-raise the exception to ensure the script exits with an error status

    finally:
        # Ensure stdout is restored
        if sys.stdout.name == output_file_path: # Check if stdout is our file
            sys.stdout.close()
        sys.stdout = original_stdout
        # This message will go to the console
        print(f"Output redirection finished. Results are in {output_file_path}")
