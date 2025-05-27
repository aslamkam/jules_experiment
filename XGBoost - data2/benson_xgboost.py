#!/usr/bin/env python3
"""
Train an XGBoost regressor on pKa values using Benson group counts as features.
Trains using fixed best hyperparameters and prints detailed error statistics on the test set.
"""

import pandas as pd
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
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


def main():
    # Load data from Excel
    df = pd.read_excel(
        r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa_and_removed_ccus.xlsx"
    )

    # Parse the benson_groups column into actual dicts
    df['features'] = df['benson_groups'].apply(parse_benson_groups)

    # Vectorize the group count dictionaries into a feature matrix
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(df['features'])
    y = df['pka_value'].values

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and train the XGBoost regressor with best hyperparameters
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

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluation metrics
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print basic metrics
    print(f"Train RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, MSE: {train_mse:.3f}, R2: {train_r2:.3f}")
    print(f"Test  RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, MSE: {test_mse:.3f}, R2: {test_r2:.3f}\n")

    # Detailed error statistics on test set
    errors = y_test_pred - y_test
    abs_errors = np.abs(errors)
    max_abs_error = abs_errors.max()

    # Thresholds
    for thresh in [0.2, 0.4]:
        pct_le = np.mean(abs_errors <= thresh) * 100
        pct_pos = np.mean((errors > 0) & (errors <= thresh)) * 100
        pct_neg = np.mean((errors < 0) & (errors >= -thresh)) * 100

        print(f"Max Abs Error: {max_abs_error:.3f}")
        print(f"% with Abs Error <={thresh}: {pct_le:.2f}%")
        print(f"% with Error in (0,{thresh}]: {pct_pos:.2f}%")
        print(f"% with Error in ( -{thresh},0 ): {pct_neg:.2f}%\n")

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.7)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
    plt.xlabel('Actual pKa')
    plt.ylabel('Predicted pKa')
    plt.title('Train: Actual vs Predicted')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual pKa')
    plt.ylabel('Predicted pKa')
    plt.title('Test: Actual vs Predicted')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
