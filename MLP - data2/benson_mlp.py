#!/usr/bin/env python3
"""
Train an MLP regressor on pKa values using Benson group counts as features.
"""
import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def parse_benson_groups(text):
    """
    Extract the dictionary literal from a string representation of a defaultdict.
    """
    match = re.match(r".*defaultdict\(<class 'int'>, (.*)\)", text)
    dict_str = match.group(1) if match else text
    return ast.literal_eval(dict_str)

def main():
    # Load data from Excel
    df = pd.read_excel(r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data2_12C_Chembl_Benson_Groups\Amines_12C_CHEMBL_benson_matched_with_pKa.xlsx")

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

    # Scale features for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the MLP regressor with best hyperparameters
    mlp = MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),
        activation='tanh',
        solver='adam',
        alpha=0.2,
        learning_rate_init=0.005,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = mlp.predict(X_train_scaled)
    y_test_pred = mlp.predict(X_test_scaled)

    # Evaluation metrics
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Train RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, MSE: {train_mse:.3f}, R2: {train_r2:.3f}")
    print(f"Test  RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, MSE: {test_mse:.3f}, R2: {test_r2:.3f}")

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
