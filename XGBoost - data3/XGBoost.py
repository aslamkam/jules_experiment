#!/usr/bin/env python3
"""
Train an XGBoost regressor on Benson group count features and sigma profiles
 to predict pKa values.
"""
import os
import re
import ast
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
    data_file = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data3_12C_Chembl_Benson_Groups_Sigma_Profile\Amines_12C_CHEMBL_with_sigma_cleaned.xlsx"
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
        rmse = mean_squared_error(y_, preds, squared=False)
        mae = mean_absolute_error(y_, preds)
        r2 = r2_score(y_, preds)
        print(f"{name} set -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f}")

if __name__ == '__main__':
    main()
