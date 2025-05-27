import pandas as pd
import ast
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, max_error
)
from xgboost import XGBRegressor

# ----------------------------
# Helper functions
# ----------------------------

def clean_benson_group(val):
    try:
        if isinstance(val, str) and val.startswith("defaultdict("):
            match = re.search(r"defaultdict\(.*?,\s*(\{.*\})\)", val)
            if match:
                dict_str = match.group(1)
                return ast.literal_eval(dict_str)
        elif isinstance(val, dict):
            return val
        return ast.literal_eval(val)
    except Exception as e:
        print(f"Error parsing: {val}\n{e}")
        return {}

def parse_pka_value(val):
    try:
        if isinstance(val, str) and 'to' in val:
            parts = val.split('to')
            nums = [float(p.strip()) for p in parts]
            return sum(nums) / len(nums)
        return float(val)
    except Exception as e:
        print(f"Could not parse pKa value: {val}\n{e}")
        return None

def print_metrics(y_true, y_pred, label=""):
    abs_error = np.abs(y_true - y_pred)
    error = y_pred - y_true

    print(f"{label} RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    print(f"{label} MSE: {mean_squared_error(y_true, y_pred):.3f}")
    print(f"{label} MAE: {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"{label} R^2: {r2_score(y_true, y_pred):.3f}")
    print(f"{label} Max Abs Error: {max_error(y_true, y_pred):.3f}")

    # Custom error stats
    within_0_2 = (abs_error <= 0.2).mean() * 100
    in_0_2 = ((error > 0) & (error <= 0.2)).mean() * 100
    in_neg_0_2 = ((error < 0) & (error >= -0.2)).mean() * 100
    within_0_4 = (abs_error <= 0.4).mean() * 100
    in_0_4 = ((error > 0) & (error <= 0.4)).mean() * 100
    in_neg_0_4 = ((error < 0) & (error >= -0.4)).mean() * 100

    print(f"{label} % with Abs Error ≤ 0.2: {within_0_2:.2f}%")
    print(f"{label} % with Error in (0, 0.2]: {in_0_2:.2f}%")
    print(f"{label} % with Error in (-0.2, 0): {in_neg_0_2:.2f}%")
    print(f"{label} % with Abs Error ≤ 0.4: {within_0_4:.2f}%")
    print(f"{label} % with Error in (0, 0.4]: {in_0_4:.2f}%")
    print(f"{label} % with Error in (-0.4, 0): {in_neg_0_4:.2f}%")
    print()

# ----------------------------
# Main Script
# ----------------------------

# Load dataset
file_path = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data5_IuPac_Benson_Groups\Filtered_IuPac_benson.xlsx"
df = pd.read_excel(file_path)

# Preprocess
df['benson_groups'] = df['benson_groups'].apply(clean_benson_group)
df = df.dropna(subset=['pka_value'])
df['pka_value'] = df['pka_value'].apply(parse_pka_value)
df = df.dropna(subset=['pka_value'])

vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(df['benson_groups'])
y = df['pka_value'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using best params
best_model = XGBRegressor(
    objective='reg:squarederror',
    subsample=0.8,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

best_model.fit(X_train, y_train)

# Evaluate
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

print_metrics(y_train, y_train_pred, label="Train")
print_metrics(y_test, y_test_pred, label="Test")
