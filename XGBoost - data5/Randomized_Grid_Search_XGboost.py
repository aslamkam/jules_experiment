import pandas as pd
import ast
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import re

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

# Load dataset
file_path = r"C:\Users\kamal\OneDrive - University of Guelph\My Research\data5_IuPac_Benson_Groups\Filtered_IuPac_benson.xlsx"
df = pd.read_excel(file_path)

# Clean and parse columns
df['benson_groups'] = df['benson_groups'].apply(clean_benson_group)
df = df.dropna(subset=['pka_value'])
df['pka_value'] = df['pka_value'].apply(parse_pka_value)
df = df.dropna(subset=['pka_value'])

# Vectorize features
vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(df['benson_groups'])
y = df['pka_value'].astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model and param distribution for randomized search
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Run search
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Metrics
def print_metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} RMSE: {rmse:.3f}")
    print(f"{label} MSE: {mse:.3f}")
    print(f"{label} MAE: {mae:.3f}")
    print(f"{label} R^2: {r2:.3f}")
    print()

print(f"Best Params: {random_search.best_params_}\n")

print_metrics(y_train, y_train_pred, "Train")
print_metrics(y_test, y_test_pred, "Test")
