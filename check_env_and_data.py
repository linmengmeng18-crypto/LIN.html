import os
import sys

print("Python version:", sys.version)

packages = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
    'sklearn', 'statsmodels', 'xgboost', 'shap'
]

print("\n--- Package Versions ---")
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")
    except ImportError as e:
        print(f"{pkg}: NOT installed ({e})")

print("\n--- Checking Data File ---")
possible_paths = [
    "P3_driver_attrition.csv",
    os.path.join("data", "P3_driver_attrition.csv")
]

csv_path = None
for p in possible_paths:
    if os.path.exists(p):
        csv_path = p
        break

if not csv_path:
    print("ERROR: CSV file not found in any expected location!")
else:
    print(f"Found CSV at: {csv_path} (Size: {os.path.getsize(csv_path)} bytes)")
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f" - {col}: {dtype} | nulls: {df[col].isnull().sum()} | unique: {df[col].nunique()}")
    
    print("\nHead 3 rows:")
    print(df.head(3))

    if 'attrition_6m' in df.columns:
        print("\nTarget variable (attrition_6m) distribution:")
        print(df['attrition_6m'].value_counts(dropna=False))
        print(df['attrition_6m'].value_counts(normalize=True, dropna=False))
    else:
        print("\nWARNING: 'attrition_6m' column NOT found!")

    if 'driver_id' in df.columns:
        print(f"\nUnique driver_id count: {df['driver_id'].nunique()}, duplicates: {df['driver_id'].duplicated().sum()}")
