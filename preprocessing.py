"""
DATA_PREPROCESSING_FULL.py
A comprehensive data cleaning & preprocessing pipeline (CSV dataset)

Features:
1) Load dataset
2) Inspect & summarize
3) Handle missing values
4) Encode categorical data
5) Scale/normalize numeric features
6) Detect & cap outliers
7) Save cleaned data to CSV
8) Optional: basic EDA visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# -------- USER CONFIGURATION --------
CSV_FILE = r"freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv"       # change to your dataset
TARGET_COL = None           # set target column if present
SAVE_PATH = "cleaned_data.csv"
DO_EDA = True               # whether to do basic plots
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------- LOAD DATA --------
df = pd.read_csv(CSV_FILE)
print("\n=== Initial Data Info ===")
print(df.info())
print("\n=== Initial Data Head ===")
print(df.head())

# -------- BASIC SUMMARY --------
print("\n=== Missing Values ===")
print(df.isnull().sum())
print("\n=== Summary Stats ===")
print(df.describe(include="all"))

# -------- BASIC CLEANING --------
# drop duplicates
df = df.drop_duplicates()
print("\nDuplicates removed:", df.shape)

# drop columns with too many missing values
thresh = int(0.6 * len(df))
df = df.dropna(axis=1, thresh=thresh)

# -------- SEPARATE FEATURES --------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if TARGET_COL and TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)
if TARGET_COL and TARGET_COL in cat_cols:
    cat_cols.remove(TARGET_COL)

print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# -------- PREPROCESSING PIPELINE --------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

# -------- OPTIONAL: OUTLIER CAPPING (IQR) --------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# -------- SPLIT & TRANSFORM --------
if TARGET_COL:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
else:
    X = df
    y = None

print("\nApplying preprocess pipeline...")
X_processed = preprocess.fit_transform(X)

print("\nProcessed shape:", X_processed.shape)

# -------- SAVE CLEAN DATA --------
clean_df = pd.DataFrame(X_processed)
clean_df.to_csv(SAVE_PATH, index=False)

print("\nCleaned data saved to:", SAVE_PATH)

# -------- OPTIONAL EDA --------
if DO_EDA:
    print("\nGenerating EDA plots...")
    # numeric histograms
    clean_num = clean_df.select_dtypes(include=[np.number])
    clean_num.hist(bins=20, figsize=(12, 10))
    plt.suptitle("Numeric Feature Distributions")
    plt.show()

    # correlation heatmap
    corr = clean_num.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

print("\nPreprocessing Completed Successfully!")
