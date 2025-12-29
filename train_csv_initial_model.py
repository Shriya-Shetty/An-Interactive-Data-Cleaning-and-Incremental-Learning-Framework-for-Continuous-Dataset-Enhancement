import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ---------------- CONFIG ----------------
CSV_PATH = r"freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv"
TARGET_COL = "annual_income_usd"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = "freelancer_earnings_model.pkl"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)

# ---------------- CLEAN TARGET COLUMN ----------------
df[TARGET_COL] = (
    df[TARGET_COL]
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

# ---------------- DROP ID COLUMN ----------------
if "freelancer_id" in df.columns:
    df = df.drop(columns=["freelancer_id"])

# ---------------- FEATURE / TARGET SPLIT ----------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ---------------- COLUMN TYPES ----------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("Numeric columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))

# ---------------- PREPROCESSING PIPELINE ----------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# ---------------- MODEL ----------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ---------------- TRAIN MODEL ----------------
pipeline.fit(X_train, y_train)
print("Model training completed.")

# ---------------- EVALUATION ----------------
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

# ---------------- SAVE MODEL ----------------
joblib.dump(pipeline, MODEL_PATH)
print("Model saved as:", MODEL_PATH)
