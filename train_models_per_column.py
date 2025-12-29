import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ---------------- CONFIG ----------------
DATASET_PATH = r"freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv"
DROP_COLS = ["freelancer_id"]
RANDOM_STATE = 42

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATASET_PATH)

# Clean annual income column
df["annual_income_usd"] = (
    df["annual_income_usd"]
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

# Drop ID column
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

print("Dataset shape:", df.shape)

# Identify column types
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# ---------------- TRAIN ONE MODEL PER COLUMN ----------------
for target in df.columns:
    print(f"\nTraining model for column: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    # Decide task type
    if target in numeric_cols:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        task_type = "regression"
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        task_type = "classification"

        # Encode target & SAVE encoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, f"encoder_{target}.pkl")

    # Column types for inputs
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(pipeline, f"model_{target}.pkl")

    print(f"Saved model_{target}.pkl ({task_type})")

print("\n✅ All column-wise models and encoders trained successfully.")
